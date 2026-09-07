#!/usr/bin/env python3
"""
roi_series_to_volume_split.py — .rois_series → label, separating touching tumours.

Purpose: like roi_series_to_volume.py, but rasterises the ellipses so tumours that
touch stay distinct connected components (a --gap-px gap between them), keeping
per-lesion counts and instance labels correct where lesions abut. --gap-px 0
reproduces roi_series_to_volume.py exactly.

Use:
    python roi_series_to_volume_split.py --rois brac46551b.rois_series \\
        --raw-dir brac46551b_raw/ --output label_split.nrrd   # --compare label_vec.nrrd
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from typedstream_roi import parse_rois_series
from extracter import ellipse_mask, seg_len, seg_angle, infinite_intersection
from build_volume import read_dicom_series
from roi_series_to_volume import (
    pair_lines, crosshair_from_pair, single_ellipse,
    _window, _save_slice_png, _square_correct, _compare,
)

_NB8 = np.ones((3, 3), dtype=bool)          # 8-neighbourhood, matches cv2 default


# ─────────────────────── separated rasterisation ─────────────────────────

def rasterise_separated(shape, ellipses, gap_px=1):
    """Rasterise ellipses so touching tumours stay separate components.

    Returns (labels, n_merged_before, n_lost): labels is int32 (1-based, 0 = bg);
    n_merged_before = fusions this call prevented; n_lost = ellipses left with no
    pixels (swallowed by a neighbour or eroded by the gap).
    """
    H, W = shape[:2]
    labels = np.zeros((H, W), dtype=np.int32)
    if not ellipses:
        return labels, 0, 0

    # Rasterise each ellipse and record how deep every pixel sits inside it,
    # normalised so a small tumour is not simply outvoted by a large one.
    masks, depth = [], np.zeros((len(ellipses), H, W), dtype=np.float32)
    for k, (cx, cy, maj, mnr, ang) in enumerate(ellipses):
        e = ellipse_mask((H, W), cx, cy, maj, mnr, ang) > 0
        masks.append(e)
        if e.any():
            d = cv2.distanceTransform(e.astype(np.uint8), cv2.DIST_L2, 5)
            m = float(d.max())
            depth[k] = d / m if m > 0 else d

    union = np.zeros((H, W), dtype=bool)
    for e in masks:
        union |= e
    if not union.any():
        return labels, 0, len(ellipses)

    # How many components would the plain OR have given? Anything below the
    # ellipse count is a fusion we are about to undo.
    n_or, _ = cv2.connectedComponents(union.astype(np.uint8))
    n_merged = max(0, len(ellipses) - (n_or - 1))

    # Winner-takes-all on normalised depth; ellipses that claim nothing keep id 0.
    labels[union] = (np.argmax(depth, axis=0)[union] + 1).astype(np.int32)

    # Carve the gap: clear pixels within gap_px of a differently-labelled pixel,
    # on both sides, so the separation survives 8-connectivity.
    if gap_px > 0:
        out = labels.copy()
        for k in np.unique(labels):
            if k == 0:
                continue
            others = (labels > 0) & (labels != k)
            if not others.any():
                continue
            grown = ndimage.binary_dilation(others, structure=_NB8,
                                            iterations=gap_px)
            out[(labels == k) & grown] = 0
        labels = out

    n_lost = sum(1 for k in range(1, len(ellipses) + 1) if not (labels == k).any())
    return labels, n_merged, n_lost


# ─────────────────────────── volume builder ──────────────────────────────

def build(rois_path, raw_dir, output_path, flip_z=False,
          pair_warn=25.0, compare_path=None, slice_results=None,
          square_correct=True, square_size=None, gap_px=1):
    ref_image, slice_map = read_dicom_series(raw_dir)
    n_slices = ref_image.GetSize()[2]
    out_w, out_h = ref_image.GetSize()[:2]
    spacing, origin, direction = (ref_image.GetSpacing(),
                                  ref_image.GetOrigin(),
                                  ref_image.GetDirection())
    print(f"Raw MR: {n_slices} slices  {out_w}×{out_h} px  "
          f"spacing xy={spacing[0]:.5f} z={spacing[2]:.4f} mm")
    print(f"Separation: {'gap of %d px between touching tumours' % gap_px}"
          if gap_px > 0 else "Separation: DISABLED (plain OR, as the original)")

    raw_vol = win_lo = win_hi = None
    inst_of = {}
    if slice_results is not None:
        slice_results.mkdir(parents=True, exist_ok=True)
        raw_vol = sitk.GetArrayFromImage(ref_image)        # (Z, H, W)
        win_lo, win_hi = _window(raw_vol)
        inst_of = {i: inst for i, inst, _ in slice_map}

    slices = parse_rois_series(rois_path)
    print(f"ROI file: {len(slices)} slots  "
          f"({sum(1 for s in slices if s)} annotated, "
          f"{sum(len(s) for s in slices)} line ROIs)")
    if square_correct:
        slices, (M, sx, sy) = _square_correct(slices, out_h, out_w, square_size)
        if abs(sy - 1) > 1e-9 or abs(sx - 1) > 1e-9:
            print(f"Square-grid correction: M={M}  scale x={sx:.4f} y={sy:.4f} "
                  f"(reconstructed {out_h}×{out_w} from {M}² acquisition)")
        else:
            print("Square-grid correction: none needed (image already square).")
    if len(slices) != n_slices:
        print(f"  [WARN] slot count {len(slices)} != raw slice count {n_slices}; "
              f"placing by index and ignoring overflow.")

    vol = np.zeros((n_slices, out_h, out_w), dtype=np.uint8)
    n_tumours = n_singles = n_far = 0
    n_sep_slices = n_sep_pairs = n_lost_total = 0
    px_or = px_sep = 0

    for arr_idx, lines in enumerate(slices):
        if arr_idx >= n_slices:
            break
        if not lines:
            continue
        z = (n_slices - 1 - arr_idx) if flip_z else arr_idx

        pairs, singles = pair_lines(lines)
        ell = []
        for i, j, d in pairs:
            if d > pair_warn:
                n_far += 1
            ell.append(crosshair_from_pair(lines[i], lines[j]))
        for i in singles:
            ell.append(single_ellipse(lines[i]))
        n_tumours += len(pairs)
        n_singles += len(singles)

        # Plain OR kept for the QC panel and the before/after pixel accounting.
        m_or = np.zeros((out_h, out_w), dtype=np.uint8)
        for cx, cy, maj, mnr, ang in ell:
            m_or |= ellipse_mask((out_h, out_w), cx, cy, maj, mnr, ang)
        px_or += int((m_or > 0).sum())

        labels, n_merged, n_lost = rasterise_separated((out_h, out_w), ell, gap_px)
        vol[z] = (labels > 0).astype(np.uint8)
        px_sep += int(vol[z].sum())
        if n_merged:
            n_sep_slices += 1
            n_sep_pairs += n_merged
        n_lost_total += n_lost

        n_cc, _ = cv2.connectedComponents(vol[z])
        note = ""
        if n_merged:
            note += f"  ← separated {n_merged} fused pair(s)"
        if n_lost:
            note += f"  [WARN] {n_lost} ellipse(s) lost entirely"
        if (n_cc - 1) != len(ell) and not n_lost:
            note += f"  [note] {n_cc - 1} component(s) for {len(ell)} ellipse(s)"

        if slice_results is not None:
            gray = np.clip((raw_vol[z].astype(np.float32) - win_lo) /
                           (win_hi - win_lo) * 255, 0, 255).astype(np.uint8)
            _save_slice_png(slice_results, z, gray, lines, ell, vol[z],
                            inst=inst_of.get(z))

        print(f"  slice {z:3d}: {len(pairs)} tumour(s)" +
              (f" +{len(singles)} lone line(s)" if singles else "") +
              f"  ({int(vol[z].sum())} px){note}")

    print(f"\nTumours: {n_tumours}   lone lines: {n_singles}", end="")
    if n_far:
        print(f"   [WARN] {n_far} pair(s) had far-apart midpoints "
              f"(>{pair_warn:.0f}px) — possible mispair", end="")
    print()
    if gap_px > 0:
        lost_px = px_or - px_sep
        print(f"Separation: {n_sep_pairs} fused pair(s) split across "
              f"{n_sep_slices} slice(s); "
              f"{lost_px} px removed as gap "
              f"({100.0 * lost_px / px_or:.2f}% of the OR volume)"
              if px_or else "Separation: nothing to split")
        if n_lost_total:
            print(f"  [WARN] {n_lost_total} ellipse(s) ended up with no pixels — "
                  f"fully contained in a neighbour, or eroded away by --gap-px")

    label = sitk.GetImageFromArray(vol)
    label.SetSpacing(spacing)
    label.SetOrigin(origin)
    label.SetDirection(direction)
    label.SetMetaData("intent_name", "label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(label, str(output_path))
    nii = output_path.with_suffix("").with_suffix(".nii.gz")
    sitk.WriteImage(label, str(nii))
    print(f"Saved  → {output_path}\n       → {nii}")
    if slice_results is not None:
        print(f"QC overlays → {slice_results}/")

    if compare_path is not None:
        _compare(vol, Path(compare_path))
    return vol


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build a label volume from a Horos .rois_series, keeping "
                    "touching tumours as separate connected components.")
    ap.add_argument("--rois", required=True, help="Path to the .rois_series file.")
    ap.add_argument("--raw-dir", required=True,
                    help="Raw MR DICOM folder (defines geometry + slice order).")
    ap.add_argument("--output", "-o", default="label_split.nrrd",
                    help="Output NRRD path (default: label_split.nrrd).")
    ap.add_argument("--gap-px", type=int, default=1,
                    help="Background gap carved between touching tumours, in "
                         "pixels removed from EACH side (default: 1, giving a "
                         "2px separation). cv2.connectedComponents uses "
                         "8-connectivity, so 0 reproduces the original OR "
                         "behaviour exactly and any value >=1 guarantees the "
                         "tumours stay distinct components.")
    ap.add_argument("--flip-z", action="store_true",
                    help="Reverse slice order (if ROI index runs opposite to GDCM).")
    ap.add_argument("--pair-warn", type=float, default=25.0,
                    help="Warn if paired axis midpoints are >this many px apart.")
    ap.add_argument("--compare", default=None,
                    help="Existing label (NRRD/NIfTI) to report IoU / slice overlap. "
                         "Point this at the roi_series_to_volume.py output to see "
                         "exactly what the separation changed.")
    ap.add_argument("--slice-results", default=None,
                    help="Directory to save per-slice QC comparison PNGs.")
    ap.add_argument("--no-square-correct", action="store_true",
                    help="Disable the square-acquisition→reconstructed-grid "
                         "aspect correction of ROI coordinates.")
    ap.add_argument("--square-size", type=int, default=None,
                    help="Override the acquisition matrix size M used for the "
                         "aspect correction (default: max(Rows, Cols)).")
    args = ap.parse_args()

    build(
        rois_path=Path(args.rois),
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        flip_z=args.flip_z,
        pair_warn=args.pair_warn,
        compare_path=args.compare,
        slice_results=Path(args.slice_results) if args.slice_results else None,
        square_correct=not args.no_square_correct,
        square_size=args.square_size,
        gap_px=args.gap_px,
    )
