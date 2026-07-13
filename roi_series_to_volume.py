#!/usr/bin/env python3
"""
roi_series_to_volume.py — build a label volume from a Horos .rois_series
-------------------------------------------------------------------------
Drop-in alternative to build_volume.py that reads the *vector* ROI export
(Horos "Save ROIs" → .rois_series) instead of detecting coloured crosshairs in
Secondary-Capture pixels. The vector path is immune to zoom / "as displayed" /
colour / JPEG problems and recovers small tumours the Hough detector misses.

Each RECIST tumour is stored as two crossing line ROIs (long + short axis).
This pairs the lines per slice, models each tumour as an ellipse (longer arm =
major axis, shorter = minor axis — identical to extracter.py), and writes a
3-D label in the raw MR space so it feeds straight into prepare_sam2.py.

Slice alignment
---------------
The .rois_series outer array is indexed by image position; the raw series is
read in SimpleITK GDCM (z-sorted) order. Both sort by slice position, so array
index i maps to raw slice i. Use --flip-z if the two happen to run opposite
ways, and --compare to check against an existing label.

Usage
-----
    python roi_series_to_volume.py \\
        --rois    brac46551b.rois_series \\
        --raw-dir brac46551b_raw/ \\
        --output  label_vec.nrrd
    # optional: verify alignment against the colour-detected label
    python roi_series_to_volume.py ... --compare label.nrrd
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from typedstream_roi import parse_rois_series
from extracter import ellipse_mask, seg_len, seg_angle, infinite_intersection
from build_volume import read_dicom_series


# ─────────────────────────── crosshair geometry ──────────────────────────

def _midpoint(line):
    (x1, y1), (x2, y2) = line
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pair_lines(lines):
    """Greedily pair lines by nearest midpoint (RECIST long/short axes of one
    tumour share a centre). Returns (pairs, singles) as index lists."""
    mids = [_midpoint(l) for l in lines]
    n = len(lines)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(mids[i][0] - mids[j][0], mids[i][1] - mids[j][1]))
            cand.append((d, i, j))
    cand.sort()
    used = [False] * n
    pairs = []
    for d, i, j in cand:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        pairs.append((i, j, d))
    singles = [i for i in range(n) if not used[i]]
    return pairs, singles


def crosshair_from_pair(l1, l2):
    """Return (cx, cy, major_len, minor_len, angle_deg) for a tumour."""
    s1 = (l1[0][0], l1[0][1], l1[1][0], l1[1][1])
    s2 = (l2[0][0], l2[0][1], l2[1][0], l2[1][1])
    L1, L2 = seg_len(*s1), seg_len(*s2)
    major, mmaj, mmin = (s1, L1, L2) if L1 >= L2 else (s2, L2, L1)
    pt = infinite_intersection(s1, s2)
    if pt is None:                       # near-parallel: fall back to centroid
        pt = ((s1[0] + s1[2] + s2[0] + s2[2]) / 4.0,
              (s1[1] + s1[3] + s2[1] + s2[3]) / 4.0)
    return pt[0], pt[1], mmaj, mmin, seg_angle(*major)


def single_ellipse(line):
    """Degenerate tumour from a lone line: thin ellipse along the line."""
    s = (line[0][0], line[0][1], line[1][0], line[1][1])
    L = seg_len(*s)
    cx, cy = _midpoint(line)
    return cx, cy, L, max(2.0, 0.25 * L), seg_angle(*s)


# ─────────────────────────── QC rendering ────────────────────────────────

def _window(raw_vol):
    """Global 1–99th percentile window → per-slice 8-bit (matches prepare_sam2)."""
    flat = raw_vol.astype(np.float32).ravel()
    lo, hi = float(np.percentile(flat, 1)), float(np.percentile(flat, 99))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _save_slice_png(out_dir, z, gray8, lines, ellipses, mask, inst=None, scale=2):
    """One QC overlay: MR (grey) + filled mask (red) + axis lines (green) +
    ellipse outlines (yellow)."""
    bg = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    ov = bg.copy()
    ov[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(bg, 0.6, ov, 0.4, 0)
    for (a, b) in lines:
        cv2.line(out, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), (0, 255, 0), 1)
    for cx, cy, maj, mnr, ang in ellipses:
        cv2.ellipse(out, (int(round(cx)), int(round(cy))),
                    (max(1, int(round(maj / 2))), max(1, int(round(mnr / 2)))),
                    ang, 0, 360, (0, 255, 255), 1)
    if scale != 1:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    tag = f"z={z}" + (f" inst={inst}" if inst is not None else "") + \
          f"  {len(ellipses)} tumour(s)"
    cv2.putText(out, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"slice_{z:03d}.png"), out)


# ─────────────────────────── volume builder ──────────────────────────────

def _square_correct(slices, out_h, out_w, square_size=None):
    """OsiriX stores ROI points in the square *acquisition* grid (e.g. 256×256).
    When the image is reconstructed with a reduced phase FOV it becomes
    Rows<Cols (here 226×256, square pixels), so the stored coords must be
    scaled per axis from that square grid onto the reconstructed pixel grid:
        x_px = x_roi · Cols/M ,  y_px = y_roi · Rows/M ,   M = acquisition size.
    M defaults to max(Rows, Cols) (the full frequency matrix)."""
    M = square_size or max(out_h, out_w)
    sx, sy = out_w / M, out_h / M
    if abs(sx - 1) < 1e-9 and abs(sy - 1) < 1e-9:
        return slices, (M, sx, sy)
    fixed = [[[(x * sx, y * sy) for (x, y) in roi] for roi in slc]
             for slc in slices]
    return fixed, (M, sx, sy)


def build(rois_path, raw_dir, output_path, flip_z=False,
          pair_warn=25.0, compare_path=None, slice_results=None,
          square_correct=True, square_size=None):
    ref_image, slice_map = read_dicom_series(raw_dir)
    n_slices = ref_image.GetSize()[2]
    out_w, out_h = ref_image.GetSize()[:2]
    spacing, origin, direction = (ref_image.GetSpacing(),
                                  ref_image.GetOrigin(),
                                  ref_image.GetDirection())
    print(f"Raw MR: {n_slices} slices  {out_w}×{out_h} px  "
          f"spacing xy={spacing[0]:.5f} z={spacing[2]:.4f} mm")

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

        m = np.zeros((out_h, out_w), dtype=np.uint8)
        for cx, cy, maj, mnr, ang in ell:
            m |= ellipse_mask((out_h, out_w), cx, cy, maj, mnr, ang)
        vol[z] = (m > 0).astype(np.uint8)

        if slice_results is not None:
            gray = np.clip((raw_vol[z].astype(np.float32) - win_lo) /
                           (win_hi - win_lo) * 255, 0, 255).astype(np.uint8)
            _save_slice_png(slice_results, z, gray, lines, ell, vol[z],
                            inst=inst_of.get(z))

        print(f"  slice {z:3d}: {len(pairs)} tumour(s)" +
              (f" +{len(singles)} lone line(s)" if singles else "") +
              f"  ({int(vol[z].sum())} px)")

    print(f"\nTumours: {n_tumours}   lone lines: {n_singles}", end="")
    if n_far:
        print(f"   [WARN] {n_far} pair(s) had far-apart midpoints "
              f"(>{pair_warn:.0f}px) — possible mispair", end="")
    print()

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


def _compare(vol, other_path):
    other = (sitk.GetArrayFromImage(sitk.ReadImage(str(other_path))) > 0).astype(np.uint8)
    print(f"\nCompare vs {other_path.name}:")
    if other.shape != vol.shape:
        print(f"  [WARN] shape mismatch {other.shape} vs {vol.shape}")
        return
    inter = int(np.logical_and(vol, other).sum())
    union = int(np.logical_or(vol, other).sum())
    iou = inter / union if union else 1.0
    print(f"  volume IoU: {iou:.3f}   "
          f"(this={int(vol.sum())}px, other={int(other.sum())}px)")
    a = {i for i in range(vol.shape[0]) if vol[i].any()}
    b = {i for i in range(other.shape[0]) if other[i].any()}
    only_v, only_o = sorted(a - b), sorted(b - a)
    print(f"  annotated slices: this={len(a)}  other={len(b)}  shared={len(a & b)}")
    if only_v:
        print(f"  only in this : {only_v}")
    if only_o:
        print(f"  only in other: {only_o}")


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build a label volume from a Horos .rois_series vector export.")
    ap.add_argument("--rois", required=True, help="Path to the .rois_series file.")
    ap.add_argument("--raw-dir", required=True,
                    help="Raw MR DICOM folder (defines geometry + slice order).")
    ap.add_argument("--output", "-o", default="label_vec.nrrd",
                    help="Output NRRD path (default: label_vec.nrrd).")
    ap.add_argument("--flip-z", action="store_true",
                    help="Reverse slice order (if ROI index runs opposite to GDCM).")
    ap.add_argument("--pair-warn", type=float, default=25.0,
                    help="Warn if paired axis midpoints are >this many px apart.")
    ap.add_argument("--compare", default=None,
                    help="Existing label (NRRD/NIfTI) to report IoU / slice overlap.")
    ap.add_argument("--slice-results", default=None,
                    help="Directory to save per-slice QC overlay PNGs "
                         "(MR + axis lines + ellipse outlines + mask).")
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
    )
