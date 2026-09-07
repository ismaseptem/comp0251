#!/usr/bin/env python3
"""
ellipse_baseline.py — model-free geometric baseline (no SAM2).

Purpose: the pseudo-label at every depth is the rasterised shrinking seed ellipse
itself (the clip's ellipse used AS the mask, not as a bound), isolating what pure
RECIST geometry buys before any learned model. Per object: r_mm from the min-
enclosing circle, depth window D = max(1, ceil(r_mm/t)), pole P0 = max(r_mm/t,
pole_floor); at each d in [-D, +D], sigma(d) = sqrt(max(0, 1-(d/P0)^2))*clip_scale
rasterised onto slice z0+d, n_min-filtered per object, then OR-merged. Ellipse fit,
rasteriser and filters are copied verbatim from sam2_propagate_clip.py. Reads only
masks/ + the label (spatial metadata); no frames, no checkpoints, no GPU.

Use:
    python ellipse_baseline.py --input-dir sam2_input --label label.nii.gz \\
        --output propagated_label_ellipse.nrrd
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


# ─── Helpers copied verbatim from sam2_propagate_clip.py ──────────────────
# (kept identical so the baseline's ellipse fit + rasterisation match the clip's
#  exactly; duplicated rather than imported so this script pulls in no torch/SAM2)

def components_with_centroids(binary_mask: np.ndarray, min_px: int = 50):
    """[(component_mask, centroid_xy), ...] for each foreground component >= min_px."""
    n, label_map = cv2.connectedComponents(binary_mask.astype(np.uint8))
    results = []
    for lbl in range(1, n):
        comp = (label_map == lbl).astype(np.uint8)
        if int(comp.sum()) < min_px:
            continue
        M = cv2.moments(comp)
        if M["m00"] == 0:
            continue
        results.append((comp, (M["m10"] / M["m00"], M["m01"] / M["m00"])))
    return results


def load_seed_components(mask_path: Path) -> list[np.ndarray]:
    """Seed-frame mask PNG -> one bool array per connected component (min_px=20)."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return []
    binary = (raw > 127).astype(np.uint8)
    return [comp.astype(bool) for comp, _ in components_with_centroids(binary, min_px=20)]


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px pixels from a bool mask."""
    n_cc, cc_map = cv2.connectedComponents(mask.astype(np.uint8))
    clean = np.zeros_like(mask)
    for lbl in range(1, n_cc):
        comp = cc_map == lbl
        if int(comp.sum()) >= min_px:
            clean |= comp
    return clean


def fit_seed_ellipse(comp_mask: np.ndarray):
    """Fit an ellipse to a seed component (FULL-length axes, matching cv2.fitEllipse).
    Falls back to the minimum enclosing circle for < 5 contour points."""
    cnts, _ = cv2.findContours(comp_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0.0, 0.0), (1.0, 1.0), 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        return (cx, cy), (MA, ma), angle
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (cx, cy), (2 * r, 2 * r), 0.0


def shrunk_ellipse_mask(H: int, W: int, ellipse, scale: float) -> np.ndarray:
    """Rasterise the seed ellipse scaled about its own centre by `scale` (0..1)."""
    (cx, cy), (MA, ma), angle = ellipse
    m = np.zeros((H, W), dtype=np.uint8)
    ax = max(1, int(round(MA / 2.0 * scale)))
    ay = max(1, int(round(ma / 2.0 * scale)))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (ax, ay),
                angle, 0, 360, color=1, thickness=-1)
    return m.astype(bool)


# ─────────────────────────── Baseline builder ────────────────────────────

def build_ellipse_baseline(
    input_dir: Path,
    label_path: Path,
    output_path: Path,
    clip_scale: float = 1.0,
    min_comp_px: int = 20,
    pole_floor: float = 1.75,
):
    masks_dir = input_dir / "masks"

    annotated_frames = sorted(
        int(p.stem) for p in masks_dir.iterdir() if p.suffix == ".png"
    )
    if not annotated_frames:
        raise FileNotFoundError(f"No mask PNGs found in {masks_dir}")

    # Spatial metadata: the label grid is also the output grid (no frames needed).
    ref = sitk.ReadImage(str(label_path))
    W, H, n_frames = ref.GetSize()                 # ITK size is (x, y, z)
    spacing = ref.GetSpacing()                     # (x_mm, y_mm, z_mm)
    pixel_spacing_mm = (spacing[0] + spacing[1]) / 2.0
    slice_thickness_mm = spacing[2]

    print(f"\nAnnotated frames : {annotated_frames}")
    print(f"Grid             : {W}x{H} x {n_frames} slices")
    print(f"Spacing          : in-plane={pixel_spacing_mm:.3f} mm  "
          f"slice={slice_thickness_mm:.3f} mm")
    print(f"Mode             : ELLIPSE BASELINE (no model) "
          f"clip_scale={clip_scale}  pole_floor={pole_floor}  "
          f"min_comp_px={min_comp_px}")

    all_binary = np.zeros((n_frames, H, W), dtype=bool)
    n_obj = 0

    for seed_frame_idx in annotated_frames:
        mask_path = masks_dir / f"{seed_frame_idx:02d}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{seed_frame_idx:03d}.png"
        comp_masks = load_seed_components(mask_path)
        if not comp_masks:
            print(f"\n[frame {seed_frame_idx}] no valid components, skipping")
            continue

        print(f"\n[frame {seed_frame_idx}] {len(comp_masks)} component(s)")
        for m in comp_masks:
            n_obj += 1
            ell = fit_seed_ellipse(m)
            # radius_px from the min-enclosing circle of the component pixels —
            # IDENTICAL to the clip (points from np.where, not the contour).
            ys, xs = np.where(m)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            _, radius_px = cv2.minEnclosingCircle(pts)
            radius_mm = radius_px * pixel_spacing_mm
            D = max(1, int(np.ceil(radius_mm / slice_thickness_mm)))
            P0 = max(radius_mm / slice_thickness_mm, pole_floor)

            covered = 0
            for d in range(-D, D + 1):
                z = seed_frame_idx + d
                if not (0 <= z < n_frames):
                    continue
                sigma = np.sqrt(max(0.0, 1.0 - (d / P0) ** 2)) * clip_scale
                if sigma <= 0.0:
                    continue
                # n_min noise filter applied PER OBJECT, before the OR-merge —
                # same ordering as the clip (remove_small_components acts on each
                # object's mask inside clip_seed_results, not on the merged frame).
                em = remove_small_components(shrunk_ellipse_mask(H, W, ell, sigma),
                                             min_comp_px)
                if em.any():
                    all_binary[z] |= em
                    covered += 1
            print(f"  obj r={radius_mm:.2f} mm  D=±{D}  pole={P0:.2f}  "
                  f"slices covered={covered}")

    ann_after = int(all_binary.any(axis=(1, 2)).sum())
    print(f"\nObjects seeded                       : {n_obj}")
    print(f"Annotated slices before propagation  : {len(annotated_frames)} / {n_frames}")
    print(f"Annotated slices after  propagation  : {ann_after} / {n_frames}")

    # ── Write output volumes with the label's spatial metadata ────────────
    binary_volume = all_binary.astype(np.uint16)
    instance_volume = np.zeros((n_frames, H, W), dtype=np.uint16)
    for z in range(n_frames):
        if binary_volume[z].any():
            _, cc_map = cv2.connectedComponents(binary_volume[z].astype(np.uint8))
            instance_volume[z] = cc_map.astype(np.uint16)

    def write_vol(arr, path, intent):
        img = sitk.GetImageFromArray(arr.astype(np.uint16))
        img.CopyInformation(ref)
        img.SetMetaData("intent_name", intent)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(path))
        nii = path.with_suffix("").with_suffix(".nii.gz")
        sitk.WriteImage(img, str(nii))
        print(f"Saved {path}")
        print(f"Saved {nii}")

    stem = output_path.stem.replace(".nrrd", "")
    write_vol(binary_volume, output_path, "label")
    write_vol(instance_volume, output_path.with_name(stem + "_instances.nrrd"),
              "label_instances")


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Model-free ellipse-stack baseline (no SAM2).")
    ap.add_argument("--input-dir", default="sam2_input",
                    help="Directory containing masks/ (seed PNGs). frames/ unused.")
    ap.add_argument("--label", required=True,
                    help="Label volume for spatial metadata (.nii.gz or .nrrd).")
    ap.add_argument("--output", default="propagated_label_ellipse.nrrd",
                    help="Output NRRD path (default: propagated_label_ellipse.nrrd).")
    ap.add_argument("--clip-scale", type=float, default=1.0,
                    help="Multiplier on the ellipse size at every depth (default 1.0), "
                         "matching sam2_propagate_clip's --clip-scale.")
    ap.add_argument("--min-comp-px", type=int, default=20,
                    help="Drop connected components smaller than this many pixels "
                         "after OR-merging (default 20). Use 0 to keep every speck.")
    ap.add_argument("--pole-floor", type=float, default=1.75,
                    help="Lower bound on the sphere pole in slices (default 1.75), "
                         "so sub-slice lesions still survive d=1. 1.0 = strict sphere.")
    a = ap.parse_args()

    build_ellipse_baseline(
        input_dir=Path(a.input_dir),
        label_path=Path(a.label),
        output_path=Path(a.output),
        clip_scale=a.clip_scale,
        min_comp_px=a.min_comp_px,
        pole_floor=a.pole_floor,
    )
