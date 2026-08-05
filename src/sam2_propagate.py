#!/usr/bin/env python3
"""
sam2_propagate.py — propagate tumor masks across MRI slices using SAM2
-----------------------------------------------------------------------
Reads the frames/ and masks/ produced by prepare_sam2.py, runs SAM2 video
propagation (forward + backward), and writes a propagated label volume.

Connected components within each mask are each tracked as a separate SAM2
object. Each seed frame is propagated independently; results are filtered by
area and centroid proximity then OR-merged into a single binary volume.
Instance IDs in the output are assigned per-slice and are not matched across
seeds or slices.

Usage
-----
    python sam2_propagate.py \\
        --sam2-dir  sam2_test/sam2 \\
        --input-dir sam2_input \\
        --label     label.nii.gz \\
        --output    propagated_label.nrrd

    # Faster but less accurate:
    python sam2_propagate.py ... --model small
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── MPS compatibility ─────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ─────────────────────────── Device selection ────────────────────────────

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────── Connected-component helpers ─────────────────

def components_with_centroids(binary_mask: np.ndarray, min_px: int = 50):
    """
    Return [(component_mask, centroid_xy), ...] for each foreground component.
    Filters out components smaller than min_px pixels.
    """
    n, label_map = cv2.connectedComponents(binary_mask.astype(np.uint8))
    results = []
    for lbl in range(1, n):
        comp = (label_map == lbl).astype(np.uint8)
        px = int(comp.sum())
        if px < min_px:
            continue
        M = cv2.moments(comp)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        results.append((comp, (cx, cy)))
    return results


def load_seed_components(mask_path: Path) -> list[np.ndarray]:
    """Load a seed-frame mask PNG and return one bool array per connected component.

    Uses min_px=20 so small but genuine ground-truth annotations are not
    silently dropped (propagation results are still filtered at min_comp_px=200).
    """
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


def drift_limit_for(radius_px: float, drift_frac: float,
                    drift_floor: float, drift_cap: float) -> float:
    """Per-tumour allowed per-step centroid drift, in pixels.

    Scales with the lesion's own radius so a tiny ellipse is held tight while a
    large mass may shift proportionally: limit = drift_frac × radius_px, clamped
    to [drift_floor, drift_cap]. A fixed pixel threshold is simultaneously too
    loose for small lesions and too tight for large ones; this makes the check
    scale-invariant so it need not be retuned per case.
    """
    return float(min(drift_cap, max(drift_floor, drift_frac * radius_px)))


def _object_passes(
    prop_mask: np.ndarray,
    seed_stat: dict,
    prev_cxy: tuple[float, float],
    prev_area: float,
    max_area_ratio: float,
    shrink_factor: float,
    min_px: int,
):
    """Decide whether a single propagated object mask is a plausible continuation.

    Returns (keep: bool, centroid: (cx, cy) | None, area: int).

    Two area anchors are enforced together:
    - Seed cap (RECIST 'seed is the largest cross-section'): area may not exceed
      max_area_ratio × seed area. Bounds absolute size against the annotated max.
    - Per-step shrink (vs the PREVIOUS accepted slice): area may not exceed
      shrink_factor × prev_area, so the cross-section keeps getting smaller moving
      outward instead of holding constant. prev_area is the seed area on the first
      step. The seed cap bounds absolute size; the shrink factor bounds the trend.

    Drift is the per-STEP centroid displacement from prev_cxy (the last accepted
    slice, or the seed centroid on the first step), thresholded per-tumour via
    seed_stat["drift_limit"]. Catches a single large jump while allowing gradual
    migration. The min-size gate checks the largest connected component (mirrors
    remove_small_components) so sub-threshold fragments can't keep a frame alive.
    """
    if not prop_mask.any():
        return False, None, 0
    n_cc, cc_map = cv2.connectedComponents(prop_mask.astype(np.uint8))
    largest_cc = max((int((cc_map == i).sum()) for i in range(1, n_cc)), default=0)
    if largest_cc < min_px:
        return False, None, 0
    area = int(prop_mask.sum())
    if area > max_area_ratio * seed_stat["area"]:      # seed cap (absolute size)
        return False, None, 0
    if area > shrink_factor * prev_area:               # per-step shrink (trend)
        return False, None, 0
    ys, xs = np.where(prop_mask)
    cxy = (float(xs.mean()), float(ys.mean()))
    if np.hypot(cxy[0] - prev_cxy[0], cxy[1] - prev_cxy[1]) > seed_stat["drift_limit"]:
        return False, None, 0
    return True, cxy, area


def filter_seed_results(
    raw: dict,
    seed_frame_idx: int,
    seed_stats: list[dict],
    max_area_ratio: float,
    shrink_factor: float,
    min_comp_px: int = 200,
) -> dict:
    """
    Keep only propagated masks that are plausible extensions of the seed.

    Each object is walked OUTWARD from the seed frame (forward then backward),
    tracking the last accepted centroid AND area so drift and shrinkage are
    measured slice-to-slice rather than against the fixed seed. Per-slice rules
    (see _object_passes):
      - Area  ≤ max_area_ratio × seed area         (seed cap, absolute size)
      - Area  ≤ shrink_factor × previous area       (per-step shrink trend)
      - Centroid step ≤ seed_stat["drift_limit"] px (per-tumour, vs previous slice)
      - Connected components < min_comp_px pixels are stripped

    seed_stats carries the precomputed per-object area/centroid/radius/drift_limit
    (built once in propagate). The seed frame itself is kept unchanged.

    Returns {frame_idx: binary_bool_mask}.
    """
    frames = sorted(raw.keys())
    if not frames:
        return {}
    lo, hi = frames[0], frames[-1]

    result: dict = {}

    def _add(frame_idx, mask):
        existing = result.get(frame_idx)
        result[frame_idx] = mask if existing is None else (existing | mask)

    for lid, ss in enumerate(seed_stats, start=1):
        # Seed frame: keep SAM2's seed-frame reconstruction unchanged (ground-truth).
        seed_mask = raw.get(seed_frame_idx, {}).get(lid)
        if seed_mask is not None and seed_mask.any():
            _add(seed_frame_idx, seed_mask)

        # Walk each direction independently, comparing to the last accepted slice.
        for step in (1, -1):
            prev_cxy = (ss["cx"], ss["cy"])
            prev_area = ss["area"]
            f = seed_frame_idx + step
            while lo <= f <= hi:
                prop_mask = raw.get(f, {}).get(lid)
                if prop_mask is None:          # past this object's recorded range
                    break
                keep, cxy, area = _object_passes(
                    prop_mask, ss, prev_cxy, prev_area,
                    max_area_ratio, shrink_factor, min_comp_px
                )
                if keep:
                    cleaned = remove_small_components(prop_mask, min_comp_px)
                    if cleaned.any():
                        _add(f, cleaned)
                        prev_cxy = cxy         # advance references to this accepted slice
                        prev_area = area
                # A failed slice is skipped but does not advance prev_cxy/prev_area, so
                # the next slice is still measured against the last good one.
                f += step

    return {k: v for k, v in result.items() if v.any()}


# ─────────────────────────── SAM2 loader ─────────────────────────────────

MODEL_CFGS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def load_predictor(sam2_dir: Path, model: str, device):
    sys.path.insert(0, str(sam2_dir))
    from sam2.build_sam import build_sam2_video_predictor   # noqa: E402

    cfg, ckpt_name = MODEL_CFGS[model]
    ckpt_path = sam2_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run  {sam2_dir}/checkpoints/download_ckpts.sh  to download it."
        )
    print(f"Loading SAM2 '{model}' from {ckpt_path} on {device}")
    return build_sam2_video_predictor(cfg, str(ckpt_path), device=device)


# ─────────────────────────── Slice results ───────────────────────────────

# Colors (BGR) cycled per object id for the overlay panel
_OVERLAY_COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
]


def save_slice_results(frames_dir: Path, all_segs: dict, n_frames: int,
                       results_dir: Path):
    """
    Save per-slice visualisation images to results_dir:
        {frame:02d}_result.png  –  3-panel figure (original | overlay | mask)
        {frame:02d}_mask.png    –  binary mask PNG
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving slice results to {results_dir}/")

    # Sort frame jpgs by filename so we can index by frame number
    frame_files = sorted(
        p for p in frames_dir.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")
    )
    frame_by_idx = {i: p for i, p in enumerate(frame_files)}

    for frame_idx in range(n_frames):
        frame_path = frame_by_idx.get(frame_idx)
        if frame_path is None:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        segs = all_segs.get(frame_idx, {})

        # Build binary mask and colored overlay
        binary = np.zeros((h, w), dtype=np.uint8)
        overlay = img_rgb.copy()
        legend_patches = []

        for obj_id, m in segs.items():
            if not m.any():
                continue
            binary[m] = 255
            color_bgr = _OVERLAY_COLORS[(obj_id - 1) % len(_OVERLAY_COLORS)]
            color_rgb = color_bgr[::-1]
            tinted = overlay.copy()
            tinted[m] = (np.array(color_rgb) * 0.5 + overlay[m] * 0.5).astype(np.uint8)
            overlay = tinted
            # Draw contour
            contour_mask = m.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.drawContours(overlay_bgr, cnts, -1, color_bgr, 1)
            overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            legend_patches.append(
                mpatches.Patch(color=[c / 255 for c in color_rgb], label=f"obj {obj_id}")
            )

        # 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_rgb);       axes[0].set_title("Original");            axes[0].axis("off")
        axes[1].imshow(overlay);       axes[1].set_title(f"Propagated (frame {frame_idx})"); axes[1].axis("off")
        axes[2].imshow(binary, cmap="gray"); axes[2].set_title("Binary mask");   axes[2].axis("off")
        if legend_patches:
            axes[1].legend(handles=legend_patches, loc="lower right", fontsize=7)
        plt.tight_layout()

        stem = f"{frame_idx:02d}"
        fig.savefig(str(results_dir / f"{stem}_result.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        cv2.imwrite(str(results_dir / f"{stem}_mask.png"), binary)

    print(f"Slice results saved: {results_dir}/")


# ─────────────────────────── Main ────────────────────────────────────────

def _merge_seg(all_segs, frame_idx, obj_id, mask):
    """OR a new mask into the running per-frame-per-object accumulator."""
    existing = all_segs[frame_idx].get(obj_id)
    all_segs[frame_idx][obj_id] = mask if existing is None else (existing | mask)


def propagate(
    sam2_dir: Path,
    input_dir: Path,
    label_path: Path,
    output_path: Path,
    model: str = "large",
    max_area_ratio: float = 1.1,
    shrink_factor: float = 1.0,
    drift_frac: float = 0.5,
    drift_floor: float = 5.0,
    max_drift: float = 150.0,
    min_comp_px: int = 200,
    max_dead_frames: int = 5,
    slice_results_dir: Path | None = None,
):
    device     = select_device()
    frames_dir = input_dir / "frames"
    masks_dir  = input_dir / "masks"

    # ── Discover annotated frames ─────────────────────────────────────────
    annotated_frames = sorted(
        int(p.stem) for p in masks_dir.iterdir() if p.suffix == ".png"
    )
    if not annotated_frames:
        raise FileNotFoundError(f"No mask PNGs found in {masks_dir}")
    print(f"\nAnnotated frames : {annotated_frames}")
    print(f"Filters          : max_area_ratio={max_area_ratio}  shrink_factor={shrink_factor}  "
          f"drift={drift_frac}×radius (floor {drift_floor}px, cap {max_drift}px)  "
          f"max_dead_frames={max_dead_frames}")

    # ── Spatial metadata (used for depth estimation and output writing) ───
    ref = sitk.ReadImage(str(label_path))
    spacing = ref.GetSpacing()          # (x_mm, y_mm, z_mm)
    pixel_spacing_mm = (spacing[0] + spacing[1]) / 2.0
    slice_thickness_mm = spacing[2]
    print(f"Spacing          : in-plane={pixel_spacing_mm:.2f} mm  "
          f"slice={slice_thickness_mm:.2f} mm")

    # ── Load SAM2 ─────────────────────────────────────────────────────────
    predictor = load_predictor(sam2_dir, model, device)

    if device.type == "cuda":
        predictor = predictor.to(torch.bfloat16)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "cpu":
        torch.autocast("cpu", dtype=torch.bfloat16).__enter__()
    elif device.type == "mps":
        torch.autocast("mps", dtype=torch.float16).__enter__()

    # ── Init state (loads all frames once; reset_state reuses them) ───────
    offload = device.type != "cuda"
    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=offload,
        async_loading_frames=False,
    )
    n_frames = inference_state["num_frames"]
    print(f"Video frames loaded: {n_frames}")

    # Determine frame dimensions from a sample JPEG
    sample_jpg = next(p for p in frames_dir.iterdir() if p.suffix in (".jpg", ".jpeg"))
    H, W = cv2.imread(str(sample_jpg), cv2.IMREAD_GRAYSCALE).shape

    # ── Per-seed-frame independent propagation ────────────────────────────
    # Each annotated frame is propagated independently (forward + backward).
    # Results are filtered by area and centroid proximity before OR-merging.
    # This prevents an unusually large mask on one frame from biasing the
    # propagation context for every other frame.
    all_binary = np.zeros((n_frames, H, W), dtype=bool)

    for seed_num, seed_frame_idx in enumerate(annotated_frames):
        mask_path = masks_dir / f"{seed_frame_idx:02d}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{seed_frame_idx:03d}.png"

        comp_masks = load_seed_components(mask_path)
        if not comp_masks:
            print(f"\n[{seed_num+1}/{len(annotated_frames)}] frame {seed_frame_idx}: no valid components, skipping")
            continue

        areas = [int(m.sum()) for m in comp_masks]
        print(f"\n[{seed_num+1}/{len(annotated_frames)}] frame {seed_frame_idx}: "
              f"{len(comp_masks)} component(s)  areas={areas}")

        predictor.reset_state(inference_state)
        for lid, m in enumerate(comp_masks, start=1):
            predictor.add_new_mask(inference_state, seed_frame_idx, lid, m)

        # Seed stats used for both inline termination and post-hoc filtering.
        # radius_px is the minimum-enclosing-circle radius (≈ half the RECIST long
        # axis) — a faithful longest-extent measure rather than an area-equivalent
        # circle, which underestimates the true half-long-axis for elongated lesions.
        seed_stats = []
        for m in comp_masks:
            area = int(m.sum())
            ys, xs = np.where(m)
            if area:
                pts = np.column_stack([xs, ys]).astype(np.float32)
                _, radius_px = cv2.minEnclosingCircle(pts)
            else:
                radius_px = 0.0
            seed_stats.append({
                "area": area,
                "cx": float(xs.mean()) if area else 0.0,
                "cy": float(ys.mean()) if area else 0.0,
                "radius_px": float(radius_px),
                "drift_limit": drift_limit_for(radius_px, drift_frac,
                                               drift_floor, max_drift),
            })

        # RECIST-based depth limit, computed PER LESION (per connected component).
        # Assume each tumour is roughly spherical, so its superior-inferior extent
        # ≈ its in-plane radius each way. depth (slices, each direction) =
        # radius_mm / slice_thickness_mm. The radius is the min-enclosing-circle
        # radius (≈ half the RECIST long axis), so an elongated lesion is no longer
        # under-propagated the way an area-equivalent circle radius would.
        # Each object is bounded by ITS OWN radius, so a small component co-located
        # with a large one on the same seed frame is no longer dragged out to the
        # large one's range. The overall loop is bounded by the largest per-object
        # depth; objects are dropped individually once they pass their own limit
        # (see the per-object gate in the propagation loops).
        obj_depth = {}                              # object id (lid) → max depth in slices
        for lid, ss in enumerate(seed_stats, start=1):
            radius_mm = ss["radius_px"] * pixel_spacing_mm  # seed at equator → extends ±radius
            obj_depth[lid] = max(1, int(np.ceil(radius_mm / slice_thickness_mm)))
        max_depth = max(obj_depth.values())         # loop bound = largest per-object depth
        depth_str = ", ".join(f"obj{lid}=±{d}" for lid, d in obj_depth.items())
        print(f"  Max depth (RECIST, per-lesion): {depth_str}  (loop bound ±{max_depth})")
        drift_str = ", ".join(f"obj{lid}={ss['drift_limit']:.0f}px"
                              for lid, ss in enumerate(seed_stats, start=1))
        print(f"  Drift limit  (per-lesion)     : {drift_str}")

        # Collect raw forward + backward results with early termination.
        # Propagation stops at max_depth slices from seed (RECIST geometry) or
        # after max_dead_frames consecutive failures (tracking lost), whichever
        # comes first.
        raw = {f: {} for f in range(n_frames)}

        # Per-object last-accepted centroid, so the inline dead-frame check measures
        # drift slice-to-slice (matching filter_seed_results) rather than vs the seed.
        prev_cxy = {lid: (ss["cx"], ss["cy"]) for lid, ss in enumerate(seed_stats, start=1)}
        prev_area = {lid: ss["area"] for lid, ss in enumerate(seed_stats, start=1)}
        dead = 0
        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx
        ):
            if out_f > seed_frame_idx + max_depth:
                print(f"  → Fwd depth limit at frame {out_f}")
                break
            frame_masks = {}
            for i, oid in enumerate(out_ids):
                # Per-lesion gate: drop this object once it exceeds its own depth,
                # even though larger co-seeded objects keep propagating.
                if (out_f - seed_frame_idx) > obj_depth.get(oid, max_depth):
                    continue
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                _merge_seg(raw, out_f, oid, m)
                frame_masks[oid] = m
            if out_f != seed_frame_idx:
                passed = False
                for oid, m in frame_masks.items():
                    if oid - 1 >= len(seed_stats):
                        continue
                    ok, cxy, area = _object_passes(m, seed_stats[oid - 1], prev_cxy[oid],
                                                   prev_area[oid], max_area_ratio,
                                                   shrink_factor, min_comp_px)
                    if ok:
                        prev_cxy[oid] = cxy
                        prev_area[oid] = area
                        passed = True
                if passed:
                    dead = 0
                else:
                    dead += 1
                    if dead >= max_dead_frames:
                        print(f"  → Fwd early-stop at frame {out_f} "
                              f"({dead} consecutive dead frames)")
                        break

        prev_cxy = {lid: (ss["cx"], ss["cy"]) for lid, ss in enumerate(seed_stats, start=1)}
        prev_area = {lid: ss["area"] for lid, ss in enumerate(seed_stats, start=1)}
        dead = 0
        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx, reverse=True
        ):
            if out_f < seed_frame_idx - max_depth:
                print(f"  → Bwd depth limit at frame {out_f}")
                break
            frame_masks = {}
            for i, oid in enumerate(out_ids):
                # Per-lesion gate: drop this object once it exceeds its own depth,
                # even though larger co-seeded objects keep propagating.
                if (seed_frame_idx - out_f) > obj_depth.get(oid, max_depth):
                    continue
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                _merge_seg(raw, out_f, oid, m)
                frame_masks[oid] = m
            if out_f != seed_frame_idx:
                passed = False
                for oid, m in frame_masks.items():
                    if oid - 1 >= len(seed_stats):
                        continue
                    ok, cxy, area = _object_passes(m, seed_stats[oid - 1], prev_cxy[oid],
                                                   prev_area[oid], max_area_ratio,
                                                   shrink_factor, min_comp_px)
                    if ok:
                        prev_cxy[oid] = cxy
                        prev_area[oid] = area
                        passed = True
                if passed:
                    dead = 0
                else:
                    dead += 1
                    if dead >= max_dead_frames:
                        print(f"  → Bwd early-stop at frame {out_f} "
                              f"({dead} consecutive dead frames)")
                        break

        # Filter and accumulate
        filtered = filter_seed_results(raw, seed_frame_idx, seed_stats,
                                       max_area_ratio, shrink_factor, min_comp_px)
        covered = 0
        for frame_idx, bin_mask in filtered.items():
            all_binary[frame_idx] |= bin_mask
            covered += 1
        print(f"  → frames covered after filtering: {covered}")

    # ── Final cleanup: remove tiny fragments that survived OR-merging ────
    # Seed (annotated) frames are EXEMPT: their small components are real
    # ground-truth annotations, not SAM2 noise, so they must survive even when
    # below min_comp_px. Only propagated (non-annotated) frames get the cull.
    annotated_set = set(annotated_frames)
    for f in range(n_frames):
        if all_binary[f].any() and f not in annotated_set:
            all_binary[f] = remove_small_components(all_binary[f], min_comp_px)

    # ── Optional per-slice result images ─────────────────────────────────
    if slice_results_dir is not None:
        all_segs_vis = {f: {1: all_binary[f]}
                        for f in range(n_frames) if all_binary[f].any()}
        save_slice_results(frames_dir, all_segs_vis, n_frames, slice_results_dir)

    # ── Assemble output volumes ───────────────────────────────────────────
    binary_volume = all_binary.astype(np.uint8)

    # Instance volume: 2-D connected components per slice (consistent within
    # each slice; IDs are not matched across slices)
    instance_volume = np.zeros((n_frames, H, W), dtype=np.uint16)
    for f in range(n_frames):
        if binary_volume[f].any():
            _, cc_map = cv2.connectedComponents(binary_volume[f])
            instance_volume[f] = cc_map.astype(np.uint16)

    ann_before = len(annotated_frames)
    ann_after  = int(binary_volume.any(axis=(1, 2)).sum())
    print(f"\nAnnotated slices before propagation : {ann_before} / {n_frames}")
    print(f"Annotated slices after  propagation : {ann_after} / {n_frames}")

    # ── Write output volumes with original spatial metadata ───────────────
    def write_vol(arr, path, extra_meta=None):
        img = sitk.GetImageFromArray(arr.astype(np.uint16))
        img.CopyInformation(ref)
        if extra_meta:
            for k, v in extra_meta.items():
                img.SetMetaData(k, v)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(path))
        nii = path.with_suffix("").with_suffix(".nii.gz")
        sitk.WriteImage(img, str(nii))
        print(f"Saved {path}")
        print(f"Saved {nii}")

    stem = output_path.stem.replace(".nrrd", "")
    write_vol(binary_volume,   output_path,
              {"intent_name": "label"})
    write_vol(instance_volume, output_path.with_name(stem + "_instances.nrrd"),
              {"intent_name": "label_instances"})


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Propagate tumour masks across MRI slices with SAM2."
    )
    parser.add_argument("--sam2-dir",   default="sam2",
                        help="Path to the sam2 repo (default: sam2_test/sam2)")
    parser.add_argument("--input-dir",  default="sam2_input",
                        help="Directory with frames/ and masks/ (default: sam2_input)")
    parser.add_argument("--label",      required=True,
                        help="Original label file for spatial metadata (.nii.gz or .nrrd)")
    parser.add_argument("--output",     default="propagated_label.nrrd",
                        help="Output NRRD path (default: propagated_label.nrrd)")
    parser.add_argument("--model",      default="large",
                        choices=["tiny", "small", "base_plus", "large"],
                        help="SAM2.1 model size (default: large)")
    parser.add_argument("--max-area-ratio", type=float, default=1.1,
                        help="Seed cap: reject propagated masks whose area exceeds this "
                             "multiple of the SEED mask area (default: 1.1). Bounds absolute "
                             "size against the annotated max cross-section; 1.1 gives 10%% "
                             "tolerance for segmentation imprecision.")
    parser.add_argument("--shrink-factor", type=float, default=1.0,
                        help="Per-step shrink: reject a propagated mask whose area exceeds "
                             "this multiple of the PREVIOUS accepted slice's area "
                             "(default: 1.0 = must not grow outward). Forces the cross-section "
                             "to keep shrinking toward the poles instead of holding constant. "
                             "Raise slightly (e.g. 1.05) to tolerate segmentation noise.")
    parser.add_argument("--drift-frac", type=float, default=0.5,
                        help="Per-tumour per-step drift tolerance as a fraction of the "
                             "lesion radius (default: 0.5). A mask is rejected when its "
                             "centroid jumps more than drift_frac × radius from the PREVIOUS "
                             "accepted slice, so a small ellipse is held tight while a large "
                             "mass may shift proportionally. Catches SAM2 latching onto a "
                             "neighbour while allowing gradual slice-to-slice migration.")
    parser.add_argument("--drift-floor", type=float, default=5.0,
                        help="Minimum per-step drift tolerance in pixels (default: 5), so "
                             "tiny lesions still get some slack.")
    parser.add_argument("--max-drift", type=float, default=150.0,
                        help="Absolute cap on the per-tumour per-step drift tolerance in "
                             "pixels (default: 150).")
    parser.add_argument("--min-comp-px", type=int, default=200,
                        help="Remove connected components smaller than this many pixels "
                             "from propagated masks (default: 200). Eliminates spurious "
                             "tiny fragments from SAM2 without affecting tumour regions.")
    parser.add_argument("--max-dead-frames", type=int, default=5,
                        help="Stop propagating from a seed when this many consecutive "
                             "frames all fail the area/drift filter (default: 5). "
                             "Prevents SAM2 hidden-state corruption from distant frames.")
    parser.add_argument("--slice-results", default=None, metavar="DIR",
                        help="If set, save per-slice overlay + mask PNGs to this "
                             "directory (mirrors build_volume / extracter output).")
    args = parser.parse_args()

    propagate(
        sam2_dir=Path(args.sam2_dir),
        input_dir=Path(args.input_dir),
        label_path=Path(args.label),
        output_path=Path(args.output),
        model=args.model,
        max_area_ratio=args.max_area_ratio,
        shrink_factor=args.shrink_factor,
        drift_frac=args.drift_frac,
        drift_floor=args.drift_floor,
        max_drift=args.max_drift,
        min_comp_px=args.min_comp_px,
        max_dead_frames=args.max_dead_frames,
        slice_results_dir=Path(args.slice_results) if args.slice_results else None,
    )
