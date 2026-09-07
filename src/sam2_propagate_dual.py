#!/usr/bin/env python3
"""
sam2_propagate_dual.py — clip propagation that also emits the raw output.

Purpose: same as sam2_propagate_clip.py, but --emit-raw additionally writes the
UNCONSTRAINED SAM2 output (ungated OR-merge of every in-window mask) from the
SAME pass, so raw and constrained are comparable with no run-to-run noise. Since
the constraints only ever drop masks, constrained ⊆ raw exactly. Raw goes to
<output_stem>_raw.nrrd; the constrained keeps the normal name.

Use:
    python sam2_propagate_dual.py --sam2-dir sam2 --input-dir sam2_input \\
        --label label.nii.gz --output propagated_label_clip.nrrd --emit-raw

Core steps:
  clip_seed_results — intersect each propagated mask with the sphere-tapered seed ellipse
  propagate         — run fwd+bwd per seed; accumulate constrained (+ raw if --emit-raw)
  helpers           — connected components, seed-ellipse fit, noise removal, slice QC PNGs
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
    silently dropped (propagation results are filtered again at --min-comp-px).
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


def fit_seed_ellipse(comp_mask: np.ndarray):
    """Fit an ellipse to a seed component. Returns (center_xy, (MA, ma), angle_deg),
    matching cv2.fitEllipse (axes are FULL lengths). Falls back to the minimum
    enclosing circle for components too small for fitEllipse (< 5 contour points).
    """
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


def mask_centroid(mask: np.ndarray):
    """(cx, cy) of a bool mask, or None when it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def shrunk_ellipse_mask(H: int, W: int, ellipse, scale: float) -> np.ndarray:
    """Rasterise the seed ellipse scaled about its own centre by `scale` (0..1)."""
    (cx, cy), (MA, ma), angle = ellipse
    m = np.zeros((H, W), dtype=np.uint8)
    ax = max(1, int(round(MA / 2.0 * scale)))
    ay = max(1, int(round(ma / 2.0 * scale)))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (ax, ay),
                angle, 0, 360, color=1, thickness=-1)
    return m.astype(bool)


def clip_seed_results(
    raw: dict,
    seed_frame_idx: int,
    seed_ellipses: list,
    obj_depth: dict,
    obj_radius_mm: dict,
    slice_thickness_mm: float,
    H: int,
    W: int,
    clip_scale: float = 1.0,
    min_comp_px: int = 20,     # matches the --min-comp-px default
    pole_floor: float = 1.75,
    obj_stop_d: dict | None = None,
    extend_pole_to_stop: bool = True,
) -> dict:
    """Clip each propagated mask to a depth-scaled shrinking seed ellipse.

    scale(d) = sqrt(1 - (d/pole)^2) · clip_scale, pole = radius_mm/slice_thickness
    (floored at pole_floor). Where a lesion ends is set by obj_stop_d (continuity
    tracking) when given; the taper only sets how wide it is. The seed frame is
    kept as-is; the depth window (obj_depth) bounds propagation; noise removal is
    the only other filter. Returns {frame_idx: binary_bool_mask}.
    """
    result: dict = {}

    def _add(frame_idx, mask):
        existing = result.get(frame_idx)
        result[frame_idx] = mask if existing is None else (existing | mask)

    for lid, ell in enumerate(seed_ellipses, start=1):
        # Seed frame kept unchanged (it is the annotation itself).
        seed_mask = raw.get(seed_frame_idx, {}).get(lid)
        if seed_mask is not None and seed_mask.any():
            _add(seed_frame_idx, seed_mask)

        maxd = max(1, obj_depth.get(lid, 1))
        # True sphere pole in slices, from the measured seed radius. Floored so a
        # lesion thinner than one slice still survives d=1 instead of being zeroed.
        pole0 = max(obj_radius_mm.get(lid, 0.0) / slice_thickness_mm, pole_floor)
        stops = (obj_stop_d or {}).get(lid, {})
        for frame_idx, obj_masks in raw.items():
            if frame_idx == seed_frame_idx:
                continue
            prop_mask = obj_masks.get(lid)
            if prop_mask is None or not prop_mask.any():
                continue
            d = abs(frame_idx - seed_frame_idx)
            if d > maxd:
                continue
            # End set by tracking (obj_stop_d); taper only sets width.
            stop = stops.get("fwd" if frame_idx > seed_frame_idx else "bwd")
            if stop is None:
                pole = pole0                       # tracking disabled → pure geometry
            else:
                if d > stop:
                    continue                       # past where SAM2 held the lesion
                # Extend the taper to the tracked extent (+0.5 keeps the last slice
                # non-zero), but only when the velocity test verified the track.
                pole = max(pole0, stop + 0.5) if extend_pole_to_stop else pole0
            # Sphere cross-section at depth d, tapering to zero at `pole`.
            scale = np.sqrt(max(0.0, 1.0 - (d / pole) ** 2)) * clip_scale
            if scale <= 0.0:
                continue
            clipped = prop_mask & shrunk_ellipse_mask(H, W, ell, scale)
            clipped = remove_small_components(clipped, min_comp_px)  # noise removal kept
            if clipped.any():
                _add(frame_idx, clipped)

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
    clip_scale: float = 1.0,
    min_comp_px: int = 20,
    pole_floor: float = 1.75,
    max_step_velocity: float = 0.5,
    slice_results_dir: Path | None = None,
    emit_raw: bool = False,
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
    print(f"Mode             : ellipse-clip (no frame rejection)  "
          f"clip_scale={clip_scale}  min_comp_px={min_comp_px} (noise removal kept)")
    print(f"Taper            : sphere, zero at pole=radius_mm/slice "
          f"(pole_floor={pole_floor})")
    if max_step_velocity > 0:
        print(f"Continuity stop  : kill a direction when the centroid jumps > "
              f"{max_step_velocity} seed radii between slices, or the mask empties; "
              f"taper extends to the tracked extent")
    else:
        print("Continuity stop  : velocity OFF; dropout still ends a direction "
              "(no re-acquisition across a gap)")

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
    # Each propagated mask is CLIPPED to a depth-scaled shrinking ellipse (no
    # whole-frame rejection) before OR-merging into the output volume.
    all_binary = np.zeros((n_frames, H, W), dtype=bool)
    # Ungated raw OR-merge (only when --emit-raw): every in-window SAM2 mask,
    # no continuity gate / clip / noise removal — the no_prior equivalent.
    all_binary_raw = np.zeros((n_frames, H, W), dtype=bool) if emit_raw else None

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

        # Per-lesion RECIST depth + fitted seed ellipse: depth = radius_mm/slice,
        # radius_px from the min-enclosing circle; loop bound = the largest depth.
        obj_depth = {}                              # object id (lid) → max depth in slices
        obj_radius_mm = {}                          # object id (lid) → measured seed radius
        obj_radius_px = {}                          # object id (lid) → seed radius in pixels
        obj_seed_centroid = {}                      # object id (lid) → seed (cx, cy)
        seed_ellipses = []                          # object id order → fitted ellipse
        for m in comp_masks:
            ys, xs = np.where(m)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            _, radius_px = cv2.minEnclosingCircle(pts)
            radius_mm = radius_px * pixel_spacing_mm
            lid = len(seed_ellipses) + 1
            obj_depth[lid] = max(1, int(np.ceil(radius_mm / slice_thickness_mm)))
            obj_radius_mm[lid] = radius_mm          # drives the sphere taper
            obj_radius_px[lid] = max(radius_px, 1.0)   # normalises the step velocity
            obj_seed_centroid[lid] = mask_centroid(m)
            seed_ellipses.append(fit_seed_ellipse(m))
        max_depth = max(obj_depth.values())         # loop bound = largest per-object depth
        depth_str = ", ".join(
            f"obj{lid}=±{d} (pole {obj_radius_mm[lid] / slice_thickness_mm:.1f})"
            for lid, d in obj_depth.items())
        print(f"  Max depth (RECIST, per-lesion): {depth_str}  (loop bound ±{max_depth})")

        # Collect raw forward + backward results, bounded only by RECIST depth.
        # NO dead-frame early-stop / accept-reject filtering in this variant — the
        # shrinking-ellipse clip does the containment instead.
        raw = {f: {} for f in range(n_frames)}

        # obj_stop_d[lid][direction] = last depth SAM2 still held this lesion.
        # Dropout always applies; max_step_velocity <= 0 drops only the velocity test.
        obj_stop_d = {lid: {} for lid in obj_depth}

        def _track(direction, out_f, oid, m, prev_c, alive):
            """Continuity gate: return True to keep this slice and continue.

            Kills a direction on DROPOUT (mask empties — always) or VELOCITY (centroid
            jumps > max_step_velocity seed radii — only when max_step_velocity > 0).
            The kill is permanent, so re-acquisitions past a gap are not bridged.
            """
            d = abs(out_f - seed_frame_idx)
            if not alive.get(oid, True):
                return False
            if d == 0:
                obj_stop_d[oid][direction] = 0       # seed frame, nothing to compare
                return True
            c = mask_centroid(m)
            if c is None:                            # dropout: SAM2 lost the lesion
                alive[oid] = False
                return False
            pc = prev_c.get(oid)
            if max_step_velocity > 0 and pc is not None:
                v = np.hypot(c[0] - pc[0], c[1] - pc[1]) / obj_radius_px.get(oid, 1.0)
                if v > max_step_velocity:
                    alive[oid] = False
                    return False
            prev_c[oid] = c
            obj_stop_d[oid][direction] = d
            return True

        prev_c = dict(obj_seed_centroid)
        alive = {lid: True for lid in obj_depth}
        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx
        ):
            if out_f > seed_frame_idx + max_depth:
                print(f"  → Fwd depth limit at frame {out_f}")
                break
            for i, oid in enumerate(out_ids):
                if (out_f - seed_frame_idx) > obj_depth.get(oid, max_depth):
                    continue                        # per-lesion depth gate
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                if emit_raw:                          # ungated: before the gate
                    all_binary_raw[out_f] |= m
                if not _track("fwd", out_f, oid, m, prev_c, alive):
                    continue
                _merge_seg(raw, out_f, oid, m)

        prev_c = dict(obj_seed_centroid)             # each direction tracks separately
        alive = {lid: True for lid in obj_depth}
        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx, reverse=True
        ):
            if out_f < seed_frame_idx - max_depth:
                print(f"  → Bwd depth limit at frame {out_f}")
                break
            for i, oid in enumerate(out_ids):
                if (seed_frame_idx - out_f) > obj_depth.get(oid, max_depth):
                    continue                        # per-lesion depth gate
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                if emit_raw:                          # ungated: before the gate
                    all_binary_raw[out_f] |= m
                if not _track("bwd", out_f, oid, m, prev_c, alive):
                    continue
                _merge_seg(raw, out_f, oid, m)

        trunc = [f"obj{lid}={s.get('bwd', 0)}/{s.get('fwd', 0)}"
                 for lid, s in sorted(obj_stop_d.items())
                 if min(s.get('bwd', 0), s.get('fwd', 0)) < obj_depth[lid]]
        if trunc:
            print(f"  Continuity stop (bwd/fwd depth kept): {', '.join(trunc)}")

        # Clip to the shrinking ellipse and accumulate (noise removal kept).
        clipped = clip_seed_results(raw, seed_frame_idx, seed_ellipses, obj_depth,
                                    obj_radius_mm, slice_thickness_mm,
                                    H, W, clip_scale, min_comp_px, pole_floor,
                                    obj_stop_d,
                                    extend_pole_to_stop=max_step_velocity > 0)
        covered = 0
        for frame_idx, bin_mask in clipped.items():
            all_binary[frame_idx] |= bin_mask
            covered += 1
        print(f"  → frames covered after clipping: {covered}")

    # No post-merge noise cull: every merged mask already passed min_comp_px inside
    # clip_seed_results and OR-merging only grows components, so it could remove nothing.

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

    # ── Optional: unconstrained raw volume from the SAME pass ──────────────
    if emit_raw:
        raw_volume = all_binary_raw.astype(np.uint8)
        write_vol(raw_volume, output_path.with_name(stem + "_raw.nrrd"),
                  {"intent_name": "label_raw"})
        # Sanity: constrained must be a strict subset of raw within one run.
        extra = int((binary_volume.astype(bool) & ~all_binary_raw).sum())
        print(f"\nRaw (unconstrained) voxels : {int(all_binary_raw.sum())}")
        print(f"Constrained voxels         : {int(binary_volume.sum())}")
        print(f"Constrained \\ raw          : {extra}  "
              + ("(OK: constrained ⊆ raw)" if extra == 0
                 else "(!! non-zero — investigate)"))


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
    parser.add_argument("--output",     default="propagated_label_clip.nrrd",
                        help="Output NRRD path (default: propagated_label_clip.nrrd)")
    parser.add_argument("--model",      default="large",
                        choices=["tiny", "small", "base_plus", "large"],
                        help="SAM2.1 model size (default: large)")
    parser.add_argument("--clip-scale", type=float, default=1.0,
                        help="Multiplier on the shrinking-ellipse size (default: 1.0). "
                             "The clip ellipse at depth d is the seed ellipse scaled by "
                             "sqrt(1-(d/pole)^2) × clip_scale, where pole = "
                             "radius_mm / slice_thickness_mm is the lesion's own sphere "
                             "pole. Raise above 1.0 to keep more pixels (looser clip); "
                             "lower to clip tighter.")
    parser.add_argument("--min-comp-px", type=int, default=20,
                        help="Remove connected components smaller than this many pixels "
                             "AFTER clipping (default: 20, matching the seed-loading "
                             "threshold so a lesion good enough to seed can also survive "
                             "propagation). This is the only filter kept — noise removal, "
                             "not frame rejection. Use 0 to keep every speck.")
    parser.add_argument("--pole-floor", type=float, default=1.75,
                        help="Lower bound on a lesion's sphere pole, in slices "
                             "(default: 1.75). A lesion whose radius is under one "
                             "slice thickness would otherwise taper to nothing at "
                             "d=1 and vanish on the slice next to its own "
                             "annotation; this floor keeps it alive. 1.75 is the "
                             "smallest value at which every sub-slice lesion in "
                             "brac62721a/dabc71a survives d=1 after min-comp-px. "
                             "Binds only on lesions under ~0.875 mm radius, and "
                             "still zeroes them at d=2. Set to 1.0 for strict "
                             "sphere geometry (drops the smallest lesions).")
    parser.add_argument("--max-step-velocity", type=float, default=0.5, metavar="R",
                        help="Velocity half of the continuity stop, in SEED RADII "
                             "PER SLICE (default: 0.5; 0 disables the VELOCITY test "
                             "only — dropout still ends a direction, and the taper "
                             "then keeps the width implied by the measured radius). "
                             "Stepping outward from the seed, the centroid movement "
                             "between consecutive slices is measured and divided by "
                             "the seed's radius, so the threshold is scale-free: 0.5 "
                             "means 'half a lesion-radius of sideways movement in "
                             "one slice'. Exceeding it — or the mask going empty — "
                             "ends that direction permanently, and re-acquisitions "
                             "past the gap are not bridged. Measured on dabc71a a "
                             "genuine elongated lesion creeps at 0.04/slice while "
                             "masks that hopped onto neighbouring structures lurched "
                             "at 0.61-1.22, so the useful range is roughly 0.3-0.8. "
                             "Lower = stop sooner (fewer extra lesions, more "
                             "truncation); higher = propagate further.")
    parser.add_argument("--slice-results", default=None, metavar="DIR",
                        help="If set, save per-slice overlay + mask PNGs to this "
                             "directory (mirrors build_volume / extracter output).")
    parser.add_argument("--emit-raw", action="store_true",
                        help="Also write the UNCONSTRAINED raw SAM2 output "
                             "(ungated OR-merge of every in-window mask — no "
                             "continuity gate, no ellipse clip, no noise removal) "
                             "from the SAME propagation pass, to <output>_raw.nrrd "
                             "(+ .nii.gz). Guarantees constrained ⊆ raw exactly, so "
                             "the two can be compared without run-to-run noise.")
    args = parser.parse_args()

    propagate(
        sam2_dir=Path(args.sam2_dir),
        input_dir=Path(args.input_dir),
        label_path=Path(args.label),
        output_path=Path(args.output),
        model=args.model,
        clip_scale=args.clip_scale,
        pole_floor=args.pole_floor,
        max_step_velocity=args.max_step_velocity,
        min_comp_px=args.min_comp_px,
        slice_results_dir=Path(args.slice_results) if args.slice_results else None,
        emit_raw=args.emit_raw,
    )
