#!/usr/bin/env python3
"""
sam2_propagate_ablation.py — leave-one-out ablation of the three constraints.

Purpose: from ONE shared SAM2 pass per seed, write a family of volumes that
differ only in which constraint is off, so every difference is attributable to
that one constraint (no run-to-run noise). Guarantees full ⊆ none.

    <stem>_full           all three on  (== sam2_propagate_clip)
    <stem>_no_continuity  continuity gate off
    <stem>_no_depth       RECIST depth window off
    <stem>_no_clip        shrinking-ellipse clip off
    <stem>_none           all three off

The noise filter (min_comp_px) is kept in every variant — not an ablated constraint.

Use:
    python sam2_propagate_ablation.py --sam2-dir sam2 --input-dir sam2_input \\
        --label label.nii.gz --output case_ablation.nrrd

Core steps:
  continuity_stop  — per-direction last depth the continuity gate holds a lesion
  assemble_variant — build one volume from the shared masks with a subset of constraints
  propagate        — one ungated SAM2 pass per seed, then assemble every variant
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import SimpleITK as sitk

# ── MPS compatibility ─────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The four (or five) variants this script can emit. Each maps to the set of
# constraints that stay ON for that volume.
VARIANTS = {
    "full":          {"continuity": True,  "depth": True,  "clip": True},
    "no_continuity": {"continuity": False, "depth": True,  "clip": True},
    "no_depth":      {"continuity": True,  "depth": False, "clip": True},
    "no_clip":       {"continuity": True,  "depth": True,  "clip": False},
    "none":          {"continuity": False, "depth": False, "clip": False},
}


# ─────────────────────────── Device selection ────────────────────────────

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────── Connected-component helpers ─────────────────
# (verbatim from sam2_propagate_clip.py so the ablation shares its exact logic)

def components_with_centroids(binary_mask: np.ndarray, min_px: int = 50):
    """Return [(component_mask, centroid_xy), ...] for each foreground component."""
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
    """Load a seed-frame mask PNG and return one bool array per connected component."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return []
    binary = (raw > 127).astype(np.uint8)
    return [comp.astype(bool) for comp, _ in components_with_centroids(binary, min_px=20)]


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px pixels from a bool mask."""
    if min_px <= 0:
        return mask
    n_cc, cc_map = cv2.connectedComponents(mask.astype(np.uint8))
    clean = np.zeros_like(mask)
    for lbl in range(1, n_cc):
        comp = cc_map == lbl
        if int(comp.sum()) >= min_px:
            clean |= comp
    return clean


def fit_seed_ellipse(comp_mask: np.ndarray):
    """Fit an ellipse to a seed component (cv2.fitEllipse; axes are FULL lengths)."""
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


# ─────────────────────────── Ablation assembly ───────────────────────────

def continuity_stop(masks_by_d: dict, seed_centroid, radius_px: float,
                    bound: int, max_step_velocity: float) -> int:
    """Last depth (>=0) at which the continuity gate still holds this lesion in
    one direction. Walks outward from the seed; ends at the first empty/missing
    mask (DROPOUT — always applies) or, when max_step_velocity > 0, the first
    centroid jump exceeding max_step_velocity seed radii (VELOCITY).

    `masks_by_d` maps depth d (>=1) -> mask for this lesion in this direction.
    Returns the largest d such that every step 1..d passed the gate (0 if the
    very first slice fails). Re-acquisitions past the stop are NOT bridged — the
    walk terminates, matching the permanent kill in the live pipeline.
    """
    prev_c = seed_centroid
    stop = 0
    for d in range(1, bound + 1):
        m = masks_by_d.get(d)
        if m is None or not m.any():
            break                               # dropout: SAM2 lost the lesion
        c = mask_centroid(m)
        if c is None:
            break
        if max_step_velocity > 0 and prev_c is not None:
            v = np.hypot(c[0] - prev_c[0], c[1] - prev_c[1]) / max(radius_px, 1.0)
            if v > max_step_velocity:
                break                           # velocity: hopped to a neighbour
        prev_c = c
        stop = d
    return stop


def assemble_variant(
    seed_frame_idx: int,
    n_frames: int,
    seed_masks: dict,          # lid -> seed-frame bool mask
    seed_ellipses: dict,       # lid -> fitted ellipse
    obj_depth: dict,           # lid -> RECIST depth (slices, each direction)
    obj_radius_mm: dict,       # lid -> measured seed radius (mm)
    obj_radius_px: dict,       # lid -> seed radius (px)
    obj_seed_centroid: dict,   # lid -> seed (cx, cy)
    collected: dict,           # frame_idx -> {lid -> bool mask}   (ungated)
    slice_thickness_mm: float,
    H: int, W: int,
    *,
    use_continuity: bool,
    use_depth: bool,
    use_clip: bool,
    clip_scale: float,
    min_comp_px: int,
    pole_floor: float,
    max_step_velocity: float,
    prop_bound: int,
) -> dict:
    """Build one ablation volume as {frame_idx: bool mask} from the shared
    `collected` masks, applying only the constraints that are ON.
    """
    out: dict = {}

    def _add(frame_idx, mask):
        prev = out.get(frame_idx)
        out[frame_idx] = mask if prev is None else (prev | mask)

    for lid in seed_ellipses:
        # Seed frame is the annotation itself — kept unchanged in every variant.
        sm = seed_masks.get(lid)
        if sm is not None and sm.any():
            _add(seed_frame_idx, sm)

        ell = seed_ellipses[lid]
        maxd = max(1, obj_depth.get(lid, 1))
        pole0 = max(obj_radius_mm.get(lid, 0.0) / slice_thickness_mm, pole_floor)

        for direction, sign in (("fwd", +1), ("bwd", -1)):
            # depth d -> this lesion's mask in this direction (from shared pass)
            masks_by_d = {}
            for d in range(1, prop_bound + 1):
                z = seed_frame_idx + sign * d
                if not (0 <= z < n_frames):
                    continue
                m = collected.get(z, {}).get(lid)
                if m is not None and m.any():
                    masks_by_d[d] = m

            if use_continuity:
                stop = continuity_stop(masks_by_d, obj_seed_centroid.get(lid),
                                       obj_radius_px.get(lid, 1.0),
                                       prop_bound, max_step_velocity)
                # extend the taper to the tracked extent ONLY when the velocity
                # test actually ran (same warrant as the live pipeline).
                extend = max_step_velocity > 0
            else:
                stop = prop_bound          # no truncation
                extend = False

            for d, prop_mask in masks_by_d.items():
                if use_depth and d > maxd:
                    continue
                if use_continuity and d > stop:
                    continue
                if use_clip:
                    pole = max(pole0, stop + 0.5) if (use_continuity and extend) else pole0
                    scale = np.sqrt(max(0.0, 1.0 - (d / pole) ** 2)) * clip_scale
                    if scale <= 0.0:
                        continue
                    m = prop_mask & shrunk_ellipse_mask(H, W, ell, scale)
                else:
                    m = prop_mask
                m = remove_small_components(m, min_comp_px)   # noise removal kept
                if m.any():
                    _add(seed_frame_idx + sign * d, m)

    return {k: v for k, v in out.items() if v.any()}


# ─────────────────────────── Main ────────────────────────────────────────

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
    max_prop_depth: int = 0,       # >0 = fixed cap (slices); 0 = auto per seed
    no_depth_mult: float = 4.0,    # no_depth reach = mult × seed RECIST depth
    no_depth_floor: int = 6,       # ... but at least this many slices
    variants=("full", "no_continuity", "no_depth", "no_clip"),
    emit_instances: bool = False,
):
    device     = select_device()
    frames_dir = input_dir / "frames"
    masks_dir  = input_dir / "masks"

    annotated_frames = sorted(
        int(p.stem) for p in masks_dir.iterdir() if p.suffix == ".png"
    )
    if not annotated_frames:
        raise FileNotFoundError(f"No mask PNGs found in {masks_dir}")
    print(f"\nAnnotated frames : {annotated_frames}")
    print(f"Variants         : {', '.join(variants)}")
    print(f"clip_scale={clip_scale}  min_comp_px={min_comp_px}  "
          f"pole_floor={pole_floor}  max_step_velocity={max_step_velocity}")

    ref = sitk.ReadImage(str(label_path))
    spacing = ref.GetSpacing()
    pixel_spacing_mm = (spacing[0] + spacing[1]) / 2.0
    slice_thickness_mm = spacing[2]
    print(f"Spacing          : in-plane={pixel_spacing_mm:.2f} mm  "
          f"slice={slice_thickness_mm:.2f} mm")

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

    offload = device.type != "cuda"
    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=offload,
        async_loading_frames=False,
    )
    n_frames = inference_state["num_frames"]
    print(f"Video frames loaded: {n_frames}")

    sample_jpg = next(p for p in frames_dir.iterdir() if p.suffix in (".jpg", ".jpeg"))
    H, W = cv2.imread(str(sample_jpg), cv2.IMREAD_GRAYSCALE).shape

    # One accumulator volume per requested variant.
    vols = {name: np.zeros((n_frames, H, W), dtype=bool) for name in variants}

    # Only the no_depth variant needs propagation to run past the RECIST window;
    # every other variant is bounded by depth/continuity and uses d ≤ depth. So
    # extra reach is collected per seed ONLY when no_depth is requested (and no
    # explicit --max-prop-depth cap is set), keeping the other variants exactly as
    # cheap as the normal pipeline.
    need_extra = ("no_depth" in variants) and (max_prop_depth <= 0)

    for seed_num, seed_frame_idx in enumerate(annotated_frames):
        mask_path = masks_dir / f"{seed_frame_idx:02d}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{seed_frame_idx:03d}.png"

        comp_masks = load_seed_components(mask_path)
        if not comp_masks:
            print(f"\n[{seed_num+1}/{len(annotated_frames)}] frame {seed_frame_idx}: "
                  f"no valid components, skipping")
            continue

        areas = [int(m.sum()) for m in comp_masks]
        print(f"\n[{seed_num+1}/{len(annotated_frames)}] frame {seed_frame_idx}: "
              f"{len(comp_masks)} component(s)  areas={areas}")

        predictor.reset_state(inference_state)
        for lid, m in enumerate(comp_masks, start=1):
            predictor.add_new_mask(inference_state, seed_frame_idx, lid, m)

        # Per-lesion geometry (identical to sam2_propagate_clip.py).
        obj_depth = {}
        obj_radius_mm = {}
        obj_radius_px = {}
        obj_seed_centroid = {}
        seed_ellipses = {}
        seed_masks = {}
        for lid, m in enumerate(comp_masks, start=1):
            ys, xs = np.where(m)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            _, radius_px = cv2.minEnclosingCircle(pts)
            radius_mm = radius_px * pixel_spacing_mm
            obj_depth[lid] = max(1, int(np.ceil(radius_mm / slice_thickness_mm)))
            obj_radius_mm[lid] = radius_mm
            obj_radius_px[lid] = max(radius_px, 1.0)
            obj_seed_centroid[lid] = mask_centroid(m)
            seed_ellipses[lid] = fit_seed_ellipse(m)
            seed_masks[lid] = m
        depth_str = ", ".join(
            f"obj{lid}=±{d} (pole {obj_radius_mm[lid] / slice_thickness_mm:.1f})"
            for lid, d in obj_depth.items())
        print(f"  RECIST depth per lesion: {depth_str}")

        # How far to propagate/collect each direction for THIS seed. Bounded by the
        # RECIST window unless no_depth needs extra reach (then a multiple of the
        # window, floored), and never past the video edge.
        seed_max_depth = max(obj_depth.values())
        if max_prop_depth > 0:
            prop_bound = min(n_frames, max_prop_depth)
        elif need_extra:
            prop_bound = min(n_frames,
                             max(int(np.ceil(seed_max_depth * no_depth_mult)),
                                 no_depth_floor))
        else:
            prop_bound = min(n_frames, seed_max_depth)

        # ── Shared, UNGATED collection: every mask SAM2 emits within prop_bound,
        #    forward then backward. No depth gate, no continuity kill, no clip. ──
        collected = {}

        def _collect(out_f, oid, mask):
            collected.setdefault(out_f, {})
            prev = collected[out_f].get(oid)
            collected[out_f][oid] = mask if prev is None else (prev | mask)

        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx
        ):
            if out_f > seed_frame_idx + prop_bound:
                break
            for i, oid in enumerate(out_ids):
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                if m.any():
                    _collect(out_f, oid, m)

        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx, reverse=True
        ):
            if out_f < seed_frame_idx - prop_bound:
                break
            for i, oid in enumerate(out_ids):
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                if m.any():
                    _collect(out_f, oid, m)

        # ── Assemble every requested variant from that one collection ─────────
        for name in variants:
            flags = VARIANTS[name]
            asm = assemble_variant(
                seed_frame_idx, n_frames, seed_masks, seed_ellipses,
                obj_depth, obj_radius_mm, obj_radius_px, obj_seed_centroid,
                collected, slice_thickness_mm, H, W,
                use_continuity=flags["continuity"],
                use_depth=flags["depth"],
                use_clip=flags["clip"],
                clip_scale=clip_scale, min_comp_px=min_comp_px,
                pole_floor=pole_floor, max_step_velocity=max_step_velocity,
                prop_bound=prop_bound,
            )
            for frame_idx, bm in asm.items():
                vols[name][frame_idx] |= bm
            print(f"    {name:<14} frames covered: {len(asm)}")

    # ── Write one binary volume (+ optional instances) per variant ────────────
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
    print()
    for name in variants:
        binary = vols[name].astype(np.uint8)
        vox = int(binary.sum())
        slices = int(binary.any(axis=(1, 2)).sum())
        print(f"[{name}] voxels={vox}  slices={slices}/{n_frames}")
        write_vol(binary, output_path.with_name(f"{stem}_{name}.nrrd"), f"label_{name}")
        if emit_instances:
            inst = np.zeros_like(binary, dtype=np.uint16)
            for f in range(n_frames):
                if binary[f].any():
                    _, cc = cv2.connectedComponents(binary[f])
                    inst[f] = cc.astype(np.uint16)
            write_vol(inst, output_path.with_name(f"{stem}_{name}_instances.nrrd"),
                      f"label_{name}_instances")

    # Monotonicity sanity: full ⊆ no_X ⊆ none for each relaxed constraint.
    if "full" in vols and "none" in vols:
        full_b = vols["full"]
        none_b = vols["none"]
        leak = int((full_b & ~none_b).sum())
        print(f"\nSanity full \\ none = {leak}  "
              + ("(OK: full ⊆ none)" if leak == 0 else "(!! investigate)"))


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-one-out constraint ablation for SAM2 clip propagation."
    )
    parser.add_argument("--sam2-dir",   default="sam2")
    parser.add_argument("--input-dir",  default="sam2_input")
    parser.add_argument("--label",      required=True,
                        help="Original label file for spatial metadata (.nii.gz/.nrrd)")
    parser.add_argument("--output",     default="ablation.nrrd",
                        help="Output stem; variants become <stem>_<variant>.nrrd/.nii.gz")
    parser.add_argument("--model",      default="large",
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--clip-scale", type=float, default=1.0)
    parser.add_argument("--min-comp-px", type=int, default=20,
                        help="Noise removal threshold (kept in EVERY variant; not "
                             "one of the ablated constraints). 0 keeps every speck.")
    parser.add_argument("--pole-floor", type=float, default=1.75)
    parser.add_argument("--max-step-velocity", type=float, default=0.5, metavar="R",
                        help="Velocity half of the continuity gate (seed radii per "
                             "slice); 0 disables the velocity test (dropout still ends "
                             "a direction). Governs the continuity gate in variants "
                             "where it is ON.")
    parser.add_argument("--max-prop-depth", type=int, default=0, metavar="N",
                        help="Fixed cap (slices) on propagation each direction from a "
                             "seed. 0 = auto: bound by the RECIST window, or (when "
                             "no_depth is requested) by --no-depth-mult × that window. "
                             "Set >0 to force a fixed reach for every seed.")
    parser.add_argument("--no-depth-mult", type=float, default=4.0, metavar="M",
                        help="For the no_depth ablation only: how far past the RECIST "
                             "window to let propagation run, as a multiple of that "
                             "seed's window (default 4). Larger = more over-reach shown, "
                             "more compute. Ignored if --max-prop-depth is set.")
    parser.add_argument("--no-depth-floor", type=int, default=6, metavar="N",
                        help="Minimum no_depth reach in slices, so tiny lesions still "
                             "get room to over-propagate (default 6).")
    parser.add_argument("--variants", default="full,no_continuity,no_depth,no_clip",
                        help="Comma list from: full,no_continuity,no_depth,no_clip,none "
                             "(default: full,no_continuity,no_depth,no_clip).")
    parser.add_argument("--emit-instances", action="store_true",
                        help="Also write per-slice instance volumes for each variant.")
    args = parser.parse_args()

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in requested if v not in VARIANTS]
    if bad:
        parser.error(f"unknown variant(s): {bad}. choose from {list(VARIANTS)}")

    propagate(
        sam2_dir=Path(args.sam2_dir),
        input_dir=Path(args.input_dir),
        label_path=Path(args.label),
        output_path=Path(args.output),
        model=args.model,
        clip_scale=args.clip_scale,
        min_comp_px=args.min_comp_px,
        pole_floor=args.pole_floor,
        max_step_velocity=args.max_step_velocity,
        max_prop_depth=args.max_prop_depth,
        no_depth_mult=args.no_depth_mult,
        no_depth_floor=args.no_depth_floor,
        variants=tuple(requested),
        emit_instances=args.emit_instances,
    )
