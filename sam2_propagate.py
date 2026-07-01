#!/usr/bin/env python3
"""
sam2_propagate.py — propagate tumor masks across MRI slices using SAM2
-----------------------------------------------------------------------
Reads the frames/ and masks/ produced by prepare_sam2.py, runs SAM2 video
propagation (forward + backward), and writes a propagated label volume.

Connected components within each mask are each tracked as a separate SAM2
object. Components across consecutive annotated slices are matched by
centroid proximity so the same physical tumour keeps one object id.

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

def components_with_centroids(binary_mask: np.ndarray):
    """
    Return [(component_mask, centroid_xy), ...] for each foreground component.
    Filters out components smaller than MIN_PX pixels.
    """
    MIN_PX = 50
    n, label_map = cv2.connectedComponents(binary_mask.astype(np.uint8))
    results = []
    for lbl in range(1, n):
        comp = (label_map == lbl).astype(np.uint8)
        px = int(comp.sum())
        if px < MIN_PX:
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
    return [comp.astype(bool) for comp, _ in components_with_centroids(binary)]


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px pixels from a bool mask."""
    n_cc, cc_map = cv2.connectedComponents(mask.astype(np.uint8))
    clean = np.zeros_like(mask)
    for lbl in range(1, n_cc):
        comp = cc_map == lbl
        if int(comp.sum()) >= min_px:
            clean |= comp
    return clean


def filter_seed_results(
    raw: dict,
    seed_frame_idx: int,
    comp_masks: list[np.ndarray],
    max_area_ratio: float,
    max_drift: float,
    min_comp_px: int = 200,
) -> dict:
    """
    Keep only propagated masks that are plausible extensions of the seed.

    Rules applied to every non-seed frame:
      - Area  ≤ max_area_ratio × seed component area
      - Centroid within max_drift pixels of seed centroid
      - Connected components < min_comp_px pixels are stripped (removes
        spurious tiny fragments that accumulate when OR-ing many seeds)

    The seed frame itself is always kept unchanged (it is ground-truth).

    Returns {frame_idx: binary_bool_mask}.
    """
    seed_stats = []
    for m in comp_masks:
        area = int(m.sum())
        ys, xs = np.where(m)
        seed_stats.append({
            "area": area,
            "cx": float(xs.mean()) if area else 0.0,
            "cy": float(ys.mean()) if area else 0.0,
        })

    result = {}
    for frame_idx, obj_masks in raw.items():
        frame_bin = None
        for lid, prop_mask in obj_masks.items():
            idx = lid - 1
            if idx >= len(seed_stats) or not prop_mask.any():
                continue
            ss = seed_stats[idx]

            if frame_idx == seed_frame_idx:
                keep = True
            else:
                area = int(prop_mask.sum())
                if area > max_area_ratio * ss["area"]:
                    keep = False
                else:
                    ys, xs = np.where(prop_mask)
                    drift = np.hypot(xs.mean() - ss["cx"], ys.mean() - ss["cy"])
                    keep = drift <= max_drift

                if keep:
                    prop_mask = remove_small_components(prop_mask, min_comp_px)
                    keep = prop_mask.any()

            if keep:
                frame_bin = prop_mask if frame_bin is None else (frame_bin | prop_mask)

        if frame_bin is not None and frame_bin.any():
            result[frame_idx] = frame_bin

    return result


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
                       results_dir: Path, ref_sitk=None):
    """
    Save per-slice visualisation images to results_dir, mirroring the
    build_volume / extracter pattern:
        {frame:02d}_result.png   –  3-panel figure (original | overlay | mask)
        {frame:02d}_mask.png     –  binary mask PNG
        {frame:02d}_mask.nrrd    –  2-D binary mask with in-plane spatial metadata
        {frame:02d}_mask.nii.gz  –  same, NIfTI format
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
    max_area_ratio: float = 2.0,
    max_drift: float = 150.0,
    min_comp_px: int = 200,
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
    print(f"Filters          : max_area_ratio={max_area_ratio}  max_drift={max_drift}px")

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

        # Collect raw forward + backward results
        raw = {f: {} for f in range(n_frames)}

        for out_f, out_ids, out_logits in predictor.propagate_in_video(inference_state):
            for i, oid in enumerate(out_ids):
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                _merge_seg(raw, out_f, oid, m)

        for out_f, out_ids, out_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            for i, oid in enumerate(out_ids):
                m = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                _merge_seg(raw, out_f, oid, m)

        # Filter and accumulate
        filtered = filter_seed_results(raw, seed_frame_idx, comp_masks,
                                       max_area_ratio, max_drift, min_comp_px)
        covered = 0
        for frame_idx, bin_mask in filtered.items():
            all_binary[frame_idx] |= bin_mask
            covered += 1
        print(f"  → frames covered after filtering: {covered}")

    # ── Final cleanup: remove tiny fragments that survived OR-merging ────
    for f in range(n_frames):
        if all_binary[f].any():
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
            n_cc, cc_map = cv2.connectedComponents(binary_volume[f])
            instance_volume[f] = cc_map.astype(np.uint16)

    ann_before = len(annotated_frames)
    ann_after  = int(binary_volume.any(axis=(1, 2)).sum())
    print(f"\nAnnotated slices before propagation : {ann_before} / {n_frames}")
    print(f"Annotated slices after  propagation : {ann_after} / {n_frames}")

    # ── Write output volumes with original spatial metadata ───────────────
    ref = sitk.ReadImage(str(label_path))

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
    parser.add_argument("--max-area-ratio", type=float, default=2.0,
                        help="Reject propagated masks whose area exceeds this multiple "
                             "of the seed mask area (default: 2.0)")
    parser.add_argument("--max-drift", type=float, default=150.0,
                        help="Reject propagated masks whose centroid has drifted more "
                             "than this many pixels from the seed centroid (default: 150)")
    parser.add_argument("--min-comp-px", type=int, default=200,
                        help="Remove connected components smaller than this many pixels "
                             "from propagated masks (default: 200). Eliminates spurious "
                             "tiny fragments from SAM2 without affecting tumour regions.")
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
        max_drift=args.max_drift,
        min_comp_px=args.min_comp_px,
        slice_results_dir=Path(args.slice_results) if args.slice_results else None,
    )
