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
import contextlib
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


def assign_object_ids(annotated_frames, masks_dir, max_gap=5, max_dist=120):
    """
    Build  {frame_idx: {obj_id: component_bool_mask}}  for every annotated frame.

    Cross-slice matching: if two consecutive annotated frames are within
    `max_gap` slices, components whose centroids are within `max_dist` pixels
    are considered the same tumour and share one obj_id.

    Returns (assignments_dict, total_objects).
    """
    assignments  = {}
    next_id      = 1
    prev_idx     = None
    prev_comps   = {}          # {obj_id: centroid}

    for frame_idx in sorted(annotated_frames):
        mask_path = masks_dir / f"{frame_idx:02d}.png"
        if not mask_path.exists():
            # try 3-digit padding
            mask_path = masks_dir / f"{frame_idx:03d}.png"
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        binary = (raw > 127).astype(np.uint8)

        comps = components_with_centroids(binary)

        # Match each component to a previous obj_id or create a new one
        gap_ok  = (prev_idx is not None) and (frame_idx - prev_idx <= max_gap)
        used_prev = set()
        frame_assignment = {}
        curr_centroids   = {}

        for comp, (cx, cy) in comps:
            best_id   = None
            best_dist = float("inf")

            if gap_ok:
                for pid, (px, py) in prev_comps.items():
                    if pid in used_prev:
                        continue
                    d = np.hypot(cx - px, cy - py)
                    if d < best_dist:
                        best_dist = d
                        best_id   = pid

            if best_id is not None and best_dist <= max_dist:
                obj_id = best_id
                used_prev.add(obj_id)
            else:
                obj_id   = next_id
                next_id += 1

            frame_assignment[obj_id] = comp.astype(bool)
            curr_centroids[obj_id]   = (cx, cy)

        assignments[frame_idx] = frame_assignment
        prev_comps = curr_centroids
        prev_idx   = frame_idx

    return assignments, next_id - 1


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
    batch_size: int = 8,
    slice_results_dir: Path | None = None,
):
    device      = select_device()
    frames_dir  = input_dir / "frames"
    masks_dir   = input_dir / "masks"

    # ── Discover annotated frames ─────────────────────────────────────────
    annotated_frames = sorted(
        int(p.stem) for p in masks_dir.iterdir() if p.suffix == ".png"
    )
    if not annotated_frames:
        raise FileNotFoundError(f"No mask PNGs found in {masks_dir}")
    print(f"\nAnnotated frames: {annotated_frames}")

    # ── Assign object ids to connected components ─────────────────────────
    assignments, n_objects = assign_object_ids(annotated_frames, masks_dir)
    print(f"Unique tumour objects: {n_objects}")
    for fi, objs in sorted(assignments.items()):
        print(f"  frame {fi:2d}: {list(objs.keys())}")

    # ── Load SAM2 ─────────────────────────────────────────────────────────
    predictor = load_predictor(sam2_dir, model, device)

    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            predictor = predictor.to(torch.bfloat16)
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif device.type == "mps":
        autocast_ctx = torch.autocast("mps", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # ── Init state (loads all frames once; reset_state reuses them) ───────
    offload = device.type != "cuda"
    with autocast_ctx:
        inference_state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=offload,
            async_loading_frames=False,
        )
        n_frames = inference_state["num_frames"]
        print(f"\nVideo frames loaded: {n_frames}")

        # ── Batched propagation ───────────────────────────────────────────────
        # MPS crashes when all objects are batched together; process in small
        # groups, resetting tracking state between batches (images stay loaded).
        all_obj_ids = sorted({oid for objs in assignments.values() for oid in objs})
        n_batches   = (len(all_obj_ids) + batch_size - 1) // batch_size
        all_segs    = {f: {} for f in range(n_frames)}

        for batch_num, batch_start in enumerate(range(0, len(all_obj_ids), batch_size)):
            batch_ids = set(all_obj_ids[batch_start : batch_start + batch_size])
            print(f"\n[Batch {batch_num + 1}/{n_batches}] objects {sorted(batch_ids)}")

            predictor.reset_state(inference_state)   # clears tracking, keeps images

            # Add prompts for this batch
            for frame_idx, obj_masks in sorted(assignments.items()):
                for obj_id, mask in obj_masks.items():
                    if obj_id in batch_ids:
                        predictor.add_new_mask(inference_state, frame_idx, obj_id, mask)
                        print(f"  frame {frame_idx:2d}  obj {obj_id:3d}  ({mask.sum()} px)")

            # Forward: earliest prompt → last frame
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                for i, oid in enumerate(out_obj_ids):
                    m = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    _merge_seg(all_segs, out_frame_idx, oid, m)

            # Backward: earliest prompt → frame 0
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                for i, oid in enumerate(out_obj_ids):
                    m = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    _merge_seg(all_segs, out_frame_idx, oid, m)

    # ── Optional per-slice result images ─────────────────────────────────
    if slice_results_dir is not None:
        save_slice_results(frames_dir, all_segs, n_frames, slice_results_dir)

    # ── Assemble output volumes ───────────────────────────────────────────
    sample_jpg = next(p for p in frames_dir.iterdir() if p.suffix in (".jpg", ".jpeg"))
    sample_img = cv2.imread(str(sample_jpg), cv2.IMREAD_GRAYSCALE)
    H, W = sample_img.shape

    binary_volume   = np.zeros((n_frames, H, W), dtype=np.uint8)
    instance_volume = np.zeros((n_frames, H, W), dtype=np.uint16)

    for f in range(n_frames):
        for oid, m in all_segs[f].items():
            binary_volume[f]   |= m.astype(np.uint8)
            instance_volume[f][m] = oid

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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Objects per propagation batch (default: 8). "
                             "Reduce to 4 if MPS crashes; raise to 16+ on CUDA.")
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
        batch_size=args.batch_size,
        slice_results_dir=Path(args.slice_results) if args.slice_results else None,
    )
