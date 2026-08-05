#!/usr/bin/env python3
"""
prepare_sam2.py — Convert raw MR DICOMs + NIfTI label to SAM2 input
--------------------------------------------------------------------
Reads the raw (unannotated) MR DICOM folder and a label NIfTI/NRRD built
by build_volume.py (with --raw-dir), and writes:

  <output-dir>/
    frames/
      00.jpg   ← raw MR slice as RGB JPEG  (SAM2 video frames, no annotation overlay)
      01.jpg
      ...
    masks/
      02.png   ← binary mask for annotated slice 2  (SAM2 prompts)
      ...
    frame_info.csv  ← frame_idx, instance_number, has_mask

Usage
-----
    python prepare_sam2.py \\
        --raw-dir brac46551b_raw/ \\
        --label   label.nii.gz \\
        --output  sam2_input/

Both --raw-dir and --label must be in the same MR space (produced from the
same raw DICOM series). The frame index matches the SimpleITK GDCM slice
order used by build_volume.py.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


# ─────────────────────────── DICOM helpers ───────────────────────────────

def read_series_order(dcm_dir: Path):
    """
    Return the DICOM files in the same SimpleITK slice order used by
    build_volume.py, along with each slice's InstanceNumber.

    Returns: [(slice_idx, instance_number, Path), ...]
    """
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM series found in {dcm_dir}")

    result = []
    for i, f in enumerate(dcm_files):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        inst = int(getattr(ds, "InstanceNumber", i + 1))
        result.append((i, inst, Path(f)))
    return result


def dcm_to_rgb(path: Path, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Read a DICOM file and return an 8-bit RGB numpy array (H × W × 3).

    lo/hi: global intensity window bounds (1st/99th percentile of the whole volume).
    If not provided, falls back to per-slice min/max (less consistent across frames).
    """
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array

    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        w_lo = lo if lo is not None else arr.min()
        w_hi = hi if hi is not None else arr.max()
        if w_hi > w_lo:
            arr = (arr - w_lo) / (w_hi - w_lo) * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        arr = arr.astype(np.uint8)

    return arr


# ─────────────────────────── Label helpers ───────────────────────────────

def load_label_volume(label_path: Path) -> np.ndarray:
    """
    Read a NIfTI or NRRD label file.
    Returns a uint8 array of shape (Z, H, W) matching SimpleITK slice order.
    """
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)   # (Z, Y, X)
    return (arr > 0).astype(np.uint8)


# ─────────────────────────── Main ────────────────────────────────────────

def prepare(dcm_dir: Path, label_path: Path, output_dir: Path):
    frames_dir = output_dir / "frames"
    masks_dir  = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    slice_order = read_series_order(dcm_dir)
    n_slices    = len(slice_order)
    pad         = len(str(n_slices - 1))   # digit width for zero-padding

    label_vol = load_label_volume(label_path)
    if label_vol.shape[0] != n_slices:
        raise ValueError(
            f"Label z-size ({label_vol.shape[0]}) != DICOM slice count ({n_slices}). "
            "Make sure the label was built from the same DCM folder."
        )

    # ── Global intensity window (1st–99th percentile across all slices) ──
    # Per-slice min/max normalization makes each frame look different to SAM2
    # (a mostly-fat slice vs a tumor-containing slice have different intensity
    # ranges). A global window gives consistent appearance across the video.
    print("\nComputing global intensity window …")
    all_pixels = []
    for _, _, dcm_path in slice_order:
        ds = pydicom.dcmread(str(dcm_path))
        px = ds.pixel_array
        if px.ndim == 2:
            all_pixels.append(px.ravel())
    if all_pixels:
        stacked = np.concatenate(all_pixels).astype(np.float32)
        global_lo, global_hi = float(np.percentile(stacked, 1)), float(np.percentile(stacked, 99))
        print(f"  Global window: [{global_lo:.1f}, {global_hi:.1f}]")
    else:
        global_lo, global_hi = None, None

    csv_rows = []
    n_prompt_frames = 0

    print(f"\nConverting {n_slices} slices → {output_dir}/")
    for slice_idx, inst_num, dcm_path in slice_order:
        frame_name = f"{slice_idx:0{pad}d}.jpg"

        # ── Frame image ──────────────────────────────────────────────────
        rgb = dcm_to_rgb(dcm_path, lo=global_lo, hi=global_hi)
        cv2.imwrite(str(frames_dir / frame_name),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Prompt mask (if this slice is annotated) ──────────────────────
        mask_slice = label_vol[slice_idx]
        has_mask   = bool(mask_slice.any())

        if has_mask:
            mask_png = (mask_slice * 255).astype(np.uint8)
            cv2.imwrite(str(masks_dir / frame_name.replace(".jpg", ".png")), mask_png)
            n_prompt_frames += 1

        csv_rows.append({
            "frame_idx":       slice_idx,
            "instance_number": inst_num,
            "has_mask":        int(has_mask),
        })

        status = f"mask saved ({mask_slice.sum()} px)" if has_mask else "no mask"
        print(f"  [{slice_idx:0{pad}d}]  inst={inst_num:4d}  {status}")

    # ── Summary CSV ───────────────────────────────────────────────────────
    csv_path = output_dir / "frame_info.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "instance_number", "has_mask"])
        writer.writeheader()
        writer.writerows(csv_rows)

    prompt_indices = [r["frame_idx"] for r in csv_rows if r["has_mask"]]
    print(f"\nFrames written  : {n_slices}  → {frames_dir}/")
    print(f"Prompt masks    : {n_prompt_frames}  → {masks_dir}/")
    print(f"Frame info CSV  : {csv_path}")
    print(f"\nPrompt frame indices: {prompt_indices}")


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw MR DICOMs + label NIfTI to SAM2 video-predictor input format."
    )
    parser.add_argument("--raw-dir",  required=True,
                        help="Folder of raw MR DICOM files (clean, no annotation overlay). "
                             "These become the video frames fed to SAM2.")
    parser.add_argument("--label",    required=True,
                        help="Label NRRD/NIfTI produced by build_volume.py with --raw-dir. "
                             "Must be in the same MR space as --raw-dir (256×226).")
    parser.add_argument("--output",   default="sam2_input",
                        help="Output directory (default: sam2_input/)")
    args = parser.parse_args()

    prepare(
        dcm_dir=Path(args.raw_dir),
        label_path=Path(args.label),
        output_dir=Path(args.output),
    )
