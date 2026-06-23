#!/usr/bin/env python3
"""
build_volume.py — batch mask extraction + NRRD label builder
-------------------------------------------------------------
Applies the crosshair extractor to every annotated DICOM slice, stacks
per-slice masks into a 3D volume, and writes label NRRD and NIfTI files
ready for nnU-Net.

Usage
-----
    python build_volume.py --dcm-dir /path/to/annotated_dcms/ --output case01_label.nrrd
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from extracter import process_image


# ─────────────────────────── DICOM helpers ───────────────────────────────

def read_dicom_series(dcm_dir: Path):
    """
    Use SimpleITK to read the full DICOM series in correct slice order.
    Returns (sitk_image, [(slice_idx, instance_number, filepath), ...]).
    """
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM series found in {dcm_dir}")
    reader.SetFileNames(dcm_files)
    ref_image = reader.Execute()

    slice_instance_map = []
    for i, f in enumerate(dcm_files):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        inst = int(getattr(ds, "InstanceNumber", i + 1))
        slice_instance_map.append((i, inst, Path(f)))

    return ref_image, slice_instance_map


def instance_from_dcm(path: Path) -> int:
    """Read InstanceNumber tag from a DICOM file."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return int(getattr(ds, "InstanceNumber", 0))


# ─────────────────────────── Per-slice extraction ────────────────────────

def extract_mask(dcm_path: Path, target_hw: tuple[int, int],
                 results_dir: Path) -> np.ndarray:
    """
    Run the extracter pipeline on one annotated DICOM (RGB Secondary Capture).
    Returns a uint8 binary mask (0/1) resampled to target_hw = (H, W).
    """
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array          # H × W × 3 RGB  (or H × W grayscale)
    except Exception as e:
        print(f"    [WARN] cannot read {dcm_path.name}: {e}")
        return np.zeros(target_hw, dtype=np.uint8)

    if arr.ndim == 2:
        img_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)

    _, mask = process_image(dcm_path, output_dir=results_dir, show=False,
                            image_bgr=img_bgr)

    dcm_h, dcm_w = target_hw
    if mask.shape != (dcm_h, dcm_w):
        mask = cv2.resize(mask, (dcm_w, dcm_h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# ─────────────────────────── Main ────────────────────────────────────────

def build_volume(dcm_dir: Path, output_path: Path):
    print(f"\nReading annotated DICOM series from: {dcm_dir}")
    ref_image, slice_map = read_dicom_series(dcm_dir)

    n_slices     = ref_image.GetSize()[2]
    dcm_w, dcm_h = ref_image.GetSize()[:2]
    spacing      = ref_image.GetSpacing()
    origin       = ref_image.GetOrigin()
    direction    = ref_image.GetDirection()

    print(f"  Slices: {n_slices}   In-plane: {dcm_w}×{dcm_h} px")
    print(f"  Spacing: xy={spacing[0]:.6f} mm  z={spacing[2]:.4f} mm")

    dcm_files = {
        instance_from_dcm(p): p
        for p in dcm_dir.iterdir()
        if p.suffix.lower() == ".dcm"
    }
    print(f"\nAnnotated DCM files found: {len(dcm_files)}")

    target_hw    = (dcm_h, dcm_w)
    label_volume = np.zeros((n_slices, dcm_h, dcm_w), dtype=np.uint8)
    results_dir  = output_path.parent / "slice_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_annotated = 0
    print()
    for slice_idx, inst_num, _ in slice_map:
        dcm_path = dcm_files.get(inst_num)

        if dcm_path is None:
            print(f"  slice {slice_idx:3d}  inst={inst_num:4d}  [no DCM — blank]")
            continue

        mask = extract_mask(dcm_path, target_hw, results_dir)
        label_volume[slice_idx] = mask

        has_ann = bool(mask.any())
        n_annotated += int(has_ann)
        n_regions = int(np.max(cv2.connectedComponents(mask)[1]))
        print(f"  slice {slice_idx:3d}  inst={inst_num:4d}  "
              f"{'ANNOTATED' if has_ann else 'blank':10s}"
              + (f"  tumours≈{n_regions}" if has_ann else ""))

    print(f"\nAnnotated slices : {n_annotated} / {n_slices}")
    print(f"Per-slice results: {results_dir}/")

    label_img = sitk.GetImageFromArray(label_volume)
    label_img.SetSpacing(spacing)
    label_img.SetOrigin(origin)
    label_img.SetDirection(direction)
    label_img.SetMetaData("intent_name", "label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(label_img, str(output_path))
    print(f"Saved label NRRD   → {output_path}")

    nii_path = output_path.with_suffix("").with_suffix(".nii.gz")
    sitk.WriteImage(label_img, str(nii_path))
    print(f"Saved label NIfTI  → {nii_path}")

    return label_volume


# ─────────────────────────── CLI ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a 3D NRRD/NIfTI label from crosshair-annotated DICOM slices."
    )
    parser.add_argument("--dcm-dir", required=True,
                        help="Folder of annotated DICOM files")
    parser.add_argument("--output", "-o", default="label.nrrd",
                        help="Output NRRD path (default: label.nrrd)")
    args = parser.parse_args()

    build_volume(
        dcm_dir=Path(args.dcm_dir),
        output_path=Path(args.output),
    )
