#!/usr/bin/env python3
"""
build_volume.py — batch mask extraction + NRRD label builder
-------------------------------------------------------------
Detects crosshair annotations in ROI (Secondary Capture) DICOMs and writes
a 3-D label volume in the coordinate space of the raw MR DICOMs.

Supplying --raw-dir is strongly recommended: the output label will then have
the correct MR pixel spacing/origin/direction and match the frames produced
by prepare_sam2.py. Without --raw-dir the label is written in SC space.

Usage
-----
    python build_volume.py \\
        --roi-dir brac46551b_roi/ \\
        --raw-dir brac46551b_raw/ \\
        --output  label.nrrd
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
    Use SimpleITK GDCM ordering to read a DICOM series.
    Returns (sitk_image, [(slice_idx, instance_number, Path), ...]).
    """
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM series found in {dcm_dir}")
    reader.SetFileNames(dcm_files)
    ref_image = reader.Execute()

    slice_map = []
    for i, f in enumerate(dcm_files):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        inst = int(getattr(ds, "InstanceNumber", i + 1))
        slice_map.append((i, inst, Path(f)))

    return ref_image, slice_map


def z_position(dcm_path: Path) -> float:
    """Return the z-component of ImagePositionPatient for a DICOM file."""
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        return float(ipp[2])
    return float(getattr(ds, "SliceLocation", 0))


def build_z_map(dcm_dir: Path, tol: float = 0.01) -> dict:
    """
    Return {rounded_z: Path} for every DCM in dcm_dir.
    z is rounded to `tol` mm so floating-point differences don't cause misses.
    """
    z_map = {}
    for p in dcm_dir.iterdir():
        if p.suffix.lower() != ".dcm":
            continue
        z = round(z_position(p) / tol) * tol
        z_map[z] = p
    return z_map


# ─────────────────────────── Per-slice extraction ────────────────────────

def extract_mask(dcm_path: Path, target_hw: tuple[int, int],
                 results_dir: Path | None) -> np.ndarray:
    """
    Run the extracter on one annotated (ROI) DICOM.
    Returns a uint8 binary mask (0/1) resized to target_hw = (H, W).
    """
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array
    except Exception as e:
        print(f"    [WARN] cannot read {dcm_path.name}: {e}")
        return np.zeros(target_hw, dtype=np.uint8)

    if arr.ndim == 2:
        img_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)

    _, mask = process_image(dcm_path, output_dir=results_dir, show=False,
                            image_bgr=img_bgr)

    h, w = target_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# ─────────────────────────── Main ────────────────────────────────────────

def build_volume(roi_dir: Path, output_path: Path,
                 raw_dir: Path | None = None,
                 slice_results: Path | None = None):
    """
    roi_dir    – folder of annotated Secondary Capture DICOMs (crosshair detection)
    raw_dir    – folder of raw MR DICOMs (slice ordering + spatial metadata for output)
                 If None, spatial metadata comes from the ROI series (SC space).
    output_path – destination NRRD file
    """

    # ── Reference series: raw MR if available, else ROI SC ───────────────
    if raw_dir is not None:
        print(f"\nReading raw MR series for spatial metadata: {raw_dir}")
        ref_image, slice_map = read_dicom_series(raw_dir)
        print(f"  (crosshair detection will use matching ROI slices from {roi_dir})")
        roi_z_map = build_z_map(roi_dir)
        print(f"  ROI z-positions indexed: {len(roi_z_map)} slices")
    else:
        print(f"\nNo --raw-dir given; using ROI series for slice order + spatial metadata.")
        print(f"  Warning: output will be in Secondary Capture space, not MR space.")
        ref_image, slice_map = read_dicom_series(roi_dir)
        roi_z_map = None

    n_slices      = ref_image.GetSize()[2]
    out_w, out_h  = ref_image.GetSize()[:2]
    spacing       = ref_image.GetSpacing()
    origin        = ref_image.GetOrigin()
    direction     = ref_image.GetDirection()

    print(f"  Output volume : {n_slices} slices  {out_w}×{out_h} px")
    print(f"  Pixel spacing : xy={spacing[0]:.6f} mm  z={spacing[2]:.4f} mm")

    # When using ROI directly (no raw_dir): build instance→path map
    if roi_z_map is None:
        roi_by_inst = {
            int(getattr(pydicom.dcmread(str(p), stop_before_pixels=True),
                        "InstanceNumber", 0)): p
            for p in roi_dir.iterdir() if p.suffix.lower() == ".dcm"
        }
        print(f"\nROI DCM files found: {len(roi_by_inst)}")

    target_hw    = (out_h, out_w)
    label_volume = np.zeros((n_slices, out_h, out_w), dtype=np.uint8)

    if slice_results is not None:
        slice_results.mkdir(parents=True, exist_ok=True)

    n_annotated = 0
    print()

    for slice_idx, inst_num, ref_path in slice_map:
        # ── Find the ROI DCM for this slice ──────────────────────────────
        if roi_z_map is not None:
            # Match by z-position
            z = round(z_position(ref_path) / 0.01) * 0.01
            roi_path = roi_z_map.get(z)
            if roi_path is None:
                print(f"  slice {slice_idx:3d}  z={z:.3f}  [no matching ROI slice — blank]")
                continue
        else:
            # No raw_dir: ref_path IS the roi path
            roi_path = ref_path

        mask = extract_mask(roi_path, target_hw, slice_results)
        label_volume[slice_idx] = mask

        has_ann   = bool(mask.any())
        n_annotated += int(has_ann)
        n_regions = int(np.max(cv2.connectedComponents(mask)[1]))
        print(f"  slice {slice_idx:3d}  inst={inst_num:4d}  "
              f"{'ANNOTATED' if has_ann else 'blank':10s}"
              + (f"  tumours≈{n_regions}" if has_ann else ""))

    print(f"\nAnnotated slices : {n_annotated} / {n_slices}")
    if slice_results is not None:
        print(f"Per-slice results: {slice_results}/")

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
        description="Build a 3D label volume from crosshair-annotated DICOM slices."
    )
    parser.add_argument("--roi-dir", required=True,
                        help="Folder of annotated Secondary Capture DICOMs")
    parser.add_argument("--raw-dir", default=None,
                        help="Folder of raw MR DICOMs (sets output spatial metadata). "
                             "Strongly recommended to avoid SC/MR space mismatch.")
    parser.add_argument("--output", "-o", default="label.nrrd",
                        help="Output NRRD path (default: label.nrrd)")
    parser.add_argument("--slice-results", default=None,
                        help="Directory to save per-slice detection PNGs (optional)")
    args = parser.parse_args()

    build_volume(
        roi_dir=Path(args.roi_dir),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        output_path=Path(args.output),
        slice_results=Path(args.slice_results) if args.slice_results else None,
    )
