#!/usr/bin/env python3
"""
prepare_sam2.py  —  Build SAM2 propagation inputs from annotated DICOMs
-----------------------------------------------------------------------
For each slice in the DICOM series:
  1. Reads the clean frame from the raw (unannotated) DICOM series.
  2. Detects crosshair annotations from the matching ROI (annotated) DICOM.
  3. If crosshairs are found, runs SAM2's *image* predictor on the clean frame:
       - crosshair centre  → positive point prompt
       - arm extent (AABB) → bounding-box constraint  (when arms are long enough)
     Using the raw frame (no annotation lines) prevents SAM2 from segmenting
     the crosshair marks themselves instead of the underlying tumour tissue.
  4. Saves the resulting pixel-accurate mask as a PNG prompt for sam2_propagate.py.
  5. Writes frame_info.csv and a reference NIfTI (spatial metadata for propagation).

Usage
-----
    python prepare_sam2.py \\
        --roi-dcm-dir  brac46551b_5316 \\
        --raw-dcm-dir  brac46551b_raw \\
        --sam2-dir     sam2_test/sam2 \\
        --output       sam2_input/

    # Faster image-predictor model:
    python prepare_sam2.py ... --model small
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Arms shorter than this are treated as centre-point-only markers (no box).
_MIN_ARM_LEN = 25   # pixels

MODEL_CFGS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


# ─────────────────────────── Device ──────────────────────────────────────────

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────── DICOM helpers ───────────────────────────────────

def read_series_order(dcm_dir: Path):
    """Return slices in SimpleITK GDCM order: [(slice_idx, instance_number, Path), ...]."""
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


def dcm_to_rgb(path: Path) -> np.ndarray:
    """
    Read a DICOM and return 8-bit RGB (H × W × 3).
    Applies DICOM WindowCenter/WindowWidth when present; falls back to min-max.
    """
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth",  None)
        if wc is not None and ww is not None:
            wc = float(wc[0] if hasattr(wc, "__len__") else wc)
            ww = float(ww[0] if hasattr(ww, "__len__") else ww)
            lo, hi = wc - ww / 2.0, wc + ww / 2.0
        else:
            lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        arr = arr.astype(np.uint8)
    return arr


def build_roi_index(roi_dcm_dir: Path) -> dict:
    """Return {instance_number: Path} for every DICOM in the ROI folder."""
    index = {}
    for p in roi_dcm_dir.iterdir():
        if p.suffix.lower() == ".dcm":
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            inst = int(getattr(ds, "InstanceNumber", 0))
            index[inst] = p
    return index


def build_reference_nifti(dcm_dir: Path) -> sitk.Image:
    """
    Build a blank 3-D SimpleITK image with the correct spatial metadata
    (spacing, origin, direction) from the DICOM series.
    Used by sam2_propagate.py for CopyInformation().
    """
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    reader.SetFileNames(dcm_files)
    ref = reader.Execute()
    n = ref.GetSize()[2]
    h, w = ref.GetSize()[1], ref.GetSize()[0]
    blank = sitk.GetImageFromArray(np.zeros((n, h, w), dtype=np.uint8))
    blank.CopyInformation(ref)
    return blank


# ─────────────────────────── Crosshair detection ─────────────────────────────

def detect_crosshairs(image_rgb: np.ndarray) -> list:
    """
    Detect crosshair annotations in one frame via extracter.py.
    Returns a list of crosshair dicts (center, major_len, minor_len, angle_deg).
    """
    from extracter import detect_segments_by_color, pair_into_crosshairs
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    segs_by_color = detect_segments_by_color(img_bgr)
    return pair_into_crosshairs(segs_by_color)


# ─────────────────────────── Bounding-box helper ─────────────────────────────

def crosshair_box(cx: float, cy: float,
                  major_len: float, minor_len: float,
                  angle_deg: float,
                  margin: float = 0.10) -> np.ndarray | None:
    """
    Axis-aligned bounding box of the rotated ellipse defined by the crosshair arms,
    expanded by `margin` on each side.

    For an ellipse with semi-axes a, b rotated by angle θ the AABB half-extents are:
        hw = sqrt((a·cosθ)² + (b·sinθ)²)
        hh = sqrt((a·sinθ)² + (b·cosθ)²)

    Returns np.array([x1, y1, x2, y2]) or None when arms are too short to be useful.
    """
    if major_len < _MIN_ARM_LEN:
        return None
    a = major_len / 2.0
    b = max(minor_len / 2.0, 1.0)
    theta = np.radians(angle_deg)
    hw = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
    hh = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    hw *= (1.0 + margin)
    hh *= (1.0 + margin)
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)


# ─────────────────────────── SAM2 image predictor ────────────────────────────

def load_image_predictor(sam2_dir: Path, model: str, device):
    sys.path.insert(0, str(sam2_dir))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg, ckpt_name = MODEL_CFGS[model]
    ckpt_path = sam2_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run  {sam2_dir}/checkpoints/download_ckpts.sh  to download it."
        )
    print(f"Loading SAM2 image predictor '{model}' on {device}")
    sam2_model = build_sam2(cfg, str(ckpt_path), device=device)
    # max_hole_area / max_sprinkle_area clean up small interior holes and noise specks
    return SAM2ImagePredictor(sam2_model, max_hole_area=100, max_sprinkle_area=100)


def predict_mask_for_crosshair(predictor, ch: dict, h: int, w: int) -> np.ndarray:
    """
    Run the SAM2 image predictor for one detected crosshair.

    Strategy:
      - Always use the crosshair centre as a positive point prompt.
      - If the arms are long enough, also supply the AABB as a box constraint.
        With a box SAM2 returns one stable mask (multimask_output=False).
      - Without a box (short marker arms) SAM2 returns three candidate masks
        and we pick the one with the highest confidence score.

    Returns a bool mask (H × W).
    """
    cx, cy = ch["center"]
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1],        dtype=np.int32)

    box = crosshair_box(cx, cy, ch["major_len"], ch["minor_len"], ch["angle_deg"])

    if box is not None:
        # Clamp to image bounds
        box[0] = max(0.0, box[0]);  box[1] = max(0.0, box[1])
        box[2] = min(float(w), box[2]);  box[3] = min(float(h), box[3])
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None],           # shape (1, 4) as SAM2 expects
            multimask_output=False,
        )
    else:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    return masks[int(np.argmax(scores))].astype(bool)


# ─────────────────────────── Main ────────────────────────────────────────────

def prepare(roi_dcm_dir: Path, raw_dcm_dir: Path, sam2_dir: Path,
            output_dir: Path, model: str = "large", jpeg_quality: int = 95):

    frames_dir = output_dir / "frames"
    masks_dir  = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # extracter.py lives next to this script
    sys.path.insert(0, str(Path(__file__).parent))

    device      = select_device()
    # Slice ordering and spatial metadata come from the raw series
    slice_order = read_series_order(raw_dcm_dir)
    n_slices    = len(slice_order)
    pad         = len(str(n_slices - 1))

    # ROI DICOMs indexed by instance number for fast lookup
    roi_index = build_roi_index(roi_dcm_dir)

    predictor = load_image_predictor(sam2_dir, model, device)

    # Enable hardware-appropriate autocast and keep the context alive
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif device.type == "mps":
        autocast_ctx = torch.autocast("mps", dtype=torch.float16)
    else:
        autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)

    csv_rows        = []
    n_prompt_frames = 0

    print(f"\nProcessing {n_slices} slices → {output_dir}/")

    with autocast_ctx:
        for slice_idx, inst_num, raw_path in slice_order:
            stem = f"{slice_idx:0{pad}d}"

            # ── Frame image (clean, no annotation lines) ──────────────────────
            rgb = dcm_to_rgb(raw_path)
            h, w = rgb.shape[:2]
            cv2.imwrite(
                str(frames_dir / f"{stem}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

            # ── Detect crosshairs from ROI DICOM ─────────────────────────────
            roi_path   = roi_index.get(inst_num)
            crosshairs = detect_crosshairs(dcm_to_rgb(roi_path)) if roi_path else []
            has_mask   = len(crosshairs) > 0

            if has_mask:
                # Set the clean frame — SAM2 segments tissue, not annotation lines
                predictor.set_image(rgb)

                combined = np.zeros((h, w), dtype=bool)
                for ch in crosshairs:
                    m = predict_mask_for_crosshair(predictor, ch, h, w)
                    combined |= m

                cv2.imwrite(str(masks_dir / f"{stem}.png"),
                            combined.astype(np.uint8) * 255)
                n_prompt_frames += 1
                status = (f"{len(crosshairs)} tumor(s)  "
                          f"mask={combined.sum()} px  "
                          f"({100*combined.sum()/(h*w):.1f}%)")
            else:
                status = "no annotation"

            csv_rows.append({
                "frame_idx":       slice_idx,
                "instance_number": inst_num,
                "has_mask":        int(has_mask),
            })
            print(f"  [{stem}]  inst={inst_num:4d}  {status}")

    # ── Reference NIfTI (spatial metadata for sam2_propagate.py) ─────────────
    ref_nifti = build_reference_nifti(raw_dcm_dir)
    ref_path  = output_dir / "reference.nii.gz"
    sitk.WriteImage(ref_nifti, str(ref_path))
    print(f"\nReference NIfTI  : {ref_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / "frame_info.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_idx", "instance_number", "has_mask"])
        writer.writeheader()
        writer.writerows(csv_rows)

    prompt_indices = [r["frame_idx"] for r in csv_rows if r["has_mask"]]
    print(f"Frames written   : {n_slices}  → {frames_dir}/")
    print(f"SAM2 seed masks  : {n_prompt_frames}  → {masks_dir}/")
    print(f"Frame info CSV   : {csv_path}")
    print(f"Prompt indices   : {prompt_indices}")


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build SAM2 propagation inputs from annotated DICOMs: "
            "detect crosshairs, generate pixel-accurate seed masks with "
            "SAM2 image predictor."
        )
    )
    parser.add_argument("--roi-dcm-dir", required=True,
                        help="Folder of annotated (ROI) DICOM files — used for crosshair detection")
    parser.add_argument("--raw-dcm-dir", required=True,
                        help="Folder of raw (unannotated) DICOM files — used for frame images and SAM2")
    parser.add_argument("--sam2-dir", default="sam2_test/sam2",
                        help="Path to SAM2 repo root (default: sam2_test/sam2)")
    parser.add_argument("--output",   default="sam2_input",
                        help="Output directory (default: sam2_input/)")
    parser.add_argument("--model",    default="large",
                        choices=["tiny", "small", "base_plus", "large"],
                        help="SAM2.1 model size (default: large)")
    parser.add_argument("--jpeg-quality", type=int, default=95, metavar="Q",
                        help="JPEG quality for frame images 1-100 (default: 95)")
    args = parser.parse_args()

    prepare(
        roi_dcm_dir=Path(args.roi_dcm_dir),
        raw_dcm_dir=Path(args.raw_dcm_dir),
        sam2_dir=Path(args.sam2_dir),
        output_dir=Path(args.output),
        model=args.model,
        jpeg_quality=args.jpeg_quality,
    )
