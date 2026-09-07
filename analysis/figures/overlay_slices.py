#!/usr/bin/env python3
"""
overlay_slices.py — per-slice SAM2-vs-nnU-Net comparison for whole volumes.

Purpose: like overlay_val.py but writes ONE PNG per slice into a per-case subfolder,
covering every slice (not just tumour-bearing ones). Green = SAM2 weak label,
red = nnU-Net. Sources use the nnU-Net v2 layout (imagesTr / labelsTr / a preds dir).

Use:
    python overlay_slices.py --images $nnUNet_raw/Dataset501_Tumour/imagesTr \\
        --labels $nnUNet_raw/Dataset501_Tumour/labelsTr --preds <crossval_dir> --out slice_review
    # --cases <ids...> to limit;  --tumour-only to skip empty slices
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))   # (Z,H,W)


def dice(a, b):
    a, b = a > 0, b > 0
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * (a & b).sum() / d


def save_slice(case, z, sl, gt_z, pred_z, out_path):
    """One MR slice with green GT + red prediction contours -> PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = np.percentile(sl.astype(float), [1, 99])
    ax.imshow(np.clip(sl, lo, hi), cmap="gray")
    if (gt_z > 0).any():
        ax.contour(gt_z > 0, colors="lime", linewidths=1.2)
    if (pred_z > 0).any():
        ax.contour(pred_z > 0, colors="red", linewidths=1.2)
    ax.set_title(f"{case}   slice {z}   Dice={dice(gt_z, pred_z):.3f}\n"
                 "green = SAM2 label, red = nnU-Net", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def resolve_cases(args):
    if args.cases:
        return list(args.cases)
    return sorted(p.name.split(".")[0] for p in Path(args.preds).glob("*.nii.gz"))


def main():
    ap = argparse.ArgumentParser(
        description="Per-slice SAM2-vs-nnU-Net overlays, one subfolder per case.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="slice_review")
    ap.add_argument("--cases", nargs="*", help="explicit case ids (default: all predictions)")
    ap.add_argument("--tumour-only", action="store_true",
                    help="skip slices with no tumour in gt or pred")
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    imgd, lbld, prd = Path(a.images), Path(a.labels), Path(a.preds)

    total = 0
    for case in resolve_cases(a):
        ip = imgd / f"{case}_0000.nii.gz"
        lp = lbld / f"{case}.nii.gz"
        pp = prd / f"{case}.nii.gz"
        if not (ip.exists() and pp.exists()):
            print(f"[skip] {case}: missing image or prediction"); continue

        img = load(ip)
        pred = load(pp)
        gt = load(lp) if lp.exists() else np.zeros_like(pred)

        case_dir = outdir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        pad = len(str(img.shape[0] - 1))

        made = 0
        for z in range(img.shape[0]):
            if a.tumour_only and not ((gt[z] > 0).any() or (pred[z] > 0).any()):
                continue
            save_slice(case, z, img[z], gt[z], pred[z],
                       case_dir / f"slice_{z:0{pad}d}.png")
            made += 1
        total += made
        print(f"[ok]   {case}: {made} slices -> {case_dir}/   (vol Dice={dice(gt, pred):.3f})")

    print(f"\n{total} slice PNG(s) across {len(resolve_cases(a))} case(s) -> {outdir}")


if __name__ == "__main__":
    main()
