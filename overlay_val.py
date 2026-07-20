#!/usr/bin/env python3
"""
overlay_val.py — qualitative figures for nnU-Net validation predictions
------------------------------------------------------------------------
For each requested case, plot the raw MR slice with the SAM2 weak-label
contour (green) and the nnU-Net prediction contour (red) overlaid, on the
slices where either mask has tumour. One PNG per case → drop into slides.

Sources (nnU-Net v2 layout):
  raw image   : <nnUNet_raw>/Dataset501_Tumour/imagesTr/<case>_0000.nii.gz
  weak label  : <nnUNet_raw>/Dataset501_Tumour/labelsTr/<case>.nii.gz
  prediction  : <fold_dir>/validation/<case>.nii.gz

Usage
-----
  python overlay_val.py \\
      --images   $nnUNet_raw/Dataset501_Tumour/imagesTr \\
      --labels   $nnUNet_raw/Dataset501_Tumour/labelsTr \\
      --preds    $nnUNet_results/Dataset501_Tumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \\
      --out      overlays --cases brac62451b_00 dabc71a_00 --n-slices 3

  # or auto-pick the N best + N worst by Dice from summary.json:
  python overlay_val.py --images ... --labels ... --preds ... \\
      --summary <fold_dir>/validation/summary.json --auto 3 --out overlays
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))   # (Z,H,W)


def pick_slices(gt, pred, n):
    """Slice indices with the most tumour (union of gt+pred), up to n."""
    fg = ((gt > 0) | (pred > 0))
    per_z = fg.reshape(fg.shape[0], -1).sum(1)
    zs = np.argsort(per_z)[::-1]
    zs = [int(z) for z in zs if per_z[z] > 0][:n]
    return sorted(zs)


def dice(a, b):
    a, b = a > 0, b > 0
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * (a & b).sum() / d


def make_figure(case, img, gt, pred, zs, out_path):
    n = len(zs)
    if n == 0:
        return False
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, z in zip(axes[0], zs):
        sl = img[z].astype(float)
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray")
        if (gt[z] > 0).any():
            ax.contour(gt[z] > 0, colors="lime", linewidths=1.2)
        if (pred[z] > 0).any():
            ax.contour(pred[z] > 0, colors="red", linewidths=1.2)
        ax.set_title(f"slice {z}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case}   Dice={dice(gt, pred):.3f}   "
                 "green = SAM2 label, red = nnU-Net", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def resolve_cases(args):
    if args.cases:
        return list(args.cases)
    if args.summary:
        d = json.load(open(args.summary))
        rows = sorted((m["metrics"]["1"]["Dice"],
                       Path(m["reference"]).name.split(".")[0])
                      for m in d["metric_per_case"])
        picked = [c for _, c in rows[: args.auto]] + [c for _, c in rows[-args.auto:]]
        return list(dict.fromkeys(picked))          # de-dup, keep order
    # else: every prediction present
    return sorted(p.name.split(".")[0] for p in Path(args.preds).glob("*.nii.gz"))


def main():
    ap = argparse.ArgumentParser(description="Overlay nnU-Net val predictions on MR slices.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="overlays")
    ap.add_argument("--cases", nargs="*", help="explicit case ids")
    ap.add_argument("--summary", help="summary.json to auto-pick best/worst by Dice")
    ap.add_argument("--auto", type=int, default=3, help="N best + N worst when using --summary")
    ap.add_argument("--n-slices", type=int, default=3, help="slices per case")
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    imgd, lbld, prd = Path(a.images), Path(a.labels), Path(a.preds)

    made = 0
    for case in resolve_cases(a):
        ip = imgd / f"{case}_0000.nii.gz"
        lp = lbld / f"{case}.nii.gz"
        pp = prd / f"{case}.nii.gz"
        if not (ip.exists() and pp.exists()):
            print(f"[skip] {case}: missing image or prediction"); continue
        img = load(ip)
        pred = load(pp)
        gt = load(lp) if lp.exists() else np.zeros_like(pred)
        zs = pick_slices(gt, pred, a.n_slices)
        op = outdir / f"{case}.png"
        if make_figure(case, img, gt, pred, zs, op):
            print(f"[ok]   {op}   Dice={dice(gt, pred):.3f}"); made += 1
        else:
            print(f"[none] {case}: no tumour slices")
    print(f"\n{made} figure(s) -> {outdir}")


if __name__ == "__main__":
    main()
