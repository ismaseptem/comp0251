#!/usr/bin/env python3
"""
overlay_val.py — qualitative figures for nnU-Net validation predictions.

Purpose: per case, plot the raw MR slice with the SAM2 weak-label contour (green) and
the nnU-Net prediction contour (red) on slices where either has tumour — one PNG per
case for slides. Sources use the nnU-Net v2 layout (imagesTr / labelsTr / a preds dir).

Use:
    python overlay_val.py --images <imagesTr> --labels <labelsTr> --preds <fold>/validation \\
        --out overlays --cases brac62451b_00 dabc71a_00 --n-slices 3
    # or --summary <fold>/validation/summary.json --auto 3 to auto-pick best/worst by Dice
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
    """Slice indices to render.

    n <= 0  → every slice that has tumour in gt OR pred (whole-volume view).
    n  > 0  → the n slices with the most tumour (union of gt+pred).
    """
    fg = ((gt > 0) | (pred > 0))
    per_z = fg.reshape(fg.shape[0], -1).sum(1)
    tumour_zs = [int(z) for z in range(fg.shape[0]) if per_z[z] > 0]
    if n is None or n <= 0:
        return tumour_zs                       # all tumour-bearing slices, in order
    zs = np.argsort(per_z)[::-1]
    zs = [int(z) for z in zs if per_z[z] > 0][:n]
    return sorted(zs)


def dice(a, b):
    a, b = a > 0, b > 0
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * (a & b).sum() / d


def make_figure(case, img, gt, pred, zs, out_path, ncols=6):
    """Render the given slices as a grid montage (green = GT, red = prediction)."""
    n = len(zs)
    if n == 0:
        return False
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for ax, z in zip(axes.ravel(), zs):
        sl = img[z].astype(float)
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray")
        if (gt[z] > 0).any():
            ax.contour(gt[z] > 0, colors="lime", linewidths=1.0)
        if (pred[z] > 0).any():
            ax.contour(pred[z] > 0, colors="red", linewidths=1.0)
        ax.set_title(f"slice {z}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[n:]:          # blank any unused grid cells
        ax.axis("off")
    fig.suptitle(f"{case}   Dice={dice(gt, pred):.3f}   ({n} tumour slices)   "
                 "green = SAM2 label, red = nnU-Net", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def _case_of(m):
    """Case id from a summary.json metric_per_case entry (schema-robust)."""
    for key in ("prediction_file", "reference_file"):
        if m.get(key):
            return Path(m[key]).name.split(".")[0]
    raise KeyError("no prediction_file/reference_file in summary entry")


def resolve_cases(args):
    if args.cases:
        return list(args.cases)
    if args.summary:
        d = json.load(open(args.summary))
        rows = sorted((m["metrics"]["1"]["Dice"], _case_of(m))
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
