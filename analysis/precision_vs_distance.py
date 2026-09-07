#!/usr/bin/env python3
"""Per-lesion precision vs distance from the seed slice |z - z0|.

Purpose: companion to dice_vs_distance.py with the same per-lesion set-up but the
metric is precision(z) = |pred ∩ GT_lesion| / |pred ∩ ROI| — of the voxels a method
places near a lesion, what fraction are truly it. This is where raw and constrained
diverge: raw leakage grows with distance while the ellipse-clipped precision stays high.
Scored only where BOTH methods predict in the ROI; prefers the single-run dual pair.

Use:
    python precision_vs_distance.py                 # --max-k caps x
    python precision_vs_distance.py --no-cai
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GT_MAP = {
    "brac5168.1a_Segmentation.nii": "brac51681a_01",
    "brac52351a_Segmentation.nii":  "brac52351a_01",
    "brac52351d_Segmentation.nii":  "brac52351d_00",
    "brac54131f_Segmentation.nii":  "brac54131f_00",
    "brac62451c_Segmentation.nii":  "brac62451c_03",
    "brac62721b_Segmentation.nii":  "brac62721b_00",
    "brac62731a_Segmentation.nii":  "brac62731a_01",
    "dabj171dSegmentation.nii":     "dabj171d_02",
    "dabj201c_Segmentation.nii":    "dabj201c_01",
    "dabj91c_Segmentation.nii":     "dabj91c_02",
}


def resample_to(ref, moving):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    r.SetTransform(sitk.Transform())
    return r.Execute(sitk.Cast(moving, sitk.sitkUInt8))


def seed_volume(masks_dir: Path, shape):
    sv = np.zeros(shape, dtype=bool)
    for p in masks_dir.glob("*.png"):
        z = int(p.stem)
        m = cv2.imread(str(p), 0) > 0
        if 0 <= z < shape[0] and m.shape == shape[1:]:
            sv[z] = m
    return sv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="GT_validation")
    ap.add_argument("--images", default="nnUNet_raw/Dataset502_Tumour/imagesTr")
    ap.add_argument("--sam-dir", default=str(Path.home() / "Downloads/output/out"))
    ap.add_argument("--seed-root", default=str(Path.home() / "Downloads/output/sam2_input"))
    ap.add_argument("--out", default="report_figures")
    ap.add_argument("--csv-out", default="gt_report")
    ap.add_argument("--roi-dil", type=int, default=6)
    ap.add_argument("--max-k", type=int, default=6)
    ap.add_argument("--no-cai", action="store_true")
    a = ap.parse_args()

    img_dir = Path(a.images); sam = Path(a.sam_dir); seed_root = Path(a.seed_root)
    gt_dir = Path(a.gt_dir)

    rows = []
    for fn, case in GT_MAP.items():
        ip = img_dir / f"{case}_0000.nii.gz"
        gp = gt_dir / fn
        mp = seed_root / case / "masks"
        dual_c, dual_r = sam / f"{case}_dual.nii.gz", sam / f"{case}_dual_raw.nii.gz"
        if dual_c.exists() and dual_r.exists():
            cp, rp = dual_c, dual_r
        else:
            cp, rp = sam / f"{case}_clip.nii.gz", None
        if not (ip.exists() and gp.exists() and cp.exists() and mp.is_dir()):
            print(f"[skip] {case}: missing input"); continue
        ref = sitk.ReadImage(str(ip))
        gt = sitk.GetArrayFromImage(resample_to(ref, sitk.ReadImage(str(gp)))) > 0
        con = sitk.GetArrayFromImage(sitk.ReadImage(str(cp))) > 0
        raw = sitk.GetArrayFromImage(sitk.ReadImage(str(rp))) > 0 if rp else None
        seed = seed_volume(mp, gt.shape)
        methods = [("constrained", con)] + ([("unconstrained", raw)] if raw is not None else [])

        lab, nles = ndimage.label(gt)
        for i in range(1, nles + 1):
            L = lab == i
            zr = np.where(L.any(axis=(1, 2)))[0]
            ov = {int(z): int((seed[z] & L[z]).sum()) for z in zr}
            ov = {z: v for z, v in ov.items() if v > 0}
            if not ov:
                continue
            z0 = max(ov, key=ov.get)
            roi = ndimage.binary_dilation(L.any(axis=0), iterations=a.roi_dil)
            for z in zr:
                k = abs(int(z) - z0)
                proi = {name: (vol[z] & roi) for name, vol in methods}
                # SHARED SET: score a slice only where EVERY method predicts in the
                # ROI, so the precision lines are on the identical slices (precision
                # is undefined without a prediction). Since constrained ⊆ raw, this
                # drops the raw-only leak slices; it is the fair same-set comparison.
                if any(p.sum() == 0 for p in proi.values()):
                    continue
                for name, p_roi in proi.items():
                    denom = int(p_roi.sum())
                    tp = int((p_roi & L[z]).sum())
                    rows.append(dict(case=case, lesion=i, z=int(z), k=k,
                                     method=name, precision=tp / denom))
        print(f"[ok]  {case:16} raw={'yes' if raw is not None else 'NO'}")

    if not rows:
        print("No data."); return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv_out = Path(a.csv_out); csv_out.mkdir(parents=True, exist_ok=True)
    with open(csv_out / "precision_vs_distance_slices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "lesion", "z", "k", "method", "precision"])
        w.writeheader(); w.writerows(rows)

    agg = defaultdict(list)
    nles = defaultdict(set)                       # distinct lesions per (method, k)
    for r in rows:
        agg[(r["method"], r["k"])].append(r["precision"])
        nles[(r["method"], r["k"])].add((r["case"], r["lesion"]))
    methods = ["constrained"] + (["unconstrained"] if any(r["method"] == "unconstrained" for r in rows) else [])
    kmax = max(r["k"] for r in rows)

    with open(csv_out / "precision_vs_distance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "k", "n_slices", "mean_precision", "std_precision"])
        for m in methods:
            for k in range(kmax + 1):
                v = agg[(m, k)]
                if v:
                    w.writerow([m, k, len(v), f"{np.mean(v):.4f}", f"{np.std(v):.4f}"])

    COL = {"constrained": "#1b7837", "unconstrained": "#b2182b"}
    LAB = {"constrained": "constrained", "unconstrained": "unconstrained"}
    # Native size = final size on the page: 5.0×3.0 in, included WITHOUT scaling so
    # serif text comes out at these exact point sizes and matches the body (≈9 pt).
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for mi, m in enumerate(methods):
        ks, med, mean = [], [], []
        for k in range(min(a.max_k, kmax) + 1):
            v = agg[(m, k)]
            if v:
                ks.append(k); med.append(np.median(v)); mean.append(np.mean(v))
        ks = np.array(ks); med = np.array(med); mean = np.array(mean)
        # Faint mean: the median–mean gap shows failure is concentrated in a
        # minority of slices rather than spread across all of them.
        ax.plot(ks, mean, "--", color=COL[m], lw=1.2, alpha=0.55, zorder=4)
        ax.plot(ks, med, "-o", color=COL[m], lw=2.2, ms=6, label=LAB[m], zorder=5)
        if mi == 0:                              # both methods score the same lesions
            for k in ks:
                ytop = max(np.median(agg[(mm, k)]) for mm in methods if agg[(mm, k)])
                ax.annotate(f"n={len(nles[(m, k)])}", (k, ytop), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=6.5, color="#333333")

    ax.set_xlabel("distance from the lesion's seed slice  (slices)")
    ax.set_ylabel("per-slice precision  (median, mean)")
    ax.set_xticks(range(min(a.max_k, kmax) + 1))
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", ls="--", lw=1.2, alpha=0.7))
    labels.append("mean")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    stem = "fig_precision_vs_distance_nocai" if a.no_cai else "fig_precision_vs_distance"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)

    print(f"\nWrote {out}/{stem}.pdf/.png and {csv_out}/precision_vs_distance.csv")
    for m in methods:
        line = "  ".join(f"k{k}={np.median(agg[(m,k)]):.3f}"
                         for k in range(min(a.max_k, kmax) + 1) if agg[(m, k)])
        print(f"  {m:14}: {line}")


if __name__ == "__main__":
    main()
