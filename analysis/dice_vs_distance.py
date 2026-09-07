#!/usr/bin/env python3
"""Per-slice Dice vs distance from the seed slice |z - z0| (Cai-comparable axis).

Purpose: for each reference study, bin the 2-D Dice (pseudo-label vs GT) on each GT
slice by k = |z - z0| to the nearest seed slice, for two pseudo-labels — unconstrained
(raw SAM2, _raw) and constrained (_clip) — so the plot shows what propagation buys
(k=0 vs k>0), what the constraints buy (gap between the lines), and where drift sets in.
GT is resampled onto the image grid first (same fix as gt_compare).

Use:
    python dice_vs_distance.py                 # all defaults
    python dice_vs_distance.py --max-k 6       # x-axis cap
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

# GT filename -> case id (same mapping as gt_compare.py)
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

# Cai et al. reported anchor points, for reference on the same axis
CAI = {0: 0.92, 4: 0.55}


def resample_to(ref, moving):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    r.SetTransform(sitk.Transform())
    return r.Execute(sitk.Cast(moving, sitk.sitkUInt8))


def dice2d(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def seed_volume(masks_dir: Path, shape):
    """Stack the per-frame seed PNGs into a z,y,x boolean seed volume."""
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
    ap.add_argument("--max-k", type=int, default=6, help="x-axis cap for the plot")
    ap.add_argument("--no-cai", action="store_true",
                    help="Omit the Cai et al. reference anchors; write the figure "
                         "to fig_dice_vs_distance_nocai.* instead.")
    ap.add_argument("--constrained-only", action="store_true",
                    help="Plot only the constrained (final) line — never load or "
                         "draw the unconstrained raw SAM2 curve. Writes to "
                         "fig_dice_vs_distance_constrained.* instead.")
    a = ap.parse_args()

    img_dir = Path(a.images); sam = Path(a.sam_dir); seed_root = Path(a.seed_root)
    gt_dir = Path(a.gt_dir)

    # PER-LESION: each GT 3-D connected component is one lesion with its own seed
    # slice z0. Distance k=|z-z0| is measured along THAT lesion; the pseudo-label
    # is isolated to the lesion's dilated 2-D footprint so a neighbouring lesion
    # cannot leak into the score. Only lesions whose seed spatially overlaps them
    # are scored (an unseeded GT lesion is a recall miss, not a propagation-decay
    # sample). k=0 is the seeded slice; k>0 are propagated-only slices.
    ROI_DIL = 6                                  # px dilation of the lesion footprint
    rows = []                                    # per case/lesion/slice/method
    n_lesion_scored = n_lesion_unseeded = 0
    for fn, case in GT_MAP.items():
        ip = img_dir / f"{case}_0000.nii.gz"
        gp = gt_dir / fn
        mp = seed_root / case / "masks"
        # Prefer a single-run dual pair (<case>_dual + <case>_dual_raw) so the
        # constrained and unconstrained lines share ONE SAM2 pass — no cross-run
        # jitter. Fall back to the standalone _clip (constrained only) otherwise.
        dual_c, dual_r = sam / f"{case}_dual.nii.gz", sam / f"{case}_dual_raw.nii.gz"
        if dual_c.exists() and dual_r.exists():
            cp, rp = dual_c, dual_r
        else:
            cp, rp = sam / f"{case}_clip.nii.gz", None
        if a.constrained_only:
            rp = None                             # constrained line only
        if not (ip.exists() and gp.exists() and cp.exists() and mp.is_dir()):
            print(f"[skip] {case}: missing image/gt/clip/seed"); continue
        ref = sitk.ReadImage(str(ip))
        gt = sitk.GetArrayFromImage(resample_to(ref, sitk.ReadImage(str(gp)))) > 0
        con = sitk.GetArrayFromImage(sitk.ReadImage(str(cp))) > 0
        raw = sitk.GetArrayFromImage(sitk.ReadImage(str(rp))) > 0 if rp else None
        seed = seed_volume(mp, gt.shape)
        methods = [("constrained", con)] + ([("unconstrained", raw)] if raw is not None else [])

        lab, nles = ndimage.label(gt)            # 3-D lesions (6-connectivity)
        scored = 0
        for i in range(1, nles + 1):
            L = lab == i
            zr = np.where(L.any(axis=(1, 2)))[0]
            # z0 = seeded slice with the largest overlap with this lesion
            ov = {int(z): int((seed[z] & L[z]).sum()) for z in zr}
            ov = {z: v for z, v in ov.items() if v > 0}
            if not ov:
                n_lesion_unseeded += 1; continue
            z0 = max(ov, key=ov.get)
            roi = ndimage.binary_dilation(L.any(axis=0), iterations=ROI_DIL)
            for z in zr:
                k = abs(int(z) - z0)
                g = L[z]
                for name, vol in methods:
                    p = vol[z] & roi
                    rows.append(dict(case=case, lesion=i, z=int(z), k=k,
                                     method=name, dice=dice2d(g, p)))
            scored += 1
        n_lesion_scored += scored
        print(f"[ok]  {case:16} lesions={nles} (scored {scored})  "
              f"raw={'yes' if raw is not None else 'NO'}")
    print(f"\nLesions scored: {n_lesion_scored}   unseeded (excluded): {n_lesion_unseeded}")

    if not rows:
        print("No data."); return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv_out = Path(a.csv_out); csv_out.mkdir(parents=True, exist_ok=True)

    # ---- per-slice CSV ----
    with open(csv_out / "dice_vs_distance_slices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "lesion", "z", "k", "method", "dice"])
        w.writeheader(); w.writerows(rows)

    # ---- aggregate mean Dice per (method, k) ----
    agg = defaultdict(list)
    nles = defaultdict(set)                       # distinct lesions per (method, k)
    for r in rows:
        agg[(r["method"], r["k"])].append(r["dice"])
        nles[(r["method"], r["k"])].add((r["case"], r["lesion"]))
    methods = ["constrained"] + (["unconstrained"] if any(r["method"] == "unconstrained" for r in rows) else [])
    kmax = max(r["k"] for r in rows)

    with open(csv_out / "dice_vs_distance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "k", "n_slices", "mean_dice", "std_dice"])
        for m in methods:
            for k in range(kmax + 1):
                v = [d for d in agg[(m, k)] if d == d]
                if v:
                    w.writerow([m, k, len(v), f"{np.mean(v):.4f}", f"{np.std(v):.4f}"])

    # ---- plot ----
    COL = {"constrained": "#1b7837", "unconstrained": "#b2182b"}
    LAB = {"constrained": "median (IQR band)", "unconstrained": "unconstrained"}
    # Native size = final size on the page: 5.0×3.0 in fits \columnwidth-ish and is
    # included WITHOUT scaling, so serif text comes out at these exact point sizes
    # and matches the body. acmart body ≈ 9 pt.
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for mi, m in enumerate(methods):
        ks, med, mean, q25, q75 = [], [], [], [], []
        for k in range(min(a.max_k, kmax) + 1):
            v = [d for d in agg[(m, k)] if d == d]
            if v:
                ks.append(k); med.append(np.median(v)); mean.append(np.mean(v))
                q25.append(np.percentile(v, 25)); q75.append(np.percentile(v, 75))
        ks = np.array(ks); med = np.array(med); mean = np.array(mean)
        q25 = np.array(q25); q75 = np.array(q75)
        ax.fill_between(ks, q25, q75, color=COL[m], alpha=0.13, linewidth=0)
        # Faint mean alongside the median: the median–mean gap is where the failures
        # live — a minority of bad slices pulls the mean below the median.
        ax.plot(ks, mean, "--", color=COL[m], lw=1.2, alpha=0.55, zorder=4)
        ax.plot(ks, med, "-o", color=COL[m], lw=2.2, ms=6, label=LAB[m], zorder=5)
        if mi == 0:
            # k=0 ceiling: median Dice on the seed slice, set by the ellipse fit
            # BEFORE propagation. The finding is that the line stays near it.
            y0 = med[0]
            ax.axhline(y0, ls=":", color=COL[m], lw=1.3, alpha=0.85, zorder=2)
            ax.text(ks[0] + 0.12, y0 + 0.022, f"$k=0$ ceiling  {y0:.2f}",
                    ha="left", va="bottom", fontsize=7.5, color=COL[m])
            # Lesion counts (distinct lesions contributing at each depth) above the band.
            for k, yb in zip(ks, q75):
                ax.annotate(f"n={len(nles[(m, k)])}", (k, yb), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=6.5, color="#333333")
    # Cai et al. anchors (optional)
    if not a.no_cai:
        ck = sorted(CAI); cv = [CAI[k] for k in ck]
        ax.plot(ck, cv, "--D", color="#666666", lw=1.6, ms=7, mfc="white",
                label="Cai et al. (reported)", zorder=4)
        for k, y in CAI.items():
            ax.annotate(f"{y:.2f}", (k, y), textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color="#444444")

    ax.set_xlabel("distance from the lesion's seed slice  (slices)")
    ax.set_ylabel("per-slice Dice vs GT")
    ax.set_xticks(range(min(a.max_k, kmax) + 1))
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", ls="--", lw=1.2, alpha=0.7))
    labels.append("mean")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    stem = ("fig_dice_vs_distance_constrained" if a.constrained_only
            else "fig_dice_vs_distance_nocai" if a.no_cai
            else "fig_dice_vs_distance")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)

    print(f"\nWrote {out}/{stem}.pdf/.png")
    print(f"      {csv_out}/dice_vs_distance.csv  and  dice_vs_distance_slices.csv")
    for m in methods:
        line = "  ".join(f"k{k}={np.median([d for d in agg[(m,k)] if d==d]):.3f}"
                         for k in range(min(a.max_k, kmax) + 1) if agg[(m, k)])
        print(f"  {m:14}: {line}")


if __name__ == "__main__":
    main()
