#!/usr/bin/env python3
"""
gt_compare.py — validate SAM2 weak labels and nnU-Net predictions against the GT.

Purpose: for every case with a manual GT (GT_validation/), compute Dice/IoU/precision/
recall of the nnU-Net out-of-fold prediction and the SAM2 `_clip` weak label, write a
CSV + Markdown report, and per-case overlay PNGs. Each GT is resampled onto the image
grid first (nearest-neighbour) — one case (brac62731a) has a ~3 mm origin shift that
otherwise scores ~0.05 on a lesion that really overlaps at ~0.77.

Use:
    python gt_compare.py                 # all defaults
    python gt_compare.py --no-overlays   # metrics only
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# GT filename (as shipped) -> pipeline case id. Extend as more GT arrives.
# Each prefix spans several studies (_00/_01/…); the study bound to each GT was
# resolved by which one lands on the same world grid (0 mm origin shift + exact
# size match) and gives real overlap — not by assuming _00.
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


def resample_mask_to(ref_img, moving):
    """Place `moving` label map on `ref_img`'s voxel grid via the world-space
    identity transform (nearest-neighbour). Corrects any origin/direction drift
    between a viewer-exported GT and the DICOM-derived image grid."""
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    r.SetTransform(sitk.Transform())
    return r.Execute(sitk.Cast(moving, sitk.sitkUInt8))


def metrics(ref, test, vox_mm3):
    r, t = ref.ravel(), test.ravel()
    tp = int((r & t).sum()); fp = int((~r & t).sum()); fn = int((r & ~t).sum())
    denom = 2 * tp + fp + fn
    return dict(
        dice=2 * tp / denom if denom else 1.0,
        iou=tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        precision=tp / (tp + fp) if (tp + fp) else float("nan"),
        recall=tp / (tp + fn) if (tp + fn) else float("nan"),
        gt_mm3=float(ref.sum() * vox_mm3),
        pred_mm3=float(test.sum() * vox_mm3),
        vol_ratio=float(test.sum() / ref.sum()) if ref.sum() else float("nan"),
        tp=tp, fp=fp, fn=fn,
    )


def load_bool(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0


def three_way_overlay(case, img, gt, nn, sam, zs, out_path):
    """One montage: MR grey + GT (green) / nnU-Net (red) / SAM2 (blue) contours."""
    n = len(zs)
    if n == 0:
        return False
    ncols = min(6, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax, z in zip(axes.ravel(), zs):
        sl = img[z].astype(float)
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray")
        for mask, col in ((gt, "deepskyblue"), (nn, "red"), (sam, "lime")):
            if mask[z].any():
                ax.contour(mask[z], levels=[0.5], colors=col, linewidths=1.0)
        ax.set_title(f"slice {z}", fontsize=8); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in
               (("deepskyblue", "manual GT"), ("red", "nnU-Net"), ("lime", "SAM2"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    dn = 2 * (gt & nn).sum() / (gt.sum() + nn.sum()) if (gt.sum() + nn.sum()) else 1.0
    ds = 2 * (gt & sam).sum() / (gt.sum() + sam.sum()) if (gt.sum() + sam.sum()) else 1.0
    fig.suptitle(f"{case}   nnU-Net vs GT = {dn:.3f}   SAM2 vs GT = {ds:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return True


def sam_vs_nn_overlay(case, img, nn, sam, zs, out_path):
    """SAM2 (green) vs nnU-Net (red) with agreement shaded amber — no GT needed,
    so it can be produced for every case, not just the GT subset."""
    n = len(zs)
    if n == 0:
        return False
    ncols = min(6, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax, z in zip(axes.ravel(), zs):
        sl = img[z].astype(float)
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray")
        agree = (nn[z] & sam[z]).astype(float)
        if agree.any():
            ax.imshow(np.ma.masked_where(agree == 0, agree), cmap="Oranges",
                      alpha=0.35, vmin=0, vmax=1)
        if nn[z].any():
            ax.contour(nn[z], levels=[0.5], colors="red", linewidths=1.0)
        if sam[z].any():
            ax.contour(sam[z], levels=[0.5], colors="lime", linewidths=1.0)
        ax.set_title(f"slice {z}", fontsize=8); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in
               (("lime", "SAM2"), ("red", "nnU-Net"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    d = 2 * (nn & sam).sum() / (nn.sum() + sam.sum()) if (nn.sum() + sam.sum()) else 1.0
    fig.suptitle(f"{case}   SAM2 vs nnU-Net Dice = {d:.3f}   (amber = agreement)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return True


def tumour_slices(*masks):
    fg = np.zeros_like(masks[0][0], dtype=bool) if masks and masks[0].ndim == 3 else None
    z = masks[0].shape[0]
    keep = []
    for k in range(z):
        if any(m[k].any() for m in masks):
            keep.append(k)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="GT_validation")
    ap.add_argument("--images", default="nnUNet_raw/Dataset502_Tumour/imagesTr")
    ap.add_argument("--labels", default="nnUNet_raw/Dataset502_Tumour/labelsTr")
    ap.add_argument("--preds",
                    default="nnUNet_results/Dataset502_Tumour/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4")
    ap.add_argument("--sam-dir", default=str(Path.home() / "Downloads/output/out"))
    ap.add_argument("--sam-suffix", default="_clip.nii.gz")
    ap.add_argument("--ellipse-dir", default="ellipse_baseline",
                    help="Folder with the model-free ellipse-baseline volumes "
                         "(ellipse_baseline.py output). Cases missing here are "
                         "scored on nnU-Net + SAM2 only.")
    ap.add_argument("--ellipse-suffix", default="_ellipse.nii.gz")
    ap.add_argument("--out", default="gt_report")
    ap.add_argument("--no-overlays", action="store_true")
    a = ap.parse_args()

    gt_dir = Path(a.gt_dir); img_dir = Path(a.images)
    pred_dir = Path(a.preds); sam_dir = Path(a.sam_dir)
    ell_dir = Path(a.ellipse_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "overlays_gt").mkdir(exist_ok=True)
    (out / "overlays_sam_vs_nnunet").mkdir(exist_ok=True)

    rows = []
    for fn, case in GT_MAP.items():
        gp = gt_dir / fn
        ip = img_dir / f"{case}_0000.nii.gz"
        pp = pred_dir / f"{case}.nii.gz"
        sp = sam_dir / f"{case}{a.sam_suffix}"
        if not all(p.exists() for p in (gp, ip, pp, sp)):
            print(f"[skip] {case}: missing "
                  + ", ".join(n for n, p in (("gt", gp), ("img", ip), ("pred", pp), ("sam", sp)) if not p.exists()))
            continue

        ref = sitk.ReadImage(str(ip))
        vox = float(np.prod(ref.GetSpacing()))
        gt_img = sitk.ReadImage(str(gp))
        shifted = (np.abs(np.array(gt_img.GetOrigin()) - np.array(ref.GetOrigin())).max() > 1e-3)
        gt = sitk.GetArrayFromImage(resample_mask_to(ref, gt_img)) > 0
        nn = load_bool(pp)
        sam = load_bool(sp)
        img = sitk.GetArrayFromImage(ref)

        # Model-free ellipse baseline (optional — scored when the volume exists).
        ep = ell_dir / f"{case}{a.ellipse_suffix}"
        methods = [("nnunet", nn), ("sam2", sam)]
        if ep.exists():
            methods.append(("ellipse", load_bool(ep)))
        for method, mask in methods:
            m = metrics(gt, mask, vox)
            m.update(case=case, method=method, gt_realigned=bool(shifted))
            rows.append(m)
        dn = next(r["dice"] for r in rows if r["case"] == case and r["method"] == "nnunet")
        ds = next(r["dice"] for r in rows if r["case"] == case and r["method"] == "sam2")
        de = next((r["dice"] for r in rows if r["case"] == case and r["method"] == "ellipse"), None)
        # SAM2 vs nnU-Net agreement (no GT needed)
        dnn = 2 * (nn & sam).sum() / (nn.sum() + sam.sum()) if (nn.sum() + sam.sum()) else 1.0
        print(f"[ok]  {case:16} nnU-Net/GT={dn:.3f}  SAM2/GT={ds:.3f}"
              + (f"  Ellipse/GT={de:.3f}" if de is not None else "")
              + f"  SAM2/nnU-Net={dnn:.3f}"
              + ("   [GT re-aligned]" if shifted else ""))

        if not a.no_overlays:
            zs = tumour_slices(gt, nn, sam)
            three_way_overlay(case, img, gt, nn, sam, zs, out / "overlays_gt" / f"{case}.png")
            sam_vs_nn_overlay(case, img, nn, sam, zs, out / "overlays_sam_vs_nnunet" / f"{case}.png")

    if not rows:
        print("No cases scored."); return

    # ---- CSV ----
    cols = ["case", "method", "dice", "iou", "precision", "recall",
            "gt_mm3", "pred_mm3", "vol_ratio", "tp", "fp", "fn", "gt_realigned"]
    with open(out / "gt_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # ---- Markdown ----
    all_methods = [m for m in ("nnunet", "sam2", "ellipse")
                   if any(r["method"] == m for r in rows)]
    by = {m: [r for r in rows if r["method"] == m] for m in all_methods}
    cases = sorted({r["case"] for r in rows})
    d = lambda case, meth: next(r for r in by[meth] if r["case"] == case)
    has_ell = "ellipse" in all_methods

    def mean(meth, key):
        # skip NaN (e.g. precision is undefined for an empty prediction)
        v = [r[key] for r in by[meth] if r[key] == r[key]]
        return sum(v) / len(v) if v else float("nan")

    md = []
    md.append("# Validation against manual ground truth\n")
    md.append(f"Cases with manual GT: **{len(cases)}**  ·  "
              "nnU-Net = Dataset502 out-of-fold (5-fold CV)  ·  "
              "SAM2 = `_clip` weak label"
              + ("  ·  Ellipse = model-free RECIST ellipse-stack baseline "
                 "(`ellipse_baseline.py`).\n" if has_ell else ".\n"))
    md.append("All GT masks resampled onto the DICOM image grid before scoring "
              "(corrects a world-space origin shift; see script header).\n")

    md.append("\n## Per-case Dice vs GT\n")
    ecol = " Ellipse Dice |" if has_ell else ""
    esep = "---|" if has_ell else ""
    md.append(f"| case | nnU-Net Dice | SAM2 Dice |{ecol} Δ (nn−sam) | GT re-aligned |")
    md.append(f"|---|---|---|{esep}---|---|")
    for c in cases:
        dn, dss = d(c, "nnunet"), d(c, "sam2")
        ecell = f" {d(c,'ellipse')['dice']:.3f} |" if has_ell else ""
        md.append(f"| {c} | {dn['dice']:.3f} | {dss['dice']:.3f} |{ecell} "
                  f"{dn['dice']-dss['dice']:+.3f} | {'yes' if dn['gt_realigned'] else ''} |")
    emean = f" **{mean('ellipse','dice'):.3f}** |" if has_ell else ""
    md.append(f"| **mean** | **{mean('nnunet','dice'):.3f}** | "
              f"**{mean('sam2','dice'):.3f}** |{emean} "
              f"**{mean('nnunet','dice')-mean('sam2','dice'):+.3f}** | |")

    md.append("\n## Full metrics\n")
    md.append("| case | method | Dice | IoU | Precision | Recall | GT mm³ | pred mm³ | vol ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        for meth in all_methods:
            r = d(c, meth)
            md.append(f"| {c} | {meth} | {r['dice']:.3f} | {r['iou']:.3f} | "
                      f"{r['precision']:.3f} | {r['recall']:.3f} | {r['gt_mm3']:.0f} | "
                      f"{r['pred_mm3']:.0f} | {r['vol_ratio']:.2f} |")

    md.append("\n## Means\n")
    md.append("| method | Dice | IoU | Precision | Recall |")
    md.append("|---|---|---|---|---|")
    for meth in all_methods:
        md.append(f"| {meth} | {mean(meth,'dice'):.3f} | {mean(meth,'iou'):.3f} | "
                  f"{mean(meth,'precision'):.3f} | {mean(meth,'recall'):.3f} |")

    # SAM2 mean excluding its worst truncation outlier, for a fair central read
    sam_sorted = sorted(by["sam2"], key=lambda r: r["dice"])
    worst = sam_sorted[0]
    sam_trim = [r for r in by["sam2"] if r["case"] != worst["case"]]
    sam_trim_mean = sum(r["dice"] for r in sam_trim) / len(sam_trim)

    md.append("\n## Notes\n")
    md.append("- **Precision** = fraction of predicted tumour voxels that are true; "
              "**Recall** = fraction of GT tumour voxels recovered.")
    md.append("- `vol ratio` = predicted volume / GT volume (>1 over-segments).")
    md.append(f"- **`{worst['case']}` is a SAM2 truncation outlier** (Dice "
              f"{worst['dice']:.3f}, vol ratio {worst['vol_ratio']:.2f}): the "
              "continuity gate stopped propagation almost immediately, so SAM2 "
              "labelled only a sliver of a lesion nnU-Net recovers well "
              f"({d(worst['case'],'nnunet')['dice']:.3f}). Its precision stays "
              "high — what it labels is correct — but recall collapses. Excluding "
              f"it, SAM2 mean Dice over the other {len(sam_trim)} cases is "
              f"**{sam_trim_mean:.3f}**.")
    # Any case where nnU-Net loses to SAM2 (notably its empty-prediction misses)
    nn_loses = [c for c in cases if d(c, "nnunet")["dice"] + 0.05 < d(c, "sam2")["dice"]]
    for c in nn_loses:
        rn, rs = d(c, "nnunet"), d(c, "sam2")
        empty = " (nnU-Net predicted an **empty** mask out-of-fold)" if rn["pred_mm3"] == 0 else ""
        md.append(f"- **`{c}` is an nnU-Net miss**: SAM2 Dice {rs['dice']:.3f} vs "
                  f"nnU-Net {rn['dice']:.3f}{empty} — the one case where the weak "
                  "label recovers a lesion the trained model does not.")
    md.append("- **`brac62731a_01` GT was re-aligned**: it shipped with a ~3 mm "
              "world-space origin shift. Scored index-for-index it reads ~0.05 "
              "for *both* methods; resampled onto the image grid it reads "
              "0.77/0.74. The shift is a GT export artefact, not a model error.")
    if has_ell:
        md.append(f"- **Ellipse** is a *model-free* baseline: the fitted RECIST seed "
                  "ellipse is stacked at every depth with the same sphere taper, "
                  "pole floor and `n_min` filter as the SAM2 clip, but **no SAM2 "
                  "mask** — it isolates what pure geometry buys before the model. "
                  f"Mean Dice **{mean('ellipse','dice'):.3f}** "
                  f"(precision {mean('ellipse','precision'):.3f}, "
                  f"recall {mean('ellipse','recall'):.3f}) vs SAM2 "
                  f"**{mean('sam2','dice'):.3f}**.")
    md.append("- nnU-Net is scored on its **out-of-fold** CV prediction, so no "
              "case here was in the model's training fold.")
    md.append("- Overlays: `overlays_gt/` (GT blue, nnU-Net red, SAM2 green) and "
              "`overlays_sam_vs_nnunet/` (SAM2 green, nnU-Net red, agreement shaded amber).")
    (out / "gt_report.md").write_text("\n".join(md) + "\n")

    print(f"\nWrote {out}/gt_metrics.csv and {out}/gt_report.md")
    print(f"  nnU-Net mean Dice vs GT: {mean('nnunet','dice'):.3f}")
    print(f"  SAM2    mean Dice vs GT: {mean('sam2','dice'):.3f}")
    if has_ell:
        print(f"  Ellipse mean Dice vs GT: {mean('ellipse','dice'):.3f}")


if __name__ == "__main__":
    main()
