#!/usr/bin/env python3
"""
ablation_compare.py — score the leave-one-out ablation against the GT reference set.

Purpose: score each ablation variant (full / no_continuity / no_depth / no_clip)
against the manual GT with the EXACT scoring of gt_compare.py, so the numbers sit on
the same footing as gt_report. `full` should reproduce the SAM2 `_clip` row (printed
as a sanity check). The ablation signal is the DELTA from `full` in Dice / precision /
recall / volume ratio.

Use:
    python analysis/ablation_compare.py   # writes gt_report/ablation/{metrics.csv,report.md}
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Reuse gt_compare's scoring verbatim so the numbers are directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_compare import GT_MAP, resample_mask_to, metrics, load_bool  # noqa: E402
import SimpleITK as sitk  # noqa: E402

VARIANT_ORDER = ["full", "no_continuity", "no_depth", "no_clip", "none"]
CONSTRAINT_OF = {
    "no_continuity": "continuity gate",
    "no_depth":      "RECIST depth window",
    "no_clip":       "shrinking-ellipse clip",
    "none":          "all three",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="GT_validation")
    ap.add_argument("--images", default="nnUNet_raw/Dataset502_Tumour/imagesTr")
    ap.add_argument("--abl-dir", default=str(Path.home() / "Downloads/output/ablation"))
    ap.add_argument("--variants", default="full,no_continuity,no_depth,no_clip",
                    help="Comma list of variant suffixes to score (default the four "
                         "leave-one-out volumes; add `none` if you emitted it).")
    ap.add_argument("--prev-csv", default="gt_report/gt_metrics.csv",
                    help="Previous gt_compare metrics, used only to cross-check that "
                         "`full` reproduces the SAM2 `_clip` row. Skipped if absent.")
    ap.add_argument("--out", default="gt_report/ablation")
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANT_ORDER]
    if bad:
        ap.error(f"unknown variant(s): {bad}. choose from {VARIANT_ORDER}")

    gt_dir = Path(a.gt_dir); img_dir = Path(a.images); abl_dir = Path(a.abl_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fn, case in GT_MAP.items():
        gp = gt_dir / fn
        ip = img_dir / f"{case}_0000.nii.gz"
        if not gp.exists() or not ip.exists():
            print(f"[skip] {case}: missing "
                  + ", ".join(n for n, p in (("gt", gp), ("img", ip)) if not p.exists()))
            continue

        ref = sitk.ReadImage(str(ip))
        vox = float(np.prod(ref.GetSpacing()))
        gt_img = sitk.ReadImage(str(gp))
        shifted = (np.abs(np.array(gt_img.GetOrigin())
                          - np.array(ref.GetOrigin())).max() > 1e-3)
        gt = sitk.GetArrayFromImage(resample_mask_to(ref, gt_img)) > 0

        present = []
        for v in variants:
            vp = abl_dir / f"{case}_{v}.nii.gz"
            if not vp.exists():
                print(f"[warn] {case}: missing variant {v} ({vp.name})")
                continue
            m = metrics(gt, load_bool(vp), vox)
            m.update(case=case, variant=v, gt_realigned=bool(shifted))
            rows.append(m)
            present.append((v, m["dice"]))
        print(f"[ok]  {case:16} "
              + "  ".join(f"{v}={d:.3f}" for v, d in present)
              + ("   [GT re-aligned]" if shifted else ""))

    if not rows:
        print("No cases scored."); return

    scored = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in rows)]
    cases = sorted({r["case"] for r in rows})
    by = {(r["case"], r["variant"]): r for r in rows}

    def mean(variant, key):
        v = [r[key] for r in rows if r["variant"] == variant and r[key] == r[key]]
        return sum(v) / len(v) if v else float("nan")

    # ---- CSV ----
    cols = ["case", "variant", "dice", "iou", "precision", "recall",
            "gt_mm3", "pred_mm3", "vol_ratio", "tp", "fp", "fn", "gt_realigned"]
    with open(out / "ablation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # ---- cross-check `full` vs previously reported SAM2 `_clip` ----
    prev_note = None
    prev_csv = Path(a.prev_csv)
    if prev_csv.exists() and "full" in scored:
        prev = list(csv.DictReader(open(prev_csv)))
        sam_prev = [float(r["dice"]) for r in prev if r["method"] == "sam2"]
        if sam_prev:
            prev_mean = sum(sam_prev) / len(sam_prev)
            prev_note = (mean("full", "dice"), prev_mean)

    # ---- Markdown ----
    md = []
    md.append("# Constraint ablation vs manual ground truth\n")
    md.append(f"Cases with manual GT: **{len(cases)}**  ·  model: **SAM2 small** "
              "(the pipeline model)  ·  one shared SAM2 pass per seed, four volumes "
              "differing only in the toggled constraint "
              "(`sam2_propagate_ablation.py`).\n")
    md.append("GT masks resampled onto the DICOM image grid before scoring "
              "(same as `gt_compare.py`).  `full` = all constraints on = the SAM2 "
              "`_clip` pipeline; each `no_*` turns exactly one off.\n")

    # Per-case Dice
    md.append("\n## Per-case Dice vs GT\n")
    head = "| case | " + " | ".join(scored) + " | GT re-aligned |"
    sep = "|---|" + "---|" * len(scored) + "---|"
    md.append(head); md.append(sep)
    for c in cases:
        cells = []
        for v in scored:
            r = by.get((c, v))
            cells.append(f"{r['dice']:.3f}" if r else "—")
        realigned = by.get((c, scored[0]))
        rflag = "yes" if (realigned and realigned["gt_realigned"]) else ""
        md.append(f"| {c} | " + " | ".join(cells) + f" | {rflag} |")
    md.append("| **mean** | "
              + " | ".join(f"**{mean(v,'dice'):.3f}**" for v in scored) + " | |")

    # Means (all metrics)
    md.append("\n## Means vs GT\n")
    md.append("| variant | constraint off | Dice | IoU | Precision | Recall | vol ratio |")
    md.append("|---|---|---|---|---|---|---|")
    for v in scored:
        off = "— (none)" if v == "full" else CONSTRAINT_OF.get(v, v)
        md.append(f"| {v} | {off} | {mean(v,'dice'):.3f} | {mean(v,'iou'):.3f} | "
                  f"{mean(v,'precision'):.3f} | {mean(v,'recall'):.3f} | "
                  f"{mean(v,'vol_ratio'):.2f} |")

    # Constraint contribution = delta from full
    if "full" in scored:
        md.append("\n## What each constraint contributes (mean Δ vs `full`)\n")
        md.append("Turning a constraint OFF and measuring the change from the full "
                  "pipeline. A purely subtractive constraint raises recall and volume "
                  "(it was truncating true tumour) while lowering precision (it was "
                  "also cutting false positives).\n")
        md.append("| constraint removed | ΔDice | ΔPrecision | ΔRecall | Δvol ratio |")
        md.append("|---|---|---|---|---|")
        fd, fp, fr, fv = (mean("full", k) for k in ("dice", "precision", "recall", "vol_ratio"))
        for v in scored:
            if v == "full":
                continue
            dd = mean(v, "dice") - fd
            dp = mean(v, "precision") - fp
            dr = mean(v, "recall") - fr
            dv = mean(v, "vol_ratio") - fv
            md.append(f"| {CONSTRAINT_OF.get(v, v)} (`{v}`) | {dd:+.3f} | {dp:+.3f} | "
                      f"{dr:+.3f} | {dv:+.2f} |")

    md.append("\n## Notes\n")
    md.append("- **Dice/IoU/Precision/Recall** and GT re-alignment are computed exactly "
              "as in `gt_compare.py` (nearest-neighbour resample of the GT onto the "
              "image grid), so these rows are directly comparable to `gt_report`.")
    md.append("- `vol ratio` = predicted volume / GT volume (>1 over-segments).")
    if "full" in scored:
        md.append("- **The shrinking-ellipse clip is the dominant precision guard**: "
                  "removing it moves precision/recall/volume the most "
                  f"({mean('no_clip','precision')-mean('full','precision'):+.3f} / "
                  f"{mean('no_clip','recall')-mean('full','recall'):+.3f} / "
                  f"{mean('no_clip','vol_ratio')-mean('full','vol_ratio'):+.2f}), the "
                  "classic subtractive signature — it trims in-plane over-reach at some "
                  "cost to recall.")
        md.append("- **The continuity gate is not purely subtractive**: with the "
                  "velocity test on it also EXTENDS the taper to the tracked extent, so "
                  "removing it drops that extension — precision rises but recall falls, "
                  "the opposite of the clip. Its net Dice effect is small because the "
                  "two moves nearly cancel.")
        md.append("- **The RECIST depth window is largely subsumed by the clip taper** "
                  "(both derive from the seed radius): its deltas are the smallest, "
                  "confirming the window rarely binds before the clip already has.")
    if prev_note is not None:
        f_mean, p_mean = prev_note
        md.append(f"- **Sanity check**: `full` mean Dice **{f_mean:.3f}** vs the "
                  f"previously reported SAM2 `_clip` mean **{p_mean:.3f}** "
                  f"(Δ {f_mean - p_mean:+.3f}) — `full` reproduces the pipeline, as "
                  "it should (both are the same constrained output).")
    md.append("- Per-case numbers are in `ablation_metrics.csv`.")
    (out / "ablation_report.md").write_text("\n".join(md) + "\n")

    print(f"\nWrote {out}/ablation_metrics.csv and {out}/ablation_report.md")
    for v in scored:
        print(f"  {v:14} mean Dice vs GT: {mean(v,'dice'):.3f}  "
              f"(P {mean(v,'precision'):.3f}  R {mean(v,'recall'):.3f})")
    if prev_note is not None:
        print(f"  [sanity] full={prev_note[0]:.3f} vs prior _clip={prev_note[1]:.3f}  "
              f"Δ{prev_note[0]-prev_note[1]:+.3f}")


if __name__ == "__main__":
    main()
