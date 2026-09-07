#!/usr/bin/env python3
"""Lesion-level detection, split by whether the lesion carries a RECIST mark.

Purpose: for each reference study, take the connected components of the dense manual
reference and split them into SEEDED (overlaps a RECIST mark) and UNSEEDED (pure
generalisation target), then report the per-lesion detection rate for the constrained
pseudo-label and the nnU-Net out-of-fold prediction. A lesion counts as detected when a
method recovers at least TAU of its own voxels. The UNSEEDED gap = how far the network
generalises past its supervision.

Use:
    python lesion_detection.py                 # tau=0.10, min-vox=10
    python lesion_detection.py --tau 0.5 --min-vox 20
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

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
    ap.add_argument("--preds",
                    default="nnUNet_results/Dataset502_Tumour/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4")
    ap.add_argument("--sam-dir", default=str(Path.home() / "Downloads/output/out"))
    ap.add_argument("--seed-root", default=str(Path.home() / "Downloads/output/sam2_input"))
    ap.add_argument("--out", default="gt_report")
    ap.add_argument("--tau", type=float, default=0.10,
                    help="detection threshold: fraction of a lesion's voxels the "
                         "method must recover (default 0.10)")
    ap.add_argument("--min-vox", type=int, default=10,
                    help="ignore GT components smaller than this (default 10)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated case ids to drop (e.g. "
                         "brac51681a_01,brac62721b_00,dabj201c_01). Output files "
                         "get a _<n>studies suffix so the full table is kept.")
    a = ap.parse_args()

    img_dir = Path(a.images); pred_dir = Path(a.preds)
    sam = Path(a.sam_dir); seed_root = Path(a.seed_root); gt_dir = Path(a.gt_dir)
    excl = {c.strip() for c in a.exclude.split(",") if c.strip()}

    rows = []          # per-lesion records
    n_studies = 0
    for fn, case in GT_MAP.items():
        if case in excl:
            print(f"[excl] {case}"); continue
        ip = img_dir / f"{case}_0000.nii.gz"
        gp = gt_dir / fn
        pp = pred_dir / f"{case}.nii.gz"
        mp = seed_root / case / "masks"
        # prefer the single-run dual constrained output, else _clip
        dual_c = sam / f"{case}_dual.nii.gz"
        cp = dual_c if dual_c.exists() else sam / f"{case}_clip.nii.gz"
        if not (ip.exists() and gp.exists() and pp.exists() and cp.exists() and mp.is_dir()):
            print(f"[skip] {case}: missing input"); continue
        ref = sitk.ReadImage(str(ip))
        gt = sitk.GetArrayFromImage(resample_to(ref, sitk.ReadImage(str(gp)))) > 0
        nn = sitk.GetArrayFromImage(sitk.ReadImage(str(pp))) > 0
        ps = sitk.GetArrayFromImage(sitk.ReadImage(str(cp))) > 0
        seed = seed_volume(mp, gt.shape)
        n_studies += 1

        lab, nles = ndimage.label(gt)
        for i in range(1, nles + 1):
            L = lab == i
            vox = int(L.sum())
            if vox < a.min_vox:
                continue
            seeded = bool((seed & L).any())
            rec_ps = (ps & L).sum() / vox
            rec_nn = (nn & L).sum() / vox
            rows.append(dict(case=case, lesion=i, vox=vox,
                             group="seeded" if seeded else "unseeded",
                             recall_pseudo=rec_ps, recall_network=rec_nn,
                             det_pseudo=int(rec_ps >= a.tau),
                             det_network=int(rec_nn >= a.tau)))
        print(f"[ok]  {case:16} lesions={sum(1 for r in rows if r['case']==case)}")

    if not rows:
        print("No lesions."); return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{n_studies}studies" if excl else ""
    with open(out / f"lesion_detection{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "lesion", "vox", "group",
                                          "recall_pseudo", "recall_network",
                                          "det_pseudo", "det_network"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "recall_pseudo": f"{r['recall_pseudo']:.4f}",
                        "recall_network": f"{r['recall_network']:.4f}"})

    # ---- 2x2 table ----
    def cell(group, key):
        g = [r for r in rows if r["group"] == group]
        n = len(g)
        d = sum(r[key] for r in g)
        return d, n, (100.0 * d / n if n else float("nan"))

    groups = ["seeded", "unseeded"]
    md = []
    md.append("# Lesion-level detection vs the dense reference\n")
    if excl:
        md.append(f"_Excluding: {', '.join(sorted(excl))}._\n")
    md.append(f"Reference studies: **{n_studies}**  ·  GT connected components "
              f"≥ {a.min_vox} voxels  ·  a lesion is *detected* when the method "
              f"recovers ≥ {a.tau:.0%} of its voxels.\n")
    md.append("nnU-Net = Dataset502 out-of-fold (5-fold CV); pseudo-label = "
              "constrained SAM2 (`_clip`/`_dual`).\n")
    md.append("| group | lesions | pseudo-label detected | network detected |")
    md.append("|---|---|---|---|")
    tot = {}
    for grp in groups:
        dp, n, rp = cell(grp, "det_pseudo")
        dn, _, rn = cell(grp, "det_network")
        tot[grp] = n
        label = "seeded (carries a RECIST mark)" if grp == "seeded" else "unseeded (annotation missed it)"
        md.append(f"| {label} | {n} | {dp}/{n} = **{rp:.0f}%** | {dn}/{n} = **{rn:.0f}%** |")
    # overall row
    dpa = sum(r["det_pseudo"] for r in rows); dna = sum(r["det_network"] for r in rows)
    N = len(rows)
    md.append(f"| all | {N} | {dpa}/{N} = {100*dpa/N:.0f}% | {dna}/{N} = {100*dna/N:.0f}% |")

    md.append("\n## Mean per-lesion recall (secondary)\n")
    md.append("| group | pseudo-label | network |")
    md.append("|---|---|---|")
    for grp in groups:
        g = [r for r in rows if r["group"] == grp]
        mp_ = np.mean([r["recall_pseudo"] for r in g])
        mn_ = np.mean([r["recall_network"] for r in g])
        md.append(f"| {grp} | {mp_:.3f} | {mn_:.3f} |")

    md.append("\n## Reading it\n")
    dn_u = cell("unseeded", "det_network")[2]
    dp_u = cell("unseeded", "det_pseudo")[2]
    md.append(f"- The **unseeded** column is the thesis claim as a number: the "
              f"pseudo-labels detect {dp_u:.0f}% of the lesions the RECIST "
              f"annotation missed, the network detects **{dn_u:.0f}%**. The "
              f"network recovers tumours its own supervision never marked.")
    md.append(f"- The **seeded** column is a sanity check: both methods should "
              "find nearly every lesion that carries a mark.")
    md.append(f"- Detection threshold τ = {a.tau:.2f} of lesion voxels; components "
              f"< {a.min_vox} voxels ignored. Recall is against each lesion's own "
              "voxels, so a neighbour's mask is never credited.")
    (out / f"lesion_detection{suffix}.md").write_text("\n".join(md) + "\n")

    # ---- console ----
    print(f"\n=== Lesion detection ({n_studies} studies, {N} lesions, "
          f"tau={a.tau}, min_vox={a.min_vox}) ===")
    print(f"{'group':10} {'n':>4}  {'pseudo':>10}  {'network':>10}")
    for grp in groups:
        dp, n, rp = cell(grp, "det_pseudo")
        dn, _, rn = cell(grp, "det_network")
        print(f"{grp:10} {n:>4}  {rp:>9.0f}%  {rn:>9.0f}%")
    print(f"\nWrote {out}/lesion_detection{suffix}.md and .csv")


if __name__ == "__main__":
    main()
