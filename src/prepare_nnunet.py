#!/usr/bin/env python3
"""
prepare_nnunet.py — assemble an nnU-Net v2 raw dataset from the pipeline output
--------------------------------------------------------------------------------
Dataset-level counterpart to prepare_sam2.py. For every case it:

  1. reads the raw MR DICOM series with the SAME reader that built the labels
     (build_volume.read_dicom_series) → guarantees the image and label share the
     exact voxel grid, which nnU-Net requires (it does not resample one onto the
     other). This is why the image is generated from DICOM here rather than
     exported from Horos / dcm2niix — an independent export can land in a
     different orientation / slice order and silently misalign with the label.
  2. writes  imagesTr/<case>_0000.nii.gz  (raw intensities, channel 0000)
     and      labelsTr/<case>.nii.gz      (binary tumour mask, {0,1}).
  3. verifies image↔label geometry and skips (with a warning) any mismatch.

Then it writes  dataset.json  and a PATIENT-GROUPED  splits_final.json  so the
same patient never appears in both train and val of a CV fold (the #1 leakage
trap — cases share a brac####/dabj### root).

Layout produced:
    <output>/Dataset<ID>_<name>/
        imagesTr/<case>_0000.nii.gz
        labelsTr/<case>.nii.gz
        dataset.json
        splits_final.json          ← copy into nnUNet_preprocessed/<Dataset>/ later

Usage
-----
    python prepare_nnunet.py \\
        --raw-root    ~/Downloads/samples \\
        --labels-dir  ~/Downloads/output/out \\
        --label-suffix _clip \\
        --cases       ~/Downloads/output/cases.txt \\
        --dataset-id 501 --dataset-name Tumour \\
        --output      nnUNet_raw

Then:
    export nnUNet_raw=.../nnUNet_raw nnUNet_preprocessed=... nnUNet_results=...
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    cp nnUNet_raw/Dataset501_Tumour/splits_final.json nnUNet_preprocessed/Dataset501_Tumour/
    nnUNetv2_train 501 3d_fullres 0   # ... folds 0-4
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from build_volume import read_dicom_series

# Geometry match tolerance (mm / direction cosine) between image and label.
GEOM_TOL = 1e-3


def _geom(img):
    return (img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())


def _geom_matches(a, b, tol=GEOM_TOL):
    (sa, spa, oa, da) = _geom(a)
    (sb, spb, ob, db) = _geom(b)
    if sa != sb:
        return False, f"size {sa} vs {sb}"
    for name, va, vb in (("spacing", spa, spb), ("origin", oa, ob), ("direction", da, db)):
        if any(abs(x - y) > tol for x, y in zip(va, vb)):
            return False, f"{name} {tuple(round(x,4) for x in va)} vs {tuple(round(x,4) for x in vb)}"
    return True, ""


def patient_root(case: str) -> str:
    """Group id = leading letters+digits (e.g. brac2991a_00 -> brac2991,
    dabj171c_01 -> dabj171). Falls back to the whole name if no match."""
    m = re.match(r"([a-zA-Z]+\d+)", case)
    return m.group(1) if m else case


def grouped_splits(cases, k=5):
    """k-fold CV split that keeps every patient's cases together in one fold.
    Greedy: assign larger patient-groups first to the currently-smallest fold so
    val-set sizes stay balanced."""
    groups = defaultdict(list)
    for c in cases:
        groups[patient_root(c)].append(c)
    roots = sorted(groups, key=lambda r: (-len(groups[r]), r))
    fold_cases = [[] for _ in range(k)]
    for r in roots:
        j = min(range(k), key=lambda i: len(fold_cases[i]))
        fold_cases[j].extend(groups[r])
    splits = []
    for i in range(k):
        val = sorted(fold_cases[i])
        train = sorted(c for j in range(k) if j != i for c in fold_cases[j])
        splits.append({"train": train, "val": val})
    return splits, groups


def resolve_cases(args):
    if args.cases:
        cases = [ln.strip() for ln in Path(args.cases).expanduser().read_text().splitlines() if ln.strip()]
    else:
        suf = args.label_suffix
        end = len(suf + ".nii.gz")
        cases = sorted(p.name[:-end] if suf else p.name[: -len(".nii.gz")]
                       for p in Path(args.labels_dir).expanduser().glob(f"*{suf}.nii.gz"))
    return cases


def main():
    ap = argparse.ArgumentParser(description="Assemble an nnU-Net v2 raw dataset from pipeline outputs.")
    ap.add_argument("--raw-root", required=True, help="Dir holding <case><raw-suffix>/ DICOM folders.")
    ap.add_argument("--raw-suffix", default="_raw", help="Suffix of raw DICOM folders (default: _raw).")
    ap.add_argument("--labels-dir", required=True, help="Dir holding the label NIfTIs.")
    ap.add_argument("--label-suffix", default="_clip",
                    help="Label filename suffix (default: _clip = propagated). Use '' for seed labels.")
    ap.add_argument("--cases", default=None, help="Manifest (one case/line). Else inferred from --labels-dir.")
    ap.add_argument("--output", default="nnUNet_raw", help="nnUNet_raw root (default: nnUNet_raw).")
    ap.add_argument("--dataset-id", type=int, required=True, help="Dataset id, e.g. 501.")
    ap.add_argument("--dataset-name", required=True, help="Dataset name, e.g. Tumour.")
    ap.add_argument("--channel-name", default="MR", help="Channel 0000 name in dataset.json (default: MR).")
    ap.add_argument("--label-name", default="tumour", help="Foreground label name (default: tumour).")
    ap.add_argument("--folds", type=int, default=5, help="CV folds for splits_final.json (default: 5).")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser()
    labels_dir = Path(args.labels_dir).expanduser()
    ds_dir = Path(args.output).expanduser() / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    imagesTr = ds_dir / "imagesTr"
    labelsTr = ds_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    cases = resolve_cases(args)
    print(f"Cases: {len(cases)}   -> {ds_dir}")

    written, skipped = [], []
    for i, case in enumerate(cases, 1):
        raw_dir = raw_root / f"{case}{args.raw_suffix}"
        label_path = labels_dir / f"{case}{args.label_suffix}.nii.gz"
        if not raw_dir.is_dir():
            print(f"  [{i}/{len(cases)}] {case}: SKIP (no raw dir {raw_dir.name})"); skipped.append(case); continue
        if not label_path.exists():
            print(f"  [{i}/{len(cases)}] {case}: SKIP (no label {label_path.name})"); skipped.append(case); continue

        try:
            img, _ = read_dicom_series(raw_dir)
        except Exception as e:
            print(f"  [{i}/{len(cases)}] {case}: SKIP (DICOM read failed: {e})"); skipped.append(case); continue
        lbl = sitk.ReadImage(str(label_path))

        ok, why = _geom_matches(img, lbl)
        if not ok:
            print(f"  [{i}/{len(cases)}] {case}: SKIP (image/label geometry mismatch: {why})")
            skipped.append(case); continue

        # binarize label {0,1}, force identical header to the image (removes any
        # float drift so nnU-Net's integrity check is exact).
        larr = (sitk.GetArrayFromImage(lbl) > 0).astype(np.uint8)
        lbl_out = sitk.GetImageFromArray(larr)
        lbl_out.CopyInformation(img)

        sitk.WriteImage(img, str(imagesTr / f"{case}_0000.nii.gz"))
        sitk.WriteImage(lbl_out, str(labelsTr / f"{case}.nii.gz"))
        fg = int(larr.sum())
        print(f"  [{i}/{len(cases)}] {case}: OK  size={img.GetSize()}  fg_voxels={fg}"
              + ("   [WARN] empty label" if fg == 0 else ""))
        written.append(case)

    if not written:
        print("\nNo cases written — nothing to assemble."); sys.exit(1)

    # dataset.json (nnU-Net v2 schema)
    dataset_json = {
        "channel_names": {"0": args.channel_name},
        "labels": {"background": 0, args.label_name: 1},
        "numTraining": len(written),
        "file_ending": ".nii.gz",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    # patient-grouped splits_final.json
    splits, groups = grouped_splits(written, k=args.folds)
    (ds_dir / "splits_final.json").write_text(json.dumps(splits, indent=4))

    print(f"\nWrote {len(written)} case(s), skipped {len(skipped)}.")
    print(f"Patients (groups): {len(groups)}   -> {args.folds}-fold grouped split")
    for i, s in enumerate(splits):
        pr = sorted({patient_root(c) for c in s["val"]})
        print(f"  fold {i}: val {len(s['val'])} case(s) / {len(pr)} patient(s)")
    print(f"\ndataset.json + splits_final.json written to {ds_dir}")
    print("Next:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  cp {ds_dir}/splits_final.json  $nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0   # folds 0-{args.folds-1}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}): {skipped[:20]}{' ...' if len(skipped)>20 else ''}")


if __name__ == "__main__":
    main()
