#!/usr/bin/env python3
"""
tumour_stats.py — measure tumour counts & volumes, separately or compared
-------------------------------------------------------------------------
Three subcommands sharing the same measurement functions:

  recist       count + volume of RECIST tumours from a .rois_series (ellipsoid)
  propagated   count + volume of tumours from a propagated label (measured voxels)
  compare      both, side by side, with the propagated/RECIST volume ratio

RECIST gives only two in-plane diameters (long L, short S), so its 3-D volume is
the standard "modified ellipsoid" V = (π/6)·L·S² (semi-axes L/2, S/2, S/2).
Change the through-plane assumption with --third-axis {short,mean,sphere}.
The propagated volume is *measured*: 3-D connected components, voxels × spacing.

Geometry for the RECIST px→mm conversion + 256²-grid correction comes from a
DICOM series (--raw-dir) or any label in the same space (--geometry); in
`compare`, the propagated label supplies it.

Examples
--------
  # RECIST only (single / dataset)
  python tumour_stats.py recist --rois case_roi.rois_series --raw-dir case_raw/
  python tumour_stats.py recist --rois-dir samples --raw-root samples --out recist.csv

  # Propagated only (single / dataset)
  python tumour_stats.py propagated --label out/case_clip.nii.gz --details
  python tumour_stats.py propagated --labels-dir out --out prop.csv

  # Compare
  python tumour_stats.py compare --rois case_roi.rois_series --label out/case_clip.nii.gz
  python tumour_stats.py compare --rois-dir samples --labels-dir out --out compare.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from typedstream_roi import parse_rois_series
from roi_series_to_volume import pair_lines, _square_correct
from build_volume import read_dicom_series

MM3_PER_CM3 = 1000.0
_CONN3D = np.ones((3, 3, 3), dtype=int)   # 26-connectivity for 3-D components


# ─────────────────────────── measurement core ────────────────────────────

def _axis_len_mm(line, sx, sy):
    (x1, y1), (x2, y2) = line[0], line[1]
    return math.hypot((x2 - x1) * sx, (y2 - y1) * sy)


def _ellipsoid_volume(L, S, mode):
    """mm³ from RECIST long L / short S diameters (mm)."""
    if mode == "short":
        third = S
    elif mode == "mean":
        third = 0.5 * (L + S)
    elif mode == "sphere":
        S = third = L
    else:
        raise ValueError(mode)
    return math.pi / 6.0 * L * S * third


def measure_recist(rois_path, ref_img, third_axis="short"):
    """[{slot, cx, cy, L_mm, S_mm, vol_mm3}, ...] from crosshairs."""
    W, H, _ = ref_img.GetSize()
    sx, sy, _ = ref_img.GetSpacing()
    slices, _ = _square_correct(parse_rois_series(str(rois_path)), H, W)
    out = []
    for slot, rois in enumerate(slices):
        lines = [r for r in rois if len(r) == 2]
        if len(lines) < 2:
            continue
        pairs, _ = pair_lines(lines)
        for i, j, _d in pairs:
            l1, l2 = lines[i], lines[j]
            a, b = _axis_len_mm(l1, sx, sy), _axis_len_mm(l2, sx, sy)
            L, S = (a, b) if a >= b else (b, a)
            cx = (l1[0][0] + l1[1][0] + l2[0][0] + l2[1][0]) / 4.0
            cy = (l1[0][1] + l1[1][1] + l2[0][1] + l2[1][1]) / 4.0
            out.append({"slot": slot, "cx": cx, "cy": cy, "L_mm": L, "S_mm": S,
                        "vol_mm3": _ellipsoid_volume(L, S, third_axis)})
    return out


def measure_propagated(label_img, min_vox=1):
    """(labelmap, [{label, voxels, vol_mm3, z, cy, cx}, ...], voxel_mm3)."""
    arr = (sitk.GetArrayFromImage(label_img) > 0).astype(np.uint8)   # (Z,H,W)
    sx, sy, sz = label_img.GetSpacing()
    voxel_mm3 = sx * sy * sz
    lab, n = ndimage.label(arr, structure=_CONN3D)
    comps = []
    for k in range(1, n + 1):
        m = lab == k
        vox = int(m.sum())
        if vox < min_vox:
            continue
        zc, yc, xc = (c.mean() for c in np.where(m))
        comps.append({"label": k, "voxels": vox, "vol_mm3": vox * voxel_mm3,
                      "z": zc, "cy": yc, "cx": xc})
    return lab, comps, voxel_mm3


def geometry_from(args, raw_dir=None, geometry=None):
    """A SimpleITK image (for size+spacing) from a DICOM dir or a label file."""
    if geometry:
        return sitk.ReadImage(str(geometry))
    if raw_dir:
        return read_dicom_series(Path(raw_dir))[0]
    raise ValueError("need --raw-dir or --geometry for RECIST geometry")


# ─────────────────────────── reporting ───────────────────────────────────

def _tot(items):
    return sum(t["vol_mm3"] for t in items)


def report_recist(name, tumours, details):
    v = _tot(tumours) / MM3_PER_CM3
    print(f"\n=== {name}  [RECIST] ===")
    print(f"  tumours: {len(tumours):3d}    ellipsoid volume: {v:8.2f} cm³")
    if details:
        for k, t in enumerate(tumours):
            print(f"    #{k:2d} slot {t['slot']:2d}  L={t['L_mm']:5.1f}mm "
                  f"S={t['S_mm']:5.1f}mm  V={t['vol_mm3']/MM3_PER_CM3:6.3f} cm³")


def report_propagated(name, comps, details):
    v = _tot(comps) / MM3_PER_CM3
    print(f"\n=== {name}  [PROPAGATED] ===")
    print(f"  tumours: {len(comps):3d}    measured volume : {v:8.2f} cm³")
    if details:
        for c in sorted(comps, key=lambda c: -c["vol_mm3"]):
            print(f"    comp {c['label']:3d}  {c['voxels']:6d} vox  "
                  f"V={c['vol_mm3']/MM3_PER_CM3:6.3f} cm³  "
                  f"@ (z={c['z']:.0f}, y={c['cy']:.0f}, x={c['cx']:.0f})")


def compare_case(rois_path, label_path, third_axis, min_vox):
    ref = sitk.ReadImage(str(label_path))
    recist = measure_recist(rois_path, ref, third_axis)
    lab, comps, _ = measure_propagated(ref, min_vox)
    Z, H, W = lab.shape
    for t in recist:
        z = min(max(t["slot"], 0), Z - 1)
        y = min(max(int(round(t["cy"])), 0), H - 1)
        x = min(max(int(round(t["cx"])), 0), W - 1)
        t["hit"] = int(lab[z, y, x])
    hit_labels = {t["hit"] for t in recist if t["hit"]}
    vr, vp = _tot(recist), _tot(comps)
    return {
        "n_recist": len(recist), "n_propagated": len(comps),
        "recist_vol_mm3": vr, "propagated_vol_mm3": vp,
        "volume_ratio": (vp / vr) if vr else float("nan"),
        "n_missed": sum(1 for t in recist if not t["hit"]),
        "n_extra": sum(1 for c in comps if c["label"] not in hit_labels),
    }


def report_compare(name, r):
    print(f"\n=== {name}  [COMPARE] ===")
    print(f"  RECIST     : {r['n_recist']:3d} tumours   "
          f"{r['recist_vol_mm3']/MM3_PER_CM3:8.2f} cm³ (ellipsoid)")
    print(f"  Propagated : {r['n_propagated']:3d} tumours   "
          f"{r['propagated_vol_mm3']/MM3_PER_CM3:8.2f} cm³ (measured)")
    print(f"  volume ratio prop/RECIST: {r['volume_ratio']:.2f}   "
          f"missed={r['n_missed']}  extra={r['n_extra']}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {path}")


# ─────────────────────────── case resolution ─────────────────────────────

def _cases_recist(a):
    if a.rois:
        yield (Path(a.rois).stem, Path(a.rois),
               geometry_from(a, a.raw_dir, a.geometry))
        return
    rd = Path(a.rois_dir)
    for rp in sorted(rd.glob(f"*{a.roi_suffix}.rois_series")):
        case = rp.name[: -len(f'{a.roi_suffix}.rois_series')]
        raw = Path(a.raw_root) / f"{case}_raw" if a.raw_root else None
        if raw and not raw.is_dir():
            print(f"[skip] {case}: no {raw.name}"); continue
        try:
            yield case, rp, geometry_from(a, raw, None)
        except Exception as e:
            print(f"[skip] {case}: {e}")


def _cases_propagated(a):
    if a.label:
        yield (Path(a.label).name.split('.')[0], Path(a.label)); return
    for lp in sorted(Path(a.labels_dir).glob(f"*{a.label_suffix}.nii.gz")):
        yield lp.name[: -len('.nii.gz')], lp


def _cases_compare(a):
    if a.rois and a.label:
        yield (Path(a.rois).stem, Path(a.rois), Path(a.label)); return
    rd, ld = Path(a.rois_dir), Path(a.labels_dir)
    for rp in sorted(rd.glob(f"*{a.roi_suffix}.rois_series")):
        case = rp.name[: -len(f'{a.roi_suffix}.rois_series')]
        lp = ld / f"{case}{a.label_suffix}.nii.gz"
        if lp.exists():
            yield case, rp, lp


# ─────────────────────────── CLI ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Measure tumour counts & volumes (RECIST / propagated / compare).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("recist", help="Measure RECIST tumours from a .rois_series.")
    pr.add_argument("--rois"); pr.add_argument("--raw-dir"); pr.add_argument("--geometry")
    pr.add_argument("--rois-dir"); pr.add_argument("--raw-root")
    pr.add_argument("--roi-suffix", default="_roi")
    pr.add_argument("--third-axis", default="short", choices=["short", "mean", "sphere"])
    pr.add_argument("--details", action="store_true"); pr.add_argument("--out")

    pp = sub.add_parser("propagated", help="Measure tumours from a propagated label.")
    pp.add_argument("--label"); pp.add_argument("--labels-dir")
    pp.add_argument("--label-suffix", default="_clip")
    pp.add_argument("--min-vox", type=int, default=1)
    pp.add_argument("--details", action="store_true"); pp.add_argument("--out")

    pc = sub.add_parser("compare", help="Compare propagated volumes to RECIST ellipsoids.")
    pc.add_argument("--rois"); pc.add_argument("--label")
    pc.add_argument("--rois-dir"); pc.add_argument("--labels-dir")
    pc.add_argument("--roi-suffix", default="_roi"); pc.add_argument("--label-suffix", default="_clip")
    pc.add_argument("--third-axis", default="short", choices=["short", "mean", "sphere"])
    pc.add_argument("--min-vox", type=int, default=1); pc.add_argument("--out")

    a = ap.parse_args()
    rows = []

    if a.cmd == "recist":
        if not (a.rois or a.rois_dir):
            ap.error("give --rois (+ --raw-dir/--geometry) or --rois-dir (+ --raw-root)")
        for name, rp, ref in _cases_recist(a):
            t = measure_recist(rp, ref, a.third_axis)
            report_recist(name, t, a.details)
            rows.append({"case": name, "n_recist": len(t),
                         "recist_vol_cm3": round(_tot(t) / MM3_PER_CM3, 3)})

    elif a.cmd == "propagated":
        if not (a.label or a.labels_dir):
            ap.error("give --label or --labels-dir")
        for name, lp in _cases_propagated(a):
            _, comps, _ = measure_propagated(sitk.ReadImage(str(lp)), a.min_vox)
            report_propagated(name, comps, a.details)
            rows.append({"case": name, "n_propagated": len(comps),
                         "propagated_vol_cm3": round(_tot(comps) / MM3_PER_CM3, 3)})

    elif a.cmd == "compare":
        if not ((a.rois and a.label) or (a.rois_dir and a.labels_dir)):
            ap.error("give --rois + --label, or --rois-dir + --labels-dir")
        for name, rp, lp in _cases_compare(a):
            try:
                r = compare_case(rp, lp, a.third_axis, a.min_vox)
            except Exception as e:
                print(f"\n=== {name} ===\n  ERROR: {type(e).__name__}: {e}"); continue
            report_compare(name, r)
            rows.append({"case": name, "n_recist": r["n_recist"],
                         "n_propagated": r["n_propagated"],
                         "recist_vol_cm3": round(r["recist_vol_mm3"] / MM3_PER_CM3, 3),
                         "propagated_vol_cm3": round(r["propagated_vol_mm3"] / MM3_PER_CM3, 3),
                         "volume_ratio": round(r["volume_ratio"], 3),
                         "n_missed": r["n_missed"], "n_extra": r["n_extra"]})

    # dataset totals
    if len(rows) > 1:
        print(f"\n=== DATASET ({len(rows)} cases) ===")
        if "n_recist" in rows[0]:
            print(f"  RECIST tumours total    : {sum(x['n_recist'] for x in rows)}"
                  f"   {sum(x['recist_vol_cm3'] for x in rows):.1f} cm³")
        if "n_propagated" in rows[0]:
            print(f"  Propagated tumours total: {sum(x['n_propagated'] for x in rows)}"
                  f"   {sum(x['propagated_vol_cm3'] for x in rows):.1f} cm³")
        if a.cmd == "compare":
            vr = sum(x["recist_vol_cm3"] for x in rows)
            vp = sum(x["propagated_vol_cm3"] for x in rows)
            print(f"  overall volume ratio    : {vp/vr:.2f}" if vr else "  ratio: n/a")

    if getattr(a, "out", None):
        write_csv(a.out, rows)


if __name__ == "__main__":
    main()
