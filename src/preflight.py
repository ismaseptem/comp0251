#!/usr/bin/env python3
"""
preflight.py — validate Horos/DICOM exports before running the pipeline.

Purpose: catch the SILENT export mistakes the pipeline scripts don't (they only
crash on hard errors). Run first; fix any FAIL before building the volume.

Checks:
  RAW (--raw-dir)    : real MR pixel data (not a screenshot), geometry tags present,
                       consistent dimensions, no duplicate positions, lossless syntax.
  VECTOR (--rois)    : .rois_series parses, straight crossing line pairs only (no
                       ovals/polygons/lone/parallel lines), coords in grid, slot count
                       matches raw slices.   ← preferred path
  COLOUR (--roi-dir) : .dcm RGB annotations, position tags present, crosshairs
                       detectable, no unsupported shapes, lossless syntax.
  CROSS-CHECK        : annotated slices match raw by z-position / slot count.

Use:
    python preflight.py --raw-dir brac46551b_raw/ --rois brac46551b.rois_series
    python preflight.py --raw-dir brac46551b_raw/ --roi-dir brac46551b_roi/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from extracter import (detect_segments_by_color, pair_into_crosshairs,
                       COLOR_RANGES, infinite_intersection)
from typedstream_roi import parse_rois_series
from roi_series_to_volume import pair_lines

# Lossy transfer-syntax UIDs (JPEG baseline/extended, JPEG-LS lossy,
# JPEG2000 lossy, HTJ2K lossy).
LOSSY_TS = {
    "1.2.840.10008.1.2.4.50", "1.2.840.10008.1.2.4.51",
    "1.2.840.10008.1.2.4.81", "1.2.840.10008.1.2.4.91",
    "1.2.840.10008.1.2.4.101", "1.2.840.10008.1.2.4.102",
}

Z_TOL = 0.01   # must match build_volume.build_z_map rounding


# ─────────────────────────── report plumbing ─────────────────────────────

class Report:
    def __init__(self):
        self.items = []
        self.n_fail = 0
        self.n_warn = 0

    def section(self, title): self.items.append(("SEC", title))
    def ok(self, m):          self.items.append(("OK", m))
    def info(self, m):        self.items.append(("INFO", m))

    def warn(self, m):
        self.items.append(("WARN", m)); self.n_warn += 1

    def fail(self, m):
        self.items.append(("FAIL", m)); self.n_fail += 1

    def dump(self):
        icon = {"SEC": "\n■", "OK": "  ✓", "INFO": "  ·",
                "WARN": "  ⚠  WARN:", "FAIL": "  ✗  FAIL:"}
        for lvl, m in self.items:
            print(f"{icon[lvl]} {m}")
        print("\n" + "─" * 60)
        if self.n_fail:
            print(f"RESULT: {self.n_fail} FAIL, {self.n_warn} WARN — "
                  f"fix the FAILs before running the pipeline.")
        elif self.n_warn:
            print(f"RESULT: 0 FAIL, {self.n_warn} WARN — "
                  f"review warnings, pipeline should run.")
        else:
            print("RESULT: all checks passed. Safe to build the volume.")


# ─────────────────────────── helpers ─────────────────────────────────────

def series_files(d: Path):
    r = sitk.ImageSeriesReader()
    return list(r.GetGDCMSeriesFileNames(str(d)))


def z_of(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        return float(ipp[2])
    sl = getattr(ds, "SliceLocation", None)
    return float(sl) if sl is not None else None


def z_round(z):
    return round(z / Z_TOL) * Z_TOL


def transfer_syntax(ds):
    try:
        return str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        return None


# ─── annotation-shape probe: find colored marks the extracter can't model ──

# A slice with fewer than this many coloured pixels is treated as un-annotated
# (well below a legitimate crosshair's footprint).
MIN_COLOR_PX = 40
# Fraction of coloured pixels that may lie off the detected straight lines
# before we suspect a non-line annotation (freehand / text / stray line).
RESIDUAL_WARN = 0.45
# Fraction of a filled coloured region that may be "interior" (enclosed by a
# closed loop) before we call it a closed shape (oval / circle / filled ROI).
ENCLOSED_WARN = 0.35
# How wide to paint each detected line when testing what it "explains".
LINE_THICK = 7


def colored_mask(bgr):
    """Union of all extracter colour ranges (matches detect_segments_by_color)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in COLOR_RANGES.values():
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(m, k, iterations=1)


def segments_mask(shape, seglist, thick=LINE_THICK):
    m = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in seglist:
        cv2.line(m, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 255, thick)
    return m


def enclosed_ratio(cmask):
    """
    Fraction of the *filled* coloured region that is interior (enclosed by a
    closed contour). ~0 for an open crosshair ('+'/'×'); large for a closed
    loop (oval / circle / polygon / filled ROI) whose outline encloses area.
    """
    cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    filled = np.zeros_like(cmask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    fa = int(cv2.countNonZero(filled))
    ca = int(cv2.countNonZero(cmask))
    return (fa - ca) / fa if fa else 0.0


def shape_probe(bgr, crosshairs):
    """
    Return (reason or None) describing coloured annotation the crosshair model
    won't capture on this slice. Two independent signals:
      · enclosed area  → closed shapes (oval / circle / filled ROI); a
        crosshair is open so this stays ~0.
      · residual off the crosshair arms → freehand / text / stray lines that
        aren't part of a detected crosshair.
    """
    cmask = colored_mask(bgr)
    total = int(cv2.countNonZero(cmask))
    if total < MIN_COLOR_PX:
        return None

    if not crosshairs:
        return f"{total}px coloured but 0 crosshairs detected (unsupported shape?)"

    reasons = []
    enc = enclosed_ratio(cmask)
    if enc > ENCLOSED_WARN:
        reasons.append(f"{enc*100:.0f}% enclosed area — closed shape "
                       f"(oval / circle / polygon / filled ROI?)")

    arms = []
    for ch in crosshairs:
        arms.append(ch["major_seg"])
        arms.append(ch["minor_seg"])
    explained = int(cv2.countNonZero(
        cv2.bitwise_and(cmask, segments_mask(cmask.shape, arms))))
    residual = (total - explained) / total
    if residual > RESIDUAL_WARN:
        reasons.append(f"{residual*100:.0f}% of colour off the crosshair arms "
                       f"(text / freehand / stray line?)")
    return "; ".join(reasons) if reasons else None


# ─────────────────────────── raw checks ──────────────────────────────────

def check_raw(raw_dir: Path, rep: Report):
    rep.section(f"RAW MR series : {raw_dir}")
    files = series_files(raw_dir)
    if not files:
        rep.fail("no DICOM series found (check the folder / GDCM ordering).")
        return None
    rep.ok(f"{len(files)} slices found.")
    n_files = len(files)

    dims, zs, missing_ipp = set(), [], 0
    bad_photometric = rgb_like = lossy = 0
    modality = set()
    miss_spacing = miss_thick = 0
    bits = set()

    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        modality.add(str(getattr(ds, "Modality", "?")))
        photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        spp = int(getattr(ds, "SamplesPerPixel", 1))
        if spp == 3 or photo.startswith(("RGB", "YBR", "PALETTE")):
            rgb_like += 1
        if photo and not photo.startswith("MONOCHROME"):
            bad_photometric += 1
        bits.add(int(getattr(ds, "BitsStored", 0)))
        if transfer_syntax(ds) in LOSSY_TS:
            lossy += 1
        if getattr(ds, "ImagePositionPatient", None) is None:
            missing_ipp += 1
        if getattr(ds, "PixelSpacing", None) is None:
            miss_spacing += 1
        if getattr(ds, "SliceThickness", None) is None:
            miss_thick += 1
        dims.add((int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))))
        z = z_of(ds)
        if z is not None:
            zs.append(z_round(z))

    # rendered-screenshot detection
    if rgb_like:
        rep.fail(f"{rgb_like}/{len(files)} slices are RGB/colour — this looks "
                 f"like a rendered screenshot, not raw MR. prepare_sam2's "
                 f"global-window logic needs the original single-channel pixels.")
    else:
        rep.ok("single-channel (monochrome) pixel data — good.")

    if bits and max(bits) <= 8:
        rep.warn(f"BitsStored={sorted(bits)} (≤8-bit). Raw MR is usually "
                 f"12–16-bit; 8-bit suggests a rendered/windowed export.")
    if "MR" not in modality:
        rep.warn(f"Modality={sorted(modality)} (expected MR). "
                 f"SC/OT here often means a rendered export.")

    # geometry
    if missing_ipp:
        rep.fail(f"{missing_ipp} slices lack ImagePositionPatient — z-matching "
                 f"and per-lesion depth will break.")
    else:
        rep.ok("ImagePositionPatient present on all slices.")
    if miss_spacing:
        rep.warn(f"{miss_spacing} slices lack PixelSpacing.")
    if miss_thick:
        rep.warn(f"{miss_thick} slices lack SliceThickness "
                 f"(feeds per-lesion RECIST depth).")
    if len(dims) > 1:
        rep.fail(f"inconsistent slice dimensions: {sorted(dims)}.")
    else:
        rep.ok(f"uniform slice size {sorted(dims)[0] if dims else '?'}.")
    dup = len(zs) - len(set(zs))
    if dup:
        rep.warn(f"{dup} duplicate slice z-positions — build_z_map will "
                 f"collide on these.")
    if lossy:
        rep.warn(f"{lossy} slices use a lossy transfer syntax.")
    else:
        rep.ok("lossless transfer syntax.")

    return {"zs": set(zs), "n": n_files, "dims": sorted(dims)}


# ─────────────────────────── roi checks ──────────────────────────────────

def check_roi(roi_dir: Path, rep: Report, min_arm_len: int):
    rep.section(f"ROI annotated series : {roi_dir}")
    files = series_files(roi_dir)
    if not files:
        rep.fail("no DICOM series found.")
        return None
    rep.ok(f"{len(files)} slices found.")

    # .dcm extension gate (build_z_map only picks up *.dcm)
    all_dcm = [p for p in roi_dir.iterdir()
               if p.is_file() and not p.name.startswith(".")]
    ext_ok = [p for p in all_dcm if p.suffix.lower() == ".dcm"]
    if len(ext_ok) < len(files):
        rep.fail(f"only {len(ext_ok)}/{len(files)} files carry a .dcm extension "
                 f"— build_volume.build_z_map ignores the rest, so those "
                 f"annotations will be dropped. Rename them to *.dcm.")
    else:
        rep.ok("all files carry a .dcm extension.")

    annotated = []     # (z, n_tumours, colors)
    unusual = []       # (z, reason)
    missing_ipp = rgb = lossy = unreadable = 0

    for f in files:
        ds = pydicom.dcmread(f)
        if getattr(ds, "ImagePositionPatient", None) is None:
            missing_ipp += 1
        if int(getattr(ds, "SamplesPerPixel", 1)) == 3:
            rgb += 1
        if transfer_syntax(ds) in LOSSY_TS:
            lossy += 1
        try:
            arr = ds.pixel_array
        except Exception:
            unreadable += 1
            continue
        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        segs = detect_segments_by_color(bgr, min_arm_len=min_arm_len)
        chs = pair_into_crosshairs(segs)
        z = z_of(ds)
        zr = z_round(z) if z is not None else None
        if chs:
            annotated.append((zr, len(chs), sorted(segs.keys())))
        reason = shape_probe(bgr, chs)
        if reason:
            unusual.append((zr, reason))

    if unreadable:
        rep.warn(f"{unreadable} slices could not be decoded (missing pixel "
                 f"handler? try installing pylibjpeg / gdcm).")

    if rgb == 0:
        rep.fail("no RGB slices — annotations don't appear to be burned into "
                 "the pixels. Re-export with annotations flattened.")
    else:
        rep.ok(f"{rgb}/{len(files)} slices are RGB (annotations burned in).")

    if not annotated:
        rep.fail("0 crosshairs detected across the whole series. Either "
                 "annotations aren't flattened, colours are outside the "
                 "extracter set (red/green/blue/yellow/cyan/magenta), or "
                 "saturation is too low. Try lowering --min-arm-len.")
    else:
        total = sum(n for _, n, _ in annotated)
        rep.ok(f"{len(annotated)} annotated slices, {total} crosshairs total.")
        for z, n, cols in annotated:
            rep.info(f"z={z:+.3f}  tumours={n}  colours={cols}")

    if missing_ipp:
        rep.fail(f"{missing_ipp} ROI slices lack ImagePositionPatient — "
                 f"z-matching to the raw series will fail for them.")
    else:
        rep.ok("ImagePositionPatient present on all ROI slices.")

    if lossy:
        rep.warn(f"{lossy} ROI slices use a lossy transfer syntax — this can "
                 f"shift annotation hue/saturation. Prefer lossless.")

    # unsupported annotation shapes (oval/circle/text/arrow/freehand)
    if unusual:
        rep.warn(f"{len(unusual)} slice(s) carry coloured marks the crosshair "
                 f"model can't capture — these will be silently ignored:")
        for z, reason in unusual:
            zs = f"z={z:+.3f}" if z is not None else "z=?"
            rep.info(f"{zs}  {reason}")
    else:
        rep.ok("no unsupported annotation shapes detected "
               "(all colour explained by crosshair lines).")

    return annotated


# ─────────────────────────── cross-check ─────────────────────────────────

def cross_check(raw_info, annotated, rep: Report):
    rep.section("CROSS-CHECK : ROI ↔ RAW slice alignment (colour path)")
    if raw_info is None or not annotated:
        rep.warn("skipped (raw or ROI checks failed above).")
        return
    raw_zs = raw_info["zs"]
    matched = unmatched = 0
    for z, n, _ in annotated:
        if z is None:
            unmatched += 1
            continue
        if z in raw_zs:
            matched += 1
        else:
            unmatched += 1
            rep.fail(f"annotated slice z={z:+.3f} has no matching raw slice "
                     f"(within {Z_TOL} mm) — its {n} tumour(s) will be lost.")
    if unmatched == 0:
        rep.ok(f"all {matched} annotated slices map onto a raw slice.")


# ─────────────────────── vector (.rois_series) checks ─────────────────────

# A RECIST crosshair is two *straight crossing lines* (long + short axis).
# Anything else — a lone line, a closed oval/circle/rectangle (stored by Horos
# as a many-point polygon), or two axes that don't actually cross — is not a
# usable crosshair and is flagged.
#
# Slack (px) allowed when testing whether two axes cross: the intersection of
# the two infinite lines must fall within this margin of both segments' extent.
CROSS_SLACK_PX = 3.0
# Allow ROI coords this fraction beyond the acquisition grid before flagging
# (rounding / sub-pixel export slack).
BOUNDS_SLACK = 1.05


def _axes_cross(l1, l2):
    """True if two 2-point segments actually cross: the intersection of their
    infinite lines lies within (a small margin of) both segments' extents.
    Parallel axes never cross → False."""
    s1 = (l1[0][0], l1[0][1], l1[1][0], l1[1][1])
    s2 = (l2[0][0], l2[0][1], l2[1][0], l2[1][1])
    pt = infinite_intersection(s1, s2)
    if pt is None:
        return False
    x, y = pt

    def within(seg):
        x1, y1, x2, y2 = seg
        return (min(x1, x2) - CROSS_SLACK_PX <= x <= max(x1, x2) + CROSS_SLACK_PX
                and min(y1, y2) - CROSS_SLACK_PX <= y <= max(y1, y2) + CROSS_SLACK_PX)

    return within(s1) and within(s2)


def check_rois_series(rois_path: Path, raw_info, rep: Report):
    rep.section(f"ROI vector export : {rois_path}")
    try:
        slices = parse_rois_series(str(rois_path))
    except Exception as e:
        rep.fail(f"could not parse .rois_series ({type(e).__name__}: {e}). "
                 f"Re-export from Horos via 'Save ROIs' — the file must be a "
                 f"NeXTSTEP typedstream archive.")
        return None
    n_slots = len(slices)
    if n_slots == 0:
        rep.fail("parser returned 0 slice slots — file is empty or not a "
                 ".rois_series typedstream.")
        return None
    rep.ok(f"parsed {n_slots} slice slots.")

    annotated = [(i, rois) for i, rois in enumerate(slices) if rois]
    if not annotated:
        rep.fail("0 annotated slices — no ROIs found in the export.")
        return None
    n_roi = sum(len(r) for _, r in annotated)
    rep.ok(f"{len(annotated)} annotated slices, {n_roi} line ROIs "
           f"(~{n_roi // 2} tumour(s)).")

    # (1) every ROI must be a straight 2-point line. A closed shape (oval /
    # circle / rectangle) or freehand is stored by Horos as a many-point
    # polygon, so a point count other than 2 means an unwanted annotation type.
    non_line = []
    for i, rois in annotated:
        bad = [len(r) for r in rois if len(r) != 2]
        if bad:
            non_line.append((i, bad))
    if non_line:
        rep.fail(f"{len(non_line)} slice(s) contain ROIs that aren't straight "
                 f"2-point lines — a closed shape (oval / circle / rectangle) or "
                 f"freehand is stored as a many-point polygon. Only RECIST "
                 f"crosshair lines are supported; delete the others and re-export:")
        for i, bad in non_line:
            rep.info(f"slot {i}: ROI point-counts {bad} (expected 2)")
    else:
        rep.ok("all ROIs are straight 2-point lines (no ovals / polygons).")

    # (2) lines must come in crossing pairs — two axes per tumour. An odd count
    # means a lone, unpaired line (a half-drawn crosshair).
    odd = [i for i, rois in annotated if len([r for r in rois if len(r) == 2]) % 2]
    if odd:
        rep.fail(f"{len(odd)} slice(s) have an odd number of line ROIs — a lone, "
                 f"unpaired axis. Each tumour needs two crossing lines "
                 f"(long + short axis): slots {odd}.")
    else:
        rep.ok("every annotated slice has an even line count (paired axes).")

    # (3) each pair must actually cross. Pairing is by nearest midpoint; a pair
    # whose axes don't intersect is either a mispair (two different tumours) or
    # two parallel lines — not a crosshair.
    no_cross = []
    for i, rois in annotated:
        lines = [r for r in rois if len(r) == 2]
        if len(lines) < 2:
            continue
        pairs, _ = pair_lines(lines)
        bad = sum(1 for a, b, _ in pairs if not _axes_cross(lines[a], lines[b]))
        if bad:
            no_cross.append((i, bad, len(pairs)))
    if no_cross:
        rep.fail(f"{len(no_cross)} slice(s) have paired axes that don't cross — "
                 f"a mispair or parallel (non-crossing) lines, not a crosshair:")
        for i, bad, tot in no_cross:
            rep.info(f"slot {i}: {bad}/{tot} pair(s) don't intersect")
    elif not odd:
        rep.ok("all paired axes cross (well-formed crosshairs).")

    # coords must lie in the acquisition grid. OsiriX stores them in the square
    # M² grid (M = max raw Rows/Cols); anything well beyond that is a wrong-grid
    # or wrong-series export.
    if raw_info is not None and raw_info["dims"]:
        rows, cols = raw_info["dims"][0]
        M = max(rows, cols)
        max_x = max((x for _, rois in annotated for r in rois for x, _ in r),
                    default=0.0)
        max_y = max((y for _, rois in annotated for r in rois for _, y in r),
                    default=0.0)
        if max_x > M * BOUNDS_SLACK or max_y > M * BOUNDS_SLACK:
            rep.warn(f"ROI coords exceed the acquisition grid "
                     f"(max x={max_x:.0f}, y={max_y:.0f} vs M={M}) — the export "
                     f"may not match this raw series.")
        else:
            rep.ok(f"ROI coords lie within the {M}² acquisition grid.")

    return {"n_slots": n_slots, "annotated": annotated}


def cross_check_vector(raw_info, roi_info, rep: Report):
    rep.section("CROSS-CHECK : slot ↔ RAW slice alignment (vector path)")
    if raw_info is None or roi_info is None:
        rep.warn("skipped (raw or ROI checks failed above).")
        return
    n_raw, n_slots = raw_info["n"], roi_info["n_slots"]
    max_ann = max(i for i, _ in roi_info["annotated"])
    if n_slots == n_raw:
        rep.ok(f"slot count matches raw slice count ({n_raw}); index i → raw "
               f"slice i.")
    else:
        rep.warn(f"slot count {n_slots} != raw slice count {n_raw} — "
                 f"roi_series_to_volume places by index and ignores overflow; "
                 f"confirm the ROI array wasn't exported against a different "
                 f"series (use --flip-z if orientation is reversed).")
    if max_ann >= n_raw:
        rep.fail(f"an annotated slot (index {max_ann}) is beyond the last raw "
                 f"slice (index {n_raw - 1}) — those tumour(s) will be dropped.")


# ─────────────────────────── main ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate Horos/DICOM exports before running the pipeline.")
    ap.add_argument("--raw-dir", required=True,
                    help="Folder of raw MR DICOMs (video frames / geometry).")
    ap.add_argument("--rois", default=None,
                    help="Horos .rois_series vector export (preferred path; "
                         "feeds roi_series_to_volume.py).")
    ap.add_argument("--roi-dir", default=None,
                    help="Folder of annotated Secondary-Capture DICOMs "
                         "(legacy colour path; feeds build_volume.py).")
    ap.add_argument("--min-arm-len", type=int, default=15,
                    help="Shortest crosshair arm to accept during the colour "
                         "ROI detection probe (default 15; match build_volume).")
    args = ap.parse_args()

    if not args.rois and not args.roi_dir:
        ap.error("provide --rois (vector path) and/or --roi-dir (colour path).")

    rep = Report()
    raw_info = check_raw(Path(args.raw_dir), rep)

    if args.rois:
        roi_info = check_rois_series(Path(args.rois), raw_info, rep)
        cross_check_vector(raw_info, roi_info, rep)
    if args.roi_dir:
        annotated = check_roi(Path(args.roi_dir), rep, args.min_arm_len)
        cross_check(raw_info, annotated, rep)

    rep.dump()
    sys.exit(1 if rep.n_fail else 0)


if __name__ == "__main__":
    main()
