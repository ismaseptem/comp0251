#!/usr/bin/env python3
"""
MRI Tumor Segmentation from Crosshair Annotations
--------------------------------------------------
Detects colored crosshair annotations (two intersecting lines per tumor),
models each tumor as an ellipse (longer arm = major axis, shorter = minor axis),
and outputs an overlay image + binary segmentation mask.

Usage:
    python extracter.py image.jpg
    python extracter.py image.jpg --output results/ --no-show
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ─────────────────────────── HSV color ranges ────────────────────────────
# Each entry: list of (lower, upper) HSV bounds.
# Red wraps around 0/180 so it needs two ranges.
# Blue saturation floor is high (120) to avoid MRI greyscale/JPEG artefacts.
COLOR_RANGES = {
    "red":     [((0,   80, 80), (12,  255, 255)),
                ((160, 80, 80), (180, 255, 255))],
    "green":   [((36,  60, 60), (85,  255, 255))],
    "blue":    [((100, 120, 100), (130, 255, 255))],
    "yellow":  [((18,  80, 80), (36,  255, 255))],
    "cyan":    [((82,  60, 60), (100, 255, 255))],
    "magenta": [((130, 60, 60), (160, 255, 255))],
}


# ──────────────────────────── Geometry helpers ───────────────────────────

def seg_len(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def seg_angle(x1, y1, x2, y2):
    """Angle of segment in [0, 180) degrees."""
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def infinite_intersection(s1, s2):
    """Intersection of the infinite lines through s1 and s2. None if parallel."""
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def pt_to_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))


def segments_actually_cross(s1, s2, tol=18):
    """
    Return (True, intersection_point) only when the crossing point lies
    *within* each segment's physical extent (± tol pixels of overshoot).

    Uses the parametric form:
        P(t) = s1_start + t*(s1_end - s1_start)   t ∈ [0,1] means inside s1
        Q(u) = s2_start + u*(s2_end - s2_start)   u ∈ [0,1] means inside s2
    Overshoot in pixels = max(0, distance_outside_endpoint).
    """
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return False, None   # parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d

    overshoot1 = max(0.0, -t, t - 1.0) * seg_len(*s1)
    overshoot2 = max(0.0, -u, u - 1.0) * seg_len(*s2)

    if overshoot1 <= tol and overshoot2 <= tol:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return True, (ix, iy)
    return False, None


# ──────────────────────────── Segment merging ────────────────────────────

def _seg_parallel_gap(s1, s2):
    """Gap (px) between two collinear segments along their shared axis. 0 = overlap."""
    x1, y1, x2, y2 = s1
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    ux, uy = dx / L, dy / L
    projs = [
        (x1 - x1) * ux + (y1 - y1) * uy,   # 0
        (x2 - x1) * ux + (y2 - y1) * uy,   # L
        (s2[0] - x1) * ux + (s2[1] - y1) * uy,
        (s2[2] - x1) * ux + (s2[3] - y1) * uy,
    ]
    min1, max1 = min(projs[0], projs[1]), max(projs[0], projs[1])
    min2, max2 = min(projs[2], projs[3]), max(projs[2], projs[3])
    return float(max(0.0, max(min1, min2) - min(max1, max2)))


def merge_collinear_segments(segs, angle_tol=12, dist_tol=8, max_gap=25):
    """Merge nearly-collinear, overlapping/nearby segments into single ones.
    max_gap: don't merge if the two segments are more than this many px apart
             along the line direction — prevents stitching distant fragments.
    """
    if not segs:
        return []
    merged = [tuple(map(float, s)) for s in segs]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_round = []
        for i in range(len(merged)):
            if used[i]:
                continue
            s1 = list(merged[i])
            a1 = seg_angle(*s1)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                s2 = merged[j]
                a2 = seg_angle(*s2)
                da = abs(a1 - a2) % 180
                if min(da, 180 - da) > angle_tol:
                    continue
                # Check collinearity: midpoint of s2 is close to line through s1
                mx2, my2 = (s2[0] + s2[2]) / 2, (s2[1] + s2[3]) / 2
                if pt_to_seg_dist(mx2, my2, *s1) > dist_tol:
                    continue
                # Don't bridge a large gap between two distant fragments
                if _seg_parallel_gap(tuple(s1), s2) > max_gap:
                    continue
                # Merge: keep the two endpoints that are farthest apart
                pts = [(s1[0], s1[1]), (s1[2], s1[3]),
                       (s2[0], s2[1]), (s2[2], s2[3])]
                best_d, best_pair = -1, None
                for p in pts:
                    for q in pts:
                        d = np.hypot(p[0] - q[0], p[1] - q[1])
                        if d > best_d:
                            best_d, best_pair = d, (p, q)
                p, q = best_pair
                s1 = [p[0], p[1], q[0], q[1]]
                a1 = seg_angle(*s1)
                used[j] = True
                changed = True
            next_round.append(tuple(s1))
        merged = next_round
    return merged


# ──────────────────────────── Detection pipeline ─────────────────────────

def detect_segments_by_color(image_bgr):
    """Return {color_name: [seg, ...]} where seg = (x1,y1,x2,y2)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    result = {}

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Fill small gaps left by JPEG compression artifacts
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=1)

        if cv2.countNonZero(mask) < 20:
            continue

        lines = cv2.HoughLinesP(
            mask, rho=1, theta=np.pi / 180,
            threshold=15, minLineLength=15, maxLineGap=10,
        )
        if lines is None:
            continue

        segs = [tuple(map(float, l[0])) for l in lines]
        segs = merge_collinear_segments(segs)
        segs = [s for s in segs if seg_len(*s) > 15]
        if segs:
            result[color] = segs

    return result


def pair_into_crosshairs(segments_by_color, inter_tol=30):
    """
    Greedily pair line segments that cross each other into crosshairs.
    Same-color pairs are preferred; cross-color pairs are a fallback.

    Returns a list of dicts:
        center       – (cx, cy) intersection point
        major_len    – length of the longer arm
        minor_len    – length of the shorter arm
        angle_deg    – orientation of the major axis (0–180°)
        color        – dominant color name
        major_seg    – the longer (x1,y1,x2,y2) segment
        minor_seg    – the shorter segment
    """
    all_segs = []          # (seg, color)
    for color, segs in segments_by_color.items():
        for s in segs:
            all_segs.append((s, color))

    n = len(all_segs)
    used = [False] * n
    crosshairs = []

    # Build a list of all valid pairs sorted by quality score
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            s1, c1 = all_segs[i]
            s2, c2 = all_segs[j]

            # Reject nearly-parallel pairs — they can't form a crosshair
            angle_diff = abs(seg_angle(*s1) - seg_angle(*s2)) % 180
            if angle_diff < 25 or angle_diff > 155:
                continue

            ok, pt = segments_actually_cross(s1, s2, tol=inter_tol)
            if not ok:
                continue
            cx, cy = pt
            mx1, my1 = (s1[0] + s1[2]) / 2, (s1[1] + s1[3]) / 2
            mx2, my2 = (s2[0] + s2[2]) / 2, (s2[1] + s2[3]) / 2
            score = ((np.hypot(cx - mx1, cy - my1) +
                      np.hypot(cx - mx2, cy - my2)) / 2
                     + (0 if c1 == c2 else 40))
            candidates.append((score, i, j, pt))

    candidates.sort(key=lambda x: x[0])

    for score, i, j, pt in candidates:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True

        s1, c1 = all_segs[i]
        s2, c2 = all_segs[j]
        l1, l2 = seg_len(*s1), seg_len(*s2)
        major, minor = (s1, s2) if l1 >= l2 else (s2, s1)

        crosshairs.append({
            "center":    pt,
            "major_len": max(l1, l2),
            "minor_len": min(l1, l2),
            "angle_deg": seg_angle(*major),
            "color":     c1,
            "major_seg": major,
            "minor_seg": minor,
        })

    return crosshairs


# ──────────────────────────── Mask generation ────────────────────────────

def ellipse_mask(shape, cx, cy, major_len, minor_len, angle_deg):
    """Binary mask with a filled ellipse."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    axes = (max(1, int(round(major_len / 2))),
            max(1, int(round(minor_len / 2))))
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                axes, angle_deg, 0, 360, 255, -1)
    return mask


# ──────────────────────────── Main function ──────────────────────────────

def process_image(image_path, output_dir=None, show=True, image_bgr=None):
    """
    image_bgr: optional pre-loaded BGR numpy array (e.g. from a DICOM pixel_array).
               When provided, image_path is only used for naming output files.
    """
    if image_bgr is not None:
        img_bgr = image_bgr
    else:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # ── Step 1: detect segments per color
    segs_by_color = detect_segments_by_color(img_bgr)
    print(f"\nColors detected: {list(segs_by_color.keys())}")
    for c, s in segs_by_color.items():
        print(f"  {c:8s}: {len(s)} segment(s)")

    # ── Step 2: pair segments into crosshairs
    crosshairs = pair_into_crosshairs(segs_by_color)
    print(f"\n{len(crosshairs)} tumor(s) found:")

    # ── Step 3: build masks + figure
    combined = np.zeros((h, w), dtype=np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].set_title(f"Detections  ({len(crosshairs)} tumors)", fontsize=12)
    axes[1].axis("off")
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)

    for idx, ch in enumerate(crosshairs, start=1):
        cx, cy = ch["center"]
        maj = ch["major_len"]
        mi  = ch["minor_len"]
        ang = ch["angle_deg"]

        print(f"  Tumor {idx:2d}: center=({cx:5.0f},{cy:5.0f})  "
              f"major={maj:5.1f}px  minor={mi:5.1f}px  angle={ang:5.1f}°")

        mask = ellipse_mask((h, w), cx, cy, maj, mi, ang)
        combined = cv2.bitwise_or(combined, mask)

        for filled, alpha in [(True, 0.20), (False, 1.0)]:
            ep = Ellipse((cx, cy), width=maj, height=mi, angle=ang,
                         fill=filled, alpha=alpha,
                         facecolor="lime" if filled else "none",
                         edgecolor="lime", linewidth=1.5)
            axes[1].add_patch(ep)
        axes[1].text(cx, cy, f"T{idx}", color="white",
                     fontsize=7, ha="center", va="center",
                     fontweight="bold")

    axes[2].imshow(combined, cmap="gray")
    axes[2].set_title("Segmentation mask", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    # ── Save outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        fig.savefig(out / f"{stem}_result.png", dpi=150, bbox_inches="tight")
        cv2.imwrite(str(out / f"{stem}_mask.png"), combined)
        print(f"\nSaved to: {out}/")

    if show:
        plt.show()
    else:
        plt.close()

    return crosshairs, combined


# ──────────────────────────────── CLI ────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment MRI tumors from crosshair line annotations."
    )
    parser.add_argument("image", nargs="?",
                        default="IM-0001-0023.jpg",
                        help="Path to MRI image (default: IM-0001-0023.jpg)")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip the interactive matplotlib window")
    parser.add_argument("--tol", type=int, default=30,
                        help="Intersection tolerance in pixels (default: 30)")
    args = parser.parse_args()

    process_image(
        args.image,
        output_dir=args.output,
        show=not args.no_show,
    )
