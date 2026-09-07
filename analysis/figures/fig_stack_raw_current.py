#!/usr/bin/env python3
"""Regenerate the raw z18-25 strips from the CURRENT volumes, identical crop.

Two new files, one per volume, same tight crop + rendering so they are directly
comparable (retires the stale Aug-20 fig_stack_sam2_18_25):

  brac62721a_00_dual_raw.nii.gz         -> fig_stack_dualraw_current.*
  brac62721a_00_noprior_propagated.nii.gz -> fig_stack_noprior_current.*

The crop is a square centred on the CONSTRAINED lesion cluster over the shown
slices (so it frames the real lesions, matching the earlier tight-zoom figures).
"""
from pathlib import Path
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]   # analysis/figures/ -> repo root
IMG = ROOT / "nnUNet_raw/Dataset502_Tumour/imagesTr/brac62721a_00_0000.nii.gz"
VOLS = [("brac62721a_00_dual_raw.nii.gz",          "fig_stack_dualraw_current"),
        ("brac62721a_00_noprior_propagated.nii.gz", "fig_stack_noprior_current")]
ZS = list(range(18, 26))
GREEN = "lime"; SCALE_MM = 1.0

ref = sitk.ReadImage(str(IMG))
img = sitk.GetArrayFromImage(ref)
px_mm = (ref.GetSpacing()[0] + ref.GetSpacing()[1]) / 2.0
H, W = img.shape[1:]

# shared tight square crop from the CONSTRAINED cluster over the shown slices
con = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "brac62721a_00_dual.nii.gz"))) > 0
ys, xs = [], []
for z in ZS:
    yy, xx = np.where(con[z]); ys += list(yy); xs += list(xx)
cy, cx = (min(ys)+max(ys))//2, (min(xs)+max(xs))//2
half = int(max(max(ys)-min(ys), max(xs)-min(xs)) / 2 * 1.15)
Y0, Y1 = max(0, cy-half), min(H, cy+half)
X0, X1 = max(0, cx-half), min(W, cx+half)
w, h = X1-X0, Y1-Y0
print(f"crop y[{Y0},{Y1}] x[{X0},{X1}]")


def make(vol_fn, stem):
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / vol_fn))) > 0
    n = len(ZS)
    fig = plt.figure(figsize=(3.0*n, 3.0)); fig.patch.set_facecolor("black")
    gap = 0.006
    for k, z in enumerate(ZS):
        ax = fig.add_axes([k/n + gap/2, gap/2, 1/n - gap, 1 - gap])
        sl = img[z].astype(float); lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray", interpolation="nearest")
        if m[z].any():
            ax.contourf(m[z], levels=[.5, 1.5], colors=[GREEN], alpha=.35, zorder=3)
            ax.contour(m[z], levels=[.5], colors=GREEN, linewidths=1.6, zorder=4)
        ax.text(X0+0.045*w, Y0+0.075*h, f"z = {z}", color="white", fontsize=15,
                va="center", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=.55))
        if k == n-1:
            bar = SCALE_MM / px_mm; xe, yb = X1-0.06*w, Y1-0.10*h
            ax.add_line(Line2D([xe-bar, xe], [yb, yb], lw=2.5, color="white",
                               solid_capstyle="butt", zorder=6))
            ax.text(xe-bar/2, yb-0.03*h, f"{SCALE_MM:g} mm", color="white",
                    fontsize=12, ha="center", va="bottom", zorder=6)
        ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT/"report_figures"/f"{stem}.{ext}", dpi=200, facecolor="black", pad_inches=0)
    plt.close(fig)
    print(f"wrote {stem}  ({int(m.sum())} vox total, {int(m[ZS].sum())} over shown slices)")


for vf, st in VOLS:
    make(vf, st)
