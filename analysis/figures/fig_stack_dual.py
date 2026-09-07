#!/usr/bin/env python3
"""Raw vs constrained stack figures for brac62721a_00, from the SAME dual run.

Rebuilds fig_stack_noprior / fig_stack_final honestly: both masks come from
sam2_propagate_dual.py (one SAM2 pass), so constrained ⊆ raw exactly and the
before/after is not confounded by run-to-run non-determinism.

  raw  = brac62721a_00_dual_raw.nii.gz   -> report_figures/fig_stack_noprior.*
  clip = brac62721a_00_dual.nii.gz       -> report_figures/fig_stack_final.*

Green translucent fill + contour over greyscale MR, per-panel slice label,
black background, mm scale bar on the last panel (matches the other stacks).
"""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]   # analysis/figures/ -> repo root
IMG = ROOT / "nnUNet_raw/Dataset502_Tumour/imagesTr/brac62721a_00_0000.nii.gz"
OUTDIR = ROOT / "report_figures"

ZS = list(range(17, 26))          # z = 17..25, same as the existing stacks
GREEN = "lime"
SCALE_MM = 1.0
# shared square crop (from the union bbox y[44,194] x[45,229], padded)
Y0, Y1, X0, X1 = 34, 204, 40, 234


def load(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0


def make_strip(mask, out_stem):
    ref = sitk.ReadImage(str(IMG))
    img = sitk.GetArrayFromImage(ref)
    px_mm = (ref.GetSpacing()[0] + ref.GetSpacing()[1]) / 2.0
    w, h = X1 - X0, Y1 - Y0
    n = len(ZS)
    fig = plt.figure(figsize=(3.0 * n, 3.0))
    fig.patch.set_facecolor("black")
    gap = 0.006
    for k, z in enumerate(ZS):
        ax = fig.add_axes([k / n + gap / 2, gap / 2, 1 / n - gap, 1 - gap])
        sl = img[z].astype(float)
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip(sl, lo, hi), cmap="gray", interpolation="nearest", origin="upper")
        if mask[z].any():
            ax.contourf(mask[z], levels=[0.5, 1.5], colors=[GREEN], alpha=0.35, zorder=3)
            ax.contour(mask[z], levels=[0.5], colors=GREEN, linewidths=1.6, zorder=4)
        ax.text(X0 + 0.045 * w, Y0 + 0.075 * h, f"z = {z}", color="white",
                fontsize=15, va="center", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
        if k == n - 1:
            bar = SCALE_MM / px_mm
            xe, yb = X1 - 0.06 * w, Y1 - 0.10 * h
            ax.add_line(Line2D([xe - bar, xe], [yb, yb], lw=2.5, color="white",
                               solid_capstyle="butt", zorder=6))
            ax.text(xe - bar / 2, yb - 0.03 * h, f"{SCALE_MM:g} mm", color="white",
                    fontsize=12, ha="center", va="bottom", zorder=6)
        ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_axis_off()
    pdf = OUTDIR / f"{out_stem}.pdf"
    fig.savefig(pdf, dpi=400, facecolor="black", pad_inches=0)
    fig.savefig(str(pdf).replace(".pdf", ".png"), dpi=200, facecolor="black", pad_inches=0)
    plt.close(fig)
    print(f"wrote {pdf}  ({int(mask[ZS].sum())} vox over shown slices)")


raw = load(ROOT / "brac62721a_00_dual_raw.nii.gz")
con = load(ROOT / "brac62721a_00_dual.nii.gz")
npr = load(ROOT / "brac62721a_00_noprior_propagated.nii.gz")
make_strip(npr, "fig_stack_noprior")   # the ACTUAL no_prior volume (fixed: was dual_raw)
make_strip(raw, "fig_stack_dualraw")   # dual --emit-raw
make_strip(con, "fig_stack_final")     # dual constrained
