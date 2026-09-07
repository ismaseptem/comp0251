#!/usr/bin/env python3
"""Two-panel 3-D tumour volume: unconstrained vs constrained (same view/sizing).

This variant reads the single-run DUAL pair of NIfTI volumes (constrained _dual
and unconstrained _dual_raw, from ONE SAM2 pass) instead of PNG slice stacks, so
constrained is an exact subset of unconstrained — no cross-run jitter. Spacing is
taken from the volume header. Separate output file from fig_3d_pair.*."""
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

ROOT = Path(__file__).resolve().parents[2]   # analysis/figures/ -> repo root
OUT  = ROOT / "report_figures" / "fig_3d_pair_dual.pdf"
CON_NII = ROOT / "brac62721a_00_dual.nii.gz"       # constrained
RAW_NII = ROOT / "brac62721a_00_dual_raw.nii.gz"   # unconstrained
GREEN = np.array([0.0, 0.62, 0.45])                # #009E73
ELEV, AZIM = 18, -62

def load_nii(p):
    img = sitk.ReadImage(str(p))
    vol = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)   # (z,y,x)
    sx, sy, sz = img.GetSpacing()                              # mm (x,y,z)
    return vol, (sx, sy, sz)

con, spacing = load_nii(CON_NII)
raw, _       = load_nii(RAW_NII)
SX, SY, SZ = spacing
vols = {"Unconstrained": raw, "Constrained": con}

def mesh(vol):
    # pad so surfaces close at volume borders; spacing maps voxels -> mm
    vp = np.pad(vol, 1)
    verts, faces, _, _ = measure.marching_cubes(vp, level=0.5, spacing=(SZ, SY, SX), step_size=1)
    tris = verts[faces]                                    # (F,3,3) columns = (z,y,x) mm
    # simple Lambert shading from face normals
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.35, 0.35, 0.87]); light /= np.linalg.norm(light)
    shade = 0.58 + 0.42 * np.clip(np.abs(n @ light), 0, 1)               # (F,) brighter for white bg
    colors = np.clip(GREEN[None, :] * shade[:, None], 0, 1)
    # reorder columns -> plot axes (X=x, Y=y, Z=z) so it stands upright
    tris_xyz = tris[:, :, ::-1]                            # (z,y,x) -> (x,y,z)
    return tris_xyz, colors

# shared limits tightened to the occupied region (union of both), +2 mm margin
def bounds(v):
    zz, yy, xx = np.where(v > 0)
    return (xx.min()*SX, xx.max()*SX, yy.min()*SY, yy.max()*SY, zz.min()*SZ, zz.max()*SZ)
bb = np.array([bounds(v) for v in vols.values()])
m = 2.0
xlim = (bb[:, 0].min()-m, bb[:, 1].max()+m)
ylim = (bb[:, 2].min()-m, bb[:, 3].max()+m)
zlim = (bb[:, 4].min()-m, bb[:, 5].max()+m)
aspect = (xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])

def draw_gizmo(fig, rect):
    """Small orientation triad (same view); z drawn boldly = through-plane axis."""
    axg = fig.add_axes(rect, projection="3d")
    axg.view_init(elev=ELEV, azim=AZIM)
    axg.quiver(0, 0, 0, 1, 0, 0, color="0.35", arrow_length_ratio=0.25, lw=1.4)
    axg.quiver(0, 0, 0, 0, 1, 0, color="0.35", arrow_length_ratio=0.25, lw=1.4)
    axg.quiver(0, 0, 0, 0, 0, 1, color="#C1121F", arrow_length_ratio=0.28, lw=2.2)
    axg.text(1.15, 0, 0, "x", fontsize=9, color="0.25", ha="center", va="center")
    axg.text(0, 1.2, 0, "y", fontsize=9, color="0.25", ha="center", va="center")
    axg.text(0, 0, 1.30, "z", fontsize=11, color="#C1121F",
             ha="center", va="bottom", fontweight="bold")
    axg.set_xlim(-0.2, 1.2); axg.set_ylim(-0.2, 1.2); axg.set_zlim(-0.2, 1.2)
    axg.set_box_aspect((1, 1, 1)); axg.set_axis_off(); axg.patch.set_alpha(0)

def place(ax):
    try:    ax.set_box_aspect(aspect, zoom=1.45)   # mpl>=3.7: enlarge mesh in cell
    except TypeError:
        ax.set_box_aspect(aspect)
        try: ax.dist = 7.0
        except Exception: pass

plt.rcParams.update({"font.family": "sans-serif"})
fig = plt.figure(figsize=(11, 4.4)); fig.patch.set_facecolor("white")
rects = {"Unconstrained": [0.005, 0.02, 0.49, 0.96],
         "Constrained":   [0.505, 0.02, 0.49, 0.96]}
for name, vol in vols.items():
    ax = fig.add_axes(rects[name], projection="3d")
    ax.set_facecolor("white")
    tris, colors = mesh(vol)
    pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
    pc.set_rasterized(True)                          # embed mesh as one image, not vector paths
    ax.add_collection3d(pc)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    place(ax)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    vmm = vol.sum() * SX * SY * SZ
    ax.text2D(0.5, 0.99, name, transform=ax.transAxes, color="black",
              fontsize=15, ha="center", va="top", fontweight="bold")
    ax.text2D(0.5, 0.10, f"{vmm/1000:.2f} cm³", transform=ax.transAxes,
              color="black", fontsize=12, ha="center")

draw_gizmo(fig, [0.045, 0.06, 0.13, 0.30])          # bottom-left triad
fig.text(0.045, 0.03, "z = through-plane axis", color="#C1121F", fontsize=11,
         fontweight="bold", ha="left", va="bottom")   # clear bottom-left margin
fig.savefig(OUT, dpi=400, facecolor="white", pad_inches=0.01, bbox_inches="tight")
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, facecolor="white", pad_inches=0.01, bbox_inches="tight")
print("wrote", OUT, "| spacing mm:", (round(SX,4), round(SY,4), round(SZ,4)),
      "| volumes cm^3:", {n: round(v.sum()*SX*SY*SZ/1000, 2) for n, v in vols.items()})
