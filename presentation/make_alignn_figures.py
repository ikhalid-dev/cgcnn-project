#!/usr/bin/env python3
"""
Figures for the ALIGNN/PINK deck that did not exist yet.

  alignn_line_graph.png   how ALIGNN turns a CIF into TWO graphs - the atom
                          graph CGCNN already uses, plus the LINE graph whose
                          nodes are bonds and whose edges are bond ANGLES.
                          This is the one structural idea that separates ALIGNN
                          from CGCNN, so the deck needs a picture of it.
  training_data_stats.png the training set's own statistics - this project's
                          answer to PINK's Figure 4.

Palette and axis styling are copied from make_figures.py so the two scripts
produce visually identical output.
"""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

NAVY, BLUE, LIGHT_BLUE = "#0F1B33", "#1450AA", "#6FA8FF"
GREEN, AMBER, INK = "#1E8E5A", "#D97B12", "#1F2937"
MUTED, GRID = "#5B6B82", "#E1E6EF"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =============================================================================
#  1. THE LINE GRAPH - the whole point of ALIGNN
# =============================================================================
def figure_line_graph(path):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    # A small motif: one centre atom with three neighbours. Enough to carry two
    # distinct angles, which is all the idea needs.
    centre = np.array([0.0, 0.0])
    nb = {"1": np.array([-0.95, 0.55]),
          "2": np.array([0.95, 0.55]),
          "3": np.array([0.0, -1.05])}

    # ---- panel A: the crystal, atoms and bonds --------------------------------
    ax = axes[0]
    for k, pos in nb.items():
        ax.plot([centre[0], pos[0]], [centre[1], pos[1]], color=MUTED, lw=2.2, zorder=1)
    ax.scatter(*centre, s=560, color=BLUE, edgecolor="white", lw=2, zorder=3)
    ax.text(centre[0], centre[1], "i", color="white", ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=4)
    for k, pos in nb.items():
        ax.scatter(*pos, s=460, color=LIGHT_BLUE, edgecolor="white", lw=2, zorder=3)
        ax.text(pos[0], pos[1], k, color=NAVY, ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=4)
    # the two angles this motif contains
    for (a, b), col, lbl in [(("1", "2"), AMBER, r"$\theta_{12}$"),
                             (("2", "3"), GREEN, r"$\theta_{23}$")]:
        v1, v2 = nb[a] / np.linalg.norm(nb[a]), nb[b] / np.linalg.norm(nb[b])
        t = np.linspace(0, 1, 40)
        arc = np.array([(v1 * (1 - s) + v2 * s) for s in t])
        arc = 0.42 * arc / np.linalg.norm(arc, axis=1)[:, None]
        ax.plot(arc[:, 0], arc[:, 1], color=col, lw=2.4, zorder=2)
        mid = arc[len(arc) // 2] * 1.55
        ax.text(mid[0], mid[1], lbl, color=col, ha="center", va="center", fontsize=12)
    ax.set_title("A   the crystal\natoms, bonds, and the angles between them",
                 fontsize=10.5, color=INK)

    # ---- panel B: the ATOM graph (what CGCNN sees) ---------------------------
    ax = axes[1]
    for k, pos in nb.items():
        ax.annotate("", xy=pos * 0.80, xytext=centre * 0.8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=2.2))
    ax.scatter(*centre, s=560, color=BLUE, edgecolor="white", lw=2, zorder=3)
    ax.text(centre[0], centre[1], "i", color="white", ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=4)
    for k, pos in nb.items():
        ax.scatter(*pos, s=460, color=LIGHT_BLUE, edgecolor="white", lw=2, zorder=3)
        ax.text(pos[0], pos[1], k, color=NAVY, ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=4)
    for k, pos in nb.items():
        m = (centre + pos) / 2
        ax.text(m[0] + 0.16, m[1], f"$e_{{i{k}}}$", color=MUTED, fontsize=10)
    ax.set_title("B   the ATOM graph  $\\mathcal{G}$\nnodes = atoms, edges = bonds "
                 "(CGCNN stops here)", fontsize=10.5, color=INK)

    # ---- panel C: the LINE graph - bonds become nodes ------------------------
    ax = axes[2]
    # Each BOND of panel B is a NODE here; two bonds sharing atom i are joined,
    # and that new edge carries the angle between them.
    bond_pos = {"$e_{i1}$": np.array([-0.95, 0.45]),
                "$e_{i2}$": np.array([0.95, 0.45]),
                "$e_{i3}$": np.array([0.0, -0.95])}
    pairs = [("$e_{i1}$", "$e_{i2}$", AMBER, r"$\theta_{12}$"),
             ("$e_{i2}$", "$e_{i3}$", GREEN, r"$\theta_{23}$"),
             ("$e_{i1}$", "$e_{i3}$", MUTED, r"$\theta_{13}$")]
    for a, b, col, lbl in pairs:
        pa, pb = bond_pos[a], bond_pos[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, lw=2.6, zorder=1)
        m = (pa + pb) / 2
        off = 0.17 if col is not MUTED else -0.26
        if col is MUTED: m = m + np.array([-0.20, 0.0])
        ax.text(m[0], m[1] + off, lbl, color=col, ha="center", va="center", fontsize=11.5)
    for lbl, pos in bond_pos.items():
        ax.scatter(*pos, s=760, color=GREEN, edgecolor="white", lw=2, zorder=3)
        ax.text(pos[0], pos[1], lbl, color="white", ha="center", va="center",
                fontsize=10.5, fontweight="bold", zorder=4)
    ax.set_title("C   the LINE graph  $\\mathcal{L}$\nnodes = bonds, edges = ANGLES "
                 "(ALIGNN adds this)", fontsize=10.5, color=INK)

    for ax in axes:
        ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.45, 0.95)
        ax.set_aspect("equal"); ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# =============================================================================
#  2. TRAINING-SET STATISTICS - this project's PINK Figure 4
# =============================================================================
def figure_training_stats(path):
    import pandas as pd
    lab = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))

    ax = axes[0]
    ax.hist(np.log10(lab["K_VRH"][lab.K_VRH > 0]), bins=55, color=BLUE,
            edgecolor="white", lw=0.4)
    ax.set_xlabel("$\\log_{10} K_{VRH}$ (GPa)"); ax.set_ylabel("crystals")
    ax.set_title(f"A   bulk modulus\nn = {len(lab):,}", fontsize=10.5, color=INK)
    style_axes(ax)

    ax = axes[1]
    ax.hist(np.log10(lab["G_VRH"][lab.G_VRH > 0]), bins=55, color=GREEN,
            edgecolor="white", lw=0.4)
    ax.set_xlabel("$\\log_{10} G_{VRH}$ (GPa)")
    ax.set_title("B   shear modulus\nthe softer tail is where the screen operates",
                 fontsize=10.5, color=INK)
    style_axes(ax)

    ax = axes[2]
    ax.hist(lab["n_sites"], bins=range(1, 42), color=AMBER, edgecolor="white", lw=0.4)
    ax.set_xlabel("atoms per unit cell")
    ax.set_title("C   cell size\nsmall cells dominate", fontsize=10.5, color=INK)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    figure_line_graph(os.path.join(FIGS, "alignn_line_graph.png"))
    figure_training_stats(os.path.join(FIGS, "training_data_stats.png"))
