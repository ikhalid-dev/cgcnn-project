#!/usr/bin/env python3
"""
Figures for the PINK-reproduction deck that did not already exist.

Generates ONLY what is genuinely missing from results/ - anything already on
disk is used as-is and is not redrawn here. Run:

    /Users/mac/miniconda3/envs/ml_env/bin/python make_pink_figures.py

WHAT IS AND IS NOT REPRODUCIBLE
---------------------------------
PINK's Figures 9 and 10 are DFT phonon calculations on Ag3Te4X - phonon
dispersions, cumulative kappa, specific heat, mode group velocities, scattering
phase space and scattering rates. Reproducing them needs a full DFT +
finite-displacement phonon workflow with third-order force constants on
materials this project has never computed. They are NOT generated here and no
stand-in is drawn for them.

A NOTE ON POPULATIONS
-----------------------
matminer has no local cache, so the 10,987 matbench training STRUCTURES are not
available offline - only n_sites and formulas (data_full/labels.csv). Crystal
system and space group therefore come from the 1,213 CIFs in complete-data/,
which are the PINK prediction set, NOT the training set. Every panel is titled
with the population it actually describes rather than implying otherwise.
"""
import os, warnings, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition, Structure, Element

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

NAVY, BLUE, LIGHT_BLUE = "#0F1B33", "#1450AA", "#6FA8FF"
GREEN, AMBER, INK = "#1E8E5A", "#D97B12", "#1F2937"
MUTED, GRID, PURPLE = "#5B6B82", "#E1E6EF", "#7A3FA0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7, axis="y")
    ax.set_axisbelow(True)


# =============================================================================
#  PINK Figure 4 - dataset statistics
# =============================================================================
def fig4_dataset_stats(path):
    lab = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))

    # --- A/B: symmetry, from the 1,213 CIFs we actually hold ----------------
    cif_dir = os.path.join(ROOT, "complete-data")
    systems, sgnums = [], []
    for fn in sorted(os.listdir(cif_dir)):
        if not fn.endswith(".cif"):
            continue
        try:
            st = Structure.from_file(os.path.join(cif_dir, fn))
            sg, num = st.get_space_group_info()
            sgnums.append(num)
            # crystal system straight from the international number ranges
            for hi, name in [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                             (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                             (230, "cubic")]:
                if num <= hi:
                    systems.append(name); break
        except Exception:
            continue

    ax = axes[0][0]
    order = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
             "trigonal", "hexagonal", "cubic"]
    counts = collections.Counter(systems)
    vals = [counts.get(k, 0) for k in order]
    ax.bar(range(len(order)), vals, color=BLUE, edgecolor="white", lw=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("crystals")
    ax.set_title(f"A   crystal system   (PINK set, n = {len(systems)})",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    ax = axes[0][1]
    ax.hist(sgnums, bins=np.arange(1, 232, 4), color=GREEN, edgecolor="white", lw=0.3)
    ax.set_xlabel("space group number (international)")
    ax.set_ylabel("crystals")
    ax.set_title(f"B   space group   (PINK set, n = {len(sgnums)})",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    # --- C: atoms per cell, full training set --------------------------------
    ax = axes[1][0]
    ax.hist(lab["n_sites"], bins=range(1, 42), color=AMBER, edgecolor="white", lw=0.4)
    ax.set_xlabel("atoms per unit cell")
    ax.set_ylabel("crystals")
    ax.set_title(f"C   cell size   (training set, n = {len(lab):,})",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    # --- D: element frequency, full training set -----------------------------
    ax = axes[1][1]
    freq = collections.Counter()
    for f in lab["formula"]:
        try:
            for e in Composition(str(f)).elements:
                freq[e.symbol] += 1
        except Exception:
            continue
    top = freq.most_common(30)
    ax.bar(range(len(top)), [c for _, c in top], color=PURPLE,
           edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([s for s, _ in top], rotation=90, fontsize=7.5)
    ax.set_ylabel("crystals containing")
    ax.set_title(f"D   30 most common elements   (training set, n = {len(lab):,})",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# =============================================================================
#  Element frequency WITH electronegativity - the annotated version requested
# =============================================================================
def fig_elements_electroneg(path):
    scr = pd.read_csv(os.path.join(ROOT, "results/cgcnn/39_gnome_screen_all_gamma.csv"),
                      dtype={"material_id": str})
    kap = pd.to_numeric(scr["Kappa_cal_derived_matbench"], errors="coerce")
    low = kap <= 1.0

    all_f, low_f = collections.Counter(), collections.Counter()
    for f, is_low in zip(scr["formula"], low):
        try:
            els = [e.symbol for e in Composition(str(f)).elements]
        except Exception:
            continue
        for s in els:
            all_f[s] += 1
            if is_low:
                low_f[s] += 1

    rows = []
    for s, n_all in all_f.items():
        if n_all < 200:
            continue
        try:
            x = Element(s).X
        except Exception:
            x = np.nan
        if not np.isfinite(x):
            continue
        rows.append({"element": s, "n_all": n_all, "n_low": low_f.get(s, 0),
                     "enrichment": (low_f.get(s, 0) / n_all) / (low.sum() / len(scr)),
                     "electronegativity": x})
    d = pd.DataFrame(rows).sort_values("enrichment", ascending=False)

    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    sc = ax.scatter(d["electronegativity"], d["enrichment"],
                    s=np.sqrt(d["n_all"]) * 3.2, c=d["enrichment"],
                    cmap="coolwarm_r", edgecolor="white", lw=0.7, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    for _, r in d.iterrows():
        if r["enrichment"] > 1.25 or r["enrichment"] < 0.6:
            ax.annotate(r["element"], (r["electronegativity"], r["enrichment"]),
                        fontsize=8.5, color=INK, xytext=(0, 7),
                        textcoords="offset points", ha="center")
    ax.set_xlabel("Pauling electronegativity")
    ax.set_ylabel("low-$\\kappa$ enrichment  (1.0 = no preference)")
    ax.set_title("Element frequency against electronegativity, GNoME screen\n"
                 "marker area $\\propto$ how many candidates contain the element",
                 fontsize=11, color=INK)
    style(ax); ax.grid(color=GRID, lw=0.7)
    fig.colorbar(sc, ax=ax, label="enrichment", pad=0.015)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    d.to_csv(path.replace(".png", ".csv"), index=False)
    print("wrote", os.path.relpath(path, ROOT), "+ .csv")


# =============================================================================
#  Model comparison - CGCNN vs ALIGNN vs tree, both datasets
# =============================================================================
def fig_model_comparison(path):
    # Numbers come from this project's own recorded baselines; the tree row is
    # step 43's, the ALIGNN row step 12's, the CGCNN rows 02/05's.
    data = {
        "matbench $K$": {"CGCNN 1": 0.0696, "CGCNN 3-ens": 0.0630,
                         "ALIGNN 1": 0.0539, "tree (RF)": 0.0868},
        "matbench $G$": {"CGCNN 1": 0.0836, "CGCNN 3-ens": 0.0781,
                         "ALIGNN 1": 0.0725, "tree (RF)": 0.1062},
        "AFLOW $K$":    {"CGCNN 1": 0.1154, "CGCNN 3-ens": np.nan,
                         "ALIGNN 1": np.nan, "tree (RF)": 0.0592},
    }
    models = ["CGCNN 1", "CGCNN 3-ens", "ALIGNN 1", "tree (RF)"]
    cols = {"CGCNN 1": LIGHT_BLUE, "CGCNN 3-ens": BLUE,
            "ALIGNN 1": GREEN, "tree (RF)": AMBER}

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    groups = list(data)
    w = 0.19
    for i, m in enumerate(models):
        xs = [g + (i - 1.5) * w for g in range(len(groups))]
        ys = [data[g].get(m, np.nan) for g in groups]
        ax.bar(xs, ys, width=w, color=cols[m], label=m, edgecolor="white", lw=0.6)
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                ax.text(x, y + 0.0025, f"{y:.4f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("MAE $\\log_{10}$ (lower is better)")
    ax.set_title("Model comparison. Bars absent where that model was never trained\n"
                 "on that dataset - not zero, not measured.", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=4)
    style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# =============================================================================
#  Structural families, and the DFT validation set
# =============================================================================
def fig_families(path):
    d = pd.read_csv(os.path.join(ROOT, "results/families/40_family_summary.csv"))
    d = d.sort_values("gnome_candidates", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    ax = axes[0]
    y = range(len(d))
    ax.barh(y, d["gnome_candidates"], color=BLUE, label="screened", edgecolor="white", lw=0.5)
    ax.barh(y, d["gnome_low_kappa"], color=GREEN, label="low-$\\kappa$", edgecolor="white", lw=0.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(d["family"], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("GNoME candidates (log)")
    ax.set_title("A   structural families in the screen", fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    style(ax); ax.grid(color=GRID, lw=0.7, axis="x")

    ax = axes[1]
    rate = (d["gnome_low_kappa"] / d["gnome_candidates"].replace(0, np.nan) * 100)
    ax.barh(list(y), rate, color=AMBER, edgecolor="white", lw=0.5)
    for i, (v, n) in enumerate(zip(rate, d["gnome_low_kappa"])):
        if np.isfinite(v):
            ax.text(v + 0.6, i, f"{v:.0f}%  (n={int(n)})", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([""] * len(d))
    ax.set_xlabel("% of that family called low-$\\kappa$")
    ax.set_title("B   hit rate by family", fontsize=10.5, color=INK, loc="left")
    style(ax); ax.grid(color=GRID, lw=0.7, axis="x")

    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def fig_dft_set(path):
    d = pd.read_csv(os.path.join(ROOT, "dft/dft_materials.csv"))
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    arms = {"B: adjudication": AMBER, "A: consensus low": GREEN, "C: negative control": MUTED}
    x = np.arange(len(d))
    for col, mark, lbl in [("kappa_alignn", "o", "ALIGNN"),
                           ("kappa_cgcnn_ens", "s", "CGCNN ens"),
                           ("kappa_round9", "^", "round 9"),
                           ("kappa_tree", "D", "tree")]:
        ax.scatter(x, d[col], marker=mark, s=44, label=lbl, alpha=0.85,
                   edgecolor="white", lw=0.5, zorder=3)
    for i, arm in enumerate(d["arm"]):
        ax.axvspan(i - 0.5, i + 0.5, color=arms.get(arm, "#fff"), alpha=0.07, zorder=0)
    ax.axhline(1.0, color=AMBER, lw=1.1, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(d["formula"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("predicted $\\kappa_L$ (W m$^{-1}$K$^{-1}$)")
    ax.set_title("The 18-material DFT validation set: four models per material\n"
                 "amber band = adjudication arm, green = consensus low, grey = control",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=4)
    style(ax); ax.grid(color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    fig4_dataset_stats(os.path.join(FIGS, "pink_fig4_dataset_stats.png"))
    fig_elements_electroneg(os.path.join(FIGS, "pink_elements_electroneg.png"))
    fig_model_comparison(os.path.join(FIGS, "model_comparison.png"))
    fig_families(os.path.join(FIGS, "families_classification.png"))
    fig_dft_set(os.path.join(FIGS, "dft_validation_set.png"))


# =============================================================================
#  NaCl: CIF -> ALIGNN atom graph -> line graph, computed not sketched
# =============================================================================
def fig_nacl_alignn_graphs(path, cutoff=5.0, max_nbr=12):
    """Real neighbour lists from nacl.cif at ALIGNN's own graph settings.

    cutoff/max_neighbors are the values in this project's trained ALIGNN config
    (results/alignn/alignn_bulk_modulus_kv/config.json), not round numbers -
    building the picture at different settings would illustrate a graph the
    model never sees.

    Projection is ISOMETRIC, not flat. In rock salt every (x, y) holds both a
    Na and a Cl differing only in z, so an x-y projection draws one species
    exactly on top of the other and the cell looks half empty and single-
    coloured. This bit during development of this very figure.
    """
    def iso(c):
        x, y, z = c
        return (x - y) * np.cos(np.pi / 6), (x + y) * np.sin(np.pi / 6) + z

    st = Structure.from_file(os.path.join(HERE, "nacl.cif"))
    all_nbrs = st.get_all_neighbors(cutoff)
    nbrs = [sorted(n, key=lambda x: x[1])[:max_nbr] for n in all_nbrs]

    n_atoms = len(st)
    n_edges = sum(len(n) for n in nbrs)
    n_line_edges = sum(len(n) * (len(n) - 1) // 2 for n in nbrs)

    fig = plt.figure(figsize=(13.4, 4.6))
    COL = {"Na": PURPLE, "Cl": GREEN}

    # ---- panel A: the crystal ------------------------------------------
    ax = fig.add_subplot(1, 3, 1)
    for site in st:
        X, Y = iso(site.coords)
        ax.scatter(X, Y, s=340, color=COL.get(site.specie.symbol, BLUE),
                   edgecolor="white", lw=1.7, zorder=3)
    for sym, col in COL.items():
        ax.scatter([], [], s=110, color=col, label=sym, edgecolor="white")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(f"A   NaCl from the CIF\n{n_atoms} sites, $a$ = {st.lattice.a:.2f} Å",
                 fontsize=10, color=INK)
    ax.set_aspect("equal"); ax.axis("off")

    # ---- panel B: the ATOM graph ---------------------------------------
    ax = fig.add_subplot(1, 3, 2)
    for i, site in enumerate(st):
        X0, Y0 = iso(site.coords)
        for nb in nbrs[i]:
            X1, Y1 = iso(nb[0].coords)
            ax.plot([X0, X1], [Y0, Y1], color=MUTED, lw=0.45, alpha=0.40, zorder=1)
    for site in st:
        X, Y = iso(site.coords)
        ax.scatter(X, Y, s=300, color=COL.get(site.specie.symbol, BLUE),
                   edgecolor="white", lw=1.7, zorder=3)
    ax.set_title(f"B   atom graph $\\mathcal{{G}}$\n{n_atoms} nodes, {n_edges} directed "
                 f"edges   (cutoff {cutoff} Å, max {max_nbr})", fontsize=10, color=INK)
    ax.set_aspect("equal"); ax.axis("off")

    # ---- panel C: the LINE graph ----------------------------------------
    # Site 0's full line graph is K12 - 66 edges, an unreadable blot. Six of
    # its bonds are drawn so the construction is legible; the true counts are
    # in the title, not implied by the picture.
    ax = fig.add_subplot(1, 3, 3)
    k_true = len(nbrs[0])
    k_show = 6
    ang = np.linspace(0, 2 * np.pi, k_show, endpoint=False) + np.pi / 2
    px, py = np.cos(ang), np.sin(ang)
    for a in range(k_show):
        for b in range(a + 1, k_show):
            ax.plot([px[a], px[b]], [py[a], py[b]], color=AMBER, lw=1.1,
                    alpha=0.75, zorder=1)
    ax.scatter(px, py, s=300, color=GREEN, edgecolor="white", lw=1.6, zorder=3)
    for i in range(k_show):
        ax.text(px[i] * 1.28, py[i] * 1.28, f"$e_{{0{i+1}}}$", ha="center",
                va="center", fontsize=9, color=INK)
    ax.set_title(f"C   line graph $\\mathcal{{L}}$ — 6 of site 0's {k_true} bonds\n"
                 f"every edge is an ANGLE   ({n_line_edges:,} angle edges in the cell)",
                 fontsize=10, color=INK)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor="white"); plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}   "
          f"[{n_atoms} atoms, {n_edges} edges, {n_line_edges} angle edges]")
    return dict(n_atoms=n_atoms, n_edges=n_edges, n_line_edges=n_line_edges,
                k_site0=k_true, a=st.lattice.a)
