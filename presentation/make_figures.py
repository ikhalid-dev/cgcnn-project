#!/usr/bin/env python3
"""
Generate every figure and the numbers macro file for the CGCNN presentation.
=============================================================================

    python presentation/make_figures.py

Produces, all under presentation/:
    figures/nacl_crystal.png    a chunk of real NaCl, one atom's neighbourhood
                                 highlighted - "the crystal"
    figures/nacl_graph.png      that neighbourhood alone, redrawn as an
                                 abstract graph - "the same thing, as a graph"
    figures/literature.png      our result vs the paper vs the leaderboard
    figures/ensemble.png        single model vs ensemble, both targets
    metrics.tex                 \\def macros for every number the deck quotes,
                                 including Table 1 and GNoME-screen figures
                                 (requires scripts/cgcnn/10 and
                                 scripts/cgcnn/13/14's output to already exist
                                 in results/cgcnn/ - see main()'s own check)

Table 1 and GNoME-screen slides reuse the already-generated PNGs directly
from results/cgcnn/ (table1_validation.png, table1_gamma_diagnostic.png,
gnome_kappa_distributions.png, gnome_overlap_confidence.png) rather than
redrawing them here - same precedent as parity_K_VRH_ens.png etc., already
pulled straight from results/cgcnn/ by the original version of this deck.

WHY NUMBERS ARE MACROS, NEVER TYPED INTO THE SLIDES
----------------------------------------------------
metrics.tex is generated from results/cgcnn/metrics_summary.csv, the same file the
training pipeline writes. The deck \\input{}s it and refers to \\MetricKEnsMAE
etc. If a model is retrained and the numbers move, rerunning this script is the
only thing required to bring the talk back in sync - there is no hand-copied
number anywhere to forget about.

WHY THE NACL DIAGRAM USES REAL DATA, NOT A SCHEMATIC
-----------------------------------------------------
The project's own tutorial notebook (Crystal_graphs.ipynb) builds exactly this
picture - a NetworkX graph of NaCl, atoms coloured by element - from the CIF at
presentation/nacl.cif. Its rendering uses spring_layout, which is fine for
interactively poking at a graph in a notebook but produces crossing edges that
mean nothing physically. Here the SAME data goes through the SAME idea (nodes =
atoms, edges = bonds, colour = element) but atoms are placed at their real,
projected crystallographic positions - the picture a reader already has of
table salt - so nothing crosses that shouldn't.
"""

import json
import math
import os
import sys
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pymatgen.core import Structure

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
# Two results roots since the cgcnn/alignn split: RESULTS is CGCNN's own
# (moduli/kappa/table1/gnome outputs - everything except ALIGNN's), RESULTS_ALIGNN
# is ALIGNN's. Each function below uses whichever is actually correct for what
# it reads, not just whichever was in scope first.
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")
RESULTS_ALIGNN = os.path.join(PROJECT_ROOT, "results", "alignn")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# Digit-prefixed module, so a string import via importlib rather than a plain
# `import` statement - same pattern scripts/cgcnn/05 and scripts/alignn/11,12
# already use to reuse each other's helper functions instead of duplicating
# them. scripts/cgcnn specifically: both 08_compare_kappa and 10_validate_table1
# (the only two modules imported below) live there, not in scripts/alignn.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

# --- Palette - the beamer theme uses the same values (see theme.tex) -------
NAVY = "#0F1B33"
BLUE = "#1450AA"
LIGHT_BLUE = "#6FA8FF"
GREEN = "#1E8E5A"
AMBER = "#D97B12"
INK = "#1F2937"
MUTED = "#5B6B82"
GRID = "#E1E6EF"
NA_COLOR = "#7A3FA0"   # purple, matches the tutorial notebook's Na colour
CL_COLOR = "#1E8E5A"   # green, matches the tutorial notebook's Cl colour


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C3C9D6")
    ax.spines["bottom"].set_color("#C3C9D6")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ===========================================================================
# 1-2. The NaCl figures
# ===========================================================================

def isometric(x, y, z):
    """Project a 3D point onto 2D with a standard isometric transform.

    x and z run "into" the picture at 30 degrees either side of horizontal; y
    is vertical on the page. This is the projection behind every isometric
    game-tile or textbook unit-cell drawing - it renders a cubic lattice
    without the false edge-crossings a force-directed layout would introduce.
    """
    sx = (x - z) * math.cos(math.radians(30))
    sy = (x + z) * math.sin(math.radians(30)) + y
    return sx, sy


def build_nacl_supercell(n=3):
    """Read the real NaCl CIF and expand it to an n x n x n grid of ions.

    NaCl's two interpenetrating FCC sublattices are together exactly a simple
    cubic lattice of alternating ions spaced by a/2 - the standard textbook
    rock-salt picture. Rather than assert that, we DERIVE it: read the actual
    CIF, confirm the spacing and the alternation, and only then build the
    supercell from it - so the figure is traceable to real crystallographic
    data, not a hard-coded assumption.
    """
    structure = Structure.from_file(os.path.join(HERE, "nacl.cif"))
    a = structure.lattice.a
    step = a / 2  # confirmed below to be the true nearest-neighbour distance

    neighbors = structure.get_neighbors(structure[0], r=step + 0.05)
    assert neighbors, "expected nearest neighbours within a/2 + 0.05 A"
    assert all(abs(n.nn_distance - step) < 0.01 for n in neighbors), \
        "nearest-neighbour distance is not a/2 - the supercell logic below " \
        "assumes it is"

    graph = nx.Graph()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                species = "Na" if (i + j + k) % 2 == 0 else "Cl"
                graph.add_node((i, j, k), species=species,
                               pos3d=(i * step, j * step, k * step))
    for (i, j, k) in list(graph.nodes):
        for axis in range(3):
            neighbor = [i, j, k]
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            if neighbor in graph.nodes:
                graph.add_edge((i, j, k), neighbor, length=step)
    return graph, step


def figure_nacl_crystal(path, n=3):
    """A chunk of real NaCl, with one ion's coordination shell highlighted.

    This is deliberately NOT the whole picture - it is the one piece of the
    picture that matters for a CGCNN: given a cutoff radius, which neighbours
    does one atom see? Everything else fades into the background as context.
    """
    graph, step = build_nacl_supercell(n)
    center = (n // 2, n // 2, n // 2)   # the body centre - all 6 neighbours exist
    shell = list(graph.neighbors(center))
    assert len(shell) == 6, f"expected 6 nearest neighbours, got {len(shell)}"

    pos2d = {node: isometric(*data["pos3d"])
             for node, data in graph.nodes(data=True)}
    # Painter's algorithm: draw back-to-front using screen depth as the key,
    # and fade+shrink distant atoms slightly for a believable sense of depth.
    depth = {node: sum(graph.nodes[node]["pos3d"]) for node in graph.nodes}
    d_lo, d_hi = min(depth.values()), max(depth.values())

    def depth_alpha(node):
        t = (depth[node] - d_lo) / max(d_hi - d_lo, 1e-9)
        return 0.35 + 0.55 * t

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # Background bonds: every lattice bond, faint, so the picture reads as
    # "part of something larger" without competing with the highlighted shell.
    for u, v in graph.edges:
        if center in (u, v):
            continue
        (x1, y1), (x2, y2) = pos2d[u], pos2d[v]
        ax.plot([x1, x2], [y1, y2], color="#C7CEDC", linewidth=1.0,
                alpha=0.55, zorder=1)

    # The highlighted shell: bold, in the accent colour, drawn on top.
    for neighbor in shell:
        (x1, y1), (x2, y2) = pos2d[center], pos2d[neighbor]
        ax.plot([x1, x2], [y1, y2], color=BLUE, linewidth=2.6, zorder=3,
                solid_capstyle="round")

    # Atoms, back-to-front. Cl- drawn larger than Na+, matching their real
    # ionic radii (1.67 vs 1.16 A) - a physically honest way to tell them
    # apart beyond colour alone.
    order = sorted(graph.nodes, key=lambda node: depth[node])
    for node in order:
        species = graph.nodes[node]["species"]
        x, y = pos2d[node]
        is_center = node == center
        is_shell = node in shell
        color = NA_COLOR if species == "Na" else CL_COLOR
        radius = 0.135 if species == "Cl" else 0.105
        alpha = 1.0 if (is_center or is_shell) else depth_alpha(node)
        z = 5 if (is_center or is_shell) else 2
        edge_color = INK if (is_center or is_shell) else "none"
        edge_width = 1.6 if is_center else (1.1 if is_shell else 0)
        ax.add_patch(plt.Circle((x, y), radius, facecolor=color, alpha=alpha,
                                edgecolor=edge_color, linewidth=edge_width,
                                zorder=z))
        if is_center:
            # Radius chosen to sit just past the neighbour shell (bond length
            # = step), so the circle visibly ENCLOSES the six highlighted
            # atoms rather than sitting inside them. There is a genuine gap in
            # the bond directions at screen-angle 0 (horizontal) - none of the
            # six bonds travel that way - so the label goes there, with a
            # short leader line, rather than fighting the bonds for space.
            cutoff_r = step * 1.22
            circle = plt.Circle((x, y), cutoff_r, facecolor="none",
                                edgecolor=BLUE, linewidth=1.4,
                                linestyle=(0, (5, 3)), zorder=4)
            ax.add_patch(circle)
            label_x = x + cutoff_r + 0.55
            ax.plot([x + cutoff_r * 0.98, label_x - 0.08], [y, y],
                   color=BLUE, linewidth=1.0, zorder=4)
            ax.text(label_x, y, "cutoff radius $r$", fontsize=12.5, color=BLUE,
                   ha="left", va="center", fontweight="bold", zorder=4)

    # Legend as coloured text rather than a boxed legend key - lighter touch,
    # and it puts the label right next to the colour it names.
    ax.text(0.02, 0.98, "Na$^+$", transform=ax.transAxes, color=NA_COLOR,
           fontsize=15, fontweight="bold", va="top")
    ax.text(0.02, 0.90, "Cl$^-$", transform=ax.transAxes, color=CL_COLOR,
           fontsize=15, fontweight="bold", va="top")
    ax.text(0.98, 0.98, "highlighted bond = 2.81 Å", transform=ax.transAxes,
           color=BLUE, fontsize=11, fontweight="bold", ha="right", va="top")

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figure_nacl_graph(path):
    """The highlighted neighbourhood alone, redrawn as an abstract graph.

    Same seven atoms as the highlighted shell above, same colours - but no
    projection, no perspective, no "the rest of the crystal" in the
    background. This is the moment the slide sequence is building to: a
    crystal IS this, as far as the network ever sees.
    """
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    center_pos = np.array([0.0, 0.0])
    # Six neighbours around a hexagon - not physically an octahedron in 2D,
    # but reads cleanly and every edge is still labelled with its TRUE length.
    angles = [90, 150, 210, 270, 330, 30]
    radius = 2.35
    neighbor_pos = {i: radius * np.array([math.cos(math.radians(a)),
                                          math.sin(math.radians(a))])
                    for i, a in enumerate(angles)}

    for i, pos in neighbor_pos.items():
        ax.plot([center_pos[0], pos[0]], [center_pos[1], pos[1]],
               color=BLUE, linewidth=2.2, zorder=1, solid_capstyle="round")
        mid = (center_pos + pos) / 2
        offset = (pos - center_pos)
        offset = offset / np.linalg.norm(offset) * 0.32
        ax.annotate("2.81 Å", xy=mid, xytext=mid + offset * 0.4,
                   fontsize=10.5, color=MUTED, ha="center", va="center",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="none"))

    for i, pos in neighbor_pos.items():
        ax.add_patch(plt.Circle(pos, 0.40, facecolor=NA_COLOR,
                                edgecolor=INK, linewidth=1.3, zorder=3))
        ax.text(pos[0], pos[1], f"Na", color="white", fontsize=12,
               fontweight="bold", ha="center", va="center", zorder=4)

    ax.add_patch(plt.Circle(center_pos, 0.50, facecolor=CL_COLOR,
                            edgecolor=INK, linewidth=1.6, zorder=3))
    ax.text(center_pos[0], center_pos[1], "Cl", color="white", fontsize=13,
           fontweight="bold", ha="center", va="center", zorder=4)

    ax.text(0, -3.15, "node = one atom  ·  edge = one bond within the cutoff",
           fontsize=12.5, color=INK, fontweight="bold", ha="center")

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.5, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 3. Result charts (same content as before, redrawn in the new palette)
# ===========================================================================

def figure_literature(test, path):
    names = ["Our single\nmodel", "Our 3-model\nensemble", "PINK paper",
             "coGN\n(best published)"]
    values = [test.loc["K_VRH_full", "MAE_log10"],
              test.loc["K_VRH_ens", "MAE_log10"], 0.070, 0.0535]
    colors = [BLUE, BLUE, MUTED, MUTED]
    alphas = [0.45, 1.0, 0.55, 0.55]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bars = ax.bar(names, values, width=0.6)
    for bar, alpha, color, value in zip(bars, alphas, colors, values):
        bar.set_color(color)
        bar.set_alpha(alpha)
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0018,
               f"{value:.4f}", ha="center", va="bottom", fontsize=13,
               color=INK, fontweight="bold")
    ax.set_ylabel("Test MAE, log$_{10}$(GPa)  —  lower is better",
                 color=MUTED, fontsize=12)
    ax.set_ylim(0, max(values) * 1.22)
    ax.tick_params(axis="x", labelsize=12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def figure_ensemble_gain(test, path):
    labels = ["Bulk modulus\n(K_VRH)", "Shear modulus\n(G_VRH)"]
    single = [test.loc["K_VRH_full", "MAE_log10"],
              test.loc["G_VRH_full", "MAE_log10"]]
    ens = [test.loc["K_VRH_ens", "MAE_log10"], test.loc["G_VRH_ens", "MAE_log10"]]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    b1 = ax.bar(x - width / 2 - 0.03, single, width, label="Single model",
               color=MUTED, alpha=0.55)
    b2 = ax.bar(x + width / 2 + 0.03, ens, width, label="3-model ensemble",
               color=BLUE)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                   f"{bar.get_height():.4f}", ha="center", va="bottom",
                   fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Test MAE, log$_{10}$(GPa)", color=MUTED, fontsize=12)
    ax.set_ylim(0, max(single) * 1.25)
    ax.legend(frameon=False, fontsize=11.5, labelcolor=MUTED, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


# ===========================================================================
# 4. Table 1 (real-data validation) and GNoME-screen metrics
# ===========================================================================

def stage2_metrics():
    """Reuse scripts/cgcnn/08_compare_kappa.py's own functions for the
    model-vs-model kappa_L sanity check, on the paper's own 1,213-crystal
    prediction set, through identical (bit-verified) physics."""
    s2 = import_module("08_compare_kappa")
    merged = s2.load_and_merge(
        os.path.join(RESULTS, "pink_kappa_predictions.csv"),
        os.path.join(RESULTS, "pink_reference_kappa.csv"))
    summary = s2.summarise(merged)
    overlap = s2.screening_overlap(merged, top_n=20)
    return {"n": summary["n"], "r": summary["pearson_r_log"],
           "rho": summary["spearman_r"], "mae": summary["mae_log10"],
           "rel_pct": summary["rel_error_pct"], "overlap20": overlap}


def table1_metrics():
    """Reuse scripts/cgcnn/10_validate_table1.py's own verified functions rather
    than recompute the gamma-substitution test by hand. Its sign is easy to
    get backwards (kappa_cal ~ exp(-gamma), not exp(+gamma) - confirmed the
    hard way once already while drafting the extended talk, which is exactly
    the kind of mistake "reuse the physics, don't reimplement it" prevents).
    """
    t1 = import_module("10_validate_table1")
    merged = t1.load_and_merge(
        os.path.join(RESULTS, "table1_kappa_predictions.csv"),
        os.path.join(RESULTS, "table1_reference.csv"))

    ours = t1.compute_metrics(merged, t1.OURS_COL)
    paper = t1.compute_metrics(merged, "kappa_pink")
    corrected = t1.gamma_substitution_test(merged)

    log_err_ours = (np.log10(merged[t1.OURS_COL]) - np.log10(merged["kappa_exp"])).abs()
    log_err_paper = (np.log10(merged["kappa_pink"]) - np.log10(merged["kappa_exp"])).abs()
    ours_closer = int((log_err_ours < log_err_paper).sum())

    log_g = np.log10(merged["G_VRH_pred"] / merged["G_paper"])
    r_g = np.corrcoef(np.log10(merged["G_VRH_pred"]), np.log10(merged["G_paper"]))[0, 1]

    v_long = ((merged["K_VRH_pred"] + 4 * merged["G_VRH_pred"] / 3)
             / merged["Density (g cm-3)"]) ** 0.5 * 1000
    v_trans = (merged["G_VRH_pred"] / merged["Density (g cm-3)"]) ** 0.5 * 1000
    v_sound_ours = ((1 / v_long ** 3 + 2 / v_trans ** 3) / 3) ** (-1 / 3)
    r_vs = np.corrcoef(np.log10(v_sound_ours), np.log10(merged["vs_paper"]))[0, 1]

    gamma_diff = merged["Gruneisen parameter"] - merged["gamma_paper"]
    n_unseen = int((merged.provenance == "unseen").sum())
    n_train = int((merged.provenance == "in_matbench_training").sum())

    return {
        "n": len(merged), "n_train": n_train, "n_unseen": n_unseen,
        "ours_closer": ours_closer, "paper_closer": len(merged) - ours_closer,
        "mae_ours": ours["mae_log10"], "mae_paper": paper["mae_log10"],
        "mae_corrected": corrected["mae_log10"],
        "r_g": r_g, "r_vs": r_vs, "gamma_offset": gamma_diff.mean(),
    }


def alignn_metrics():
    """log10 MAE/R2/raw-GPa MAE from ALIGNN's own Test_results.json, same
    metric definitions scripts/cgcnn/03_evaluate.py's metrics() uses for CGCNN -
    not reused directly since ALIGNN's {target_out, pred_out} JSON shape
    doesn't match that function's expected DataFrame columns, but the
    formulas themselves (log10 MAE, log-space R^2) are the same ones already
    used throughout this file for the CGCNN ensemble and Table 1, not a new
    or risky computation.
    """
    out = {}
    for target, key in (("bulk_modulus_kv", "K"), ("shear_modulus_gv", "G")):
        with open(os.path.join(RESULTS_ALIGNN, f"alignn_{target}", "Test_results.json")) as fh:
            rows = json.load(fh)
        true = np.array([r["target_out"][0] for r in rows])
        pred = np.clip(np.array([r["pred_out"][0] for r in rows]), 1e-3, None)
        log_true, log_pred = np.log10(true), np.log10(pred)
        mae_log = np.mean(np.abs(log_true - log_pred))
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - log_true.mean()) ** 2)
        out[f"{key}Mae"] = mae_log
        out[f"{key}R2"] = 1 - ss_res / ss_tot
        out[f"{key}Gpa"] = np.mean(np.abs(true - pred))
        out[f"{key}Rel"] = (10 ** mae_log - 1) * 100
    return out


def alignn_kappa_metrics():
    """ALIGNN's kappa_L vs. our own CGCNN-ensemble kappa_L, identical physics.

    Reuses scripts/cgcnn/08_compare_kappa.py's summarise()/screening_overlap() via
    importlib - the exact same functions scripts/alignn/16_alignn_predict_kappa.py
    already calls for this comparison, so the deck's numbers can never drift
    from what that script's own printed report says.
    """
    s2 = import_module("08_compare_kappa")
    alignn = pd.read_csv(os.path.join(RESULTS_ALIGNN, "alignn_kappa_predictions.csv"))
    cgcnn = pd.read_csv(os.path.join(RESULTS, "pink_kappa_predictions.csv"))
    merged = (alignn.rename(columns={"Kappa_cal (W m-1 K-1)": "Kappa_cal (W m-1 K-1)_ours"})
             .merge(cgcnn[["material_id", "Kappa_cal (W m-1 K-1)"]]
                   .rename(columns={"Kappa_cal (W m-1 K-1)": "Kappa_cal (W m-1 K-1)_ref"}),
                   on="material_id", how="inner"))
    summary = s2.summarise(merged)
    overlap = s2.screening_overlap(merged, top_n=20)
    return {"n": summary["n"], "r": summary["pearson_r_log"],
           "rho": summary["spearman_r"], "mae": summary["mae_log10"],
           "rel_pct": summary["rel_error_pct"], "overlap20": overlap}


def gnome_metrics():
    """Counts and confidence tiers straight from the already-screened CSVs -
    scripts/13/14 take ~25 min end to end, so this reads their output rather
    than re-running the screen itself every time the deck's figures rebuild.
    """
    candidates = pd.read_csv(os.path.join(RESULTS, "gnome_screen_candidates.csv"))
    overlap = pd.read_csv(os.path.join(RESULTS, "gnome_overlap.csv"))
    width = overlap["Kappa_cal_p95"] / overlap["Kappa_cal_p05"]

    return {
        "n_candidates": len(candidates),
        "n_overlap": len(overlap),
        "pct_of_paper": len(overlap) / 11869 * 100,   # paper's own published count, a fixed external fact
        "pct_of_ours": len(overlap) / len(candidates) * 100,
        "n_tight": int((width < 2).sum()),
        "n_moderate": int(((width >= 2) & (width < 5)).sum()),
        "n_wide": int((width >= 5).sum()),
    }


# ===========================================================================
# 5. Metrics macros
# ===========================================================================

def tex_num(x, digits=4):
    return f"{x:.{digits}f}"


def write_metrics_tex(allrows, test, t1, gn, s2, an, ak, path):
    """Write \\def macros for every number the deck quotes.

    Macro names are CamelCase and start with a letter (TeX control sequences
    cannot contain digits), e.g. \\MetricKEnsMae. Keeping every quoted number
    behind a macro - rather than typed inline - means rerunning this script
    after a retrain is the entire update procedure for the talk.
    """
    k_ens, g_ens = test.loc["K_VRH_ens"], test.loc["G_VRH_ens"]
    k_one, g_one = test.loc["K_VRH_full"], test.loc["G_VRH_full"]
    k_train = allrows[(allrows.tag == "K_VRH_ens") &
                     (allrows.split == "train")].iloc[0]
    gain = (1 - k_ens.MAE_log10 / k_one.MAE_log10) * 100

    lines = ["% Auto-generated by make_figures.py - do not edit by hand.", ""]

    def define(name, value):
        lines.append(f"\\def\\{name}{{{value}}}")

    define("KEnsMae", tex_num(k_ens.MAE_log10))
    define("KEnsRtwo", tex_num(k_ens.R2, 3))
    define("KEnsGpa", tex_num(k_ens.MAE_GPa, 2))
    define("KEnsRel", tex_num(k_ens.rel_error_pct, 1))
    define("KEnsRange", tex_num(k_ens.pct_of_range, 2))

    define("KOneMae", tex_num(k_one.MAE_log10))
    define("KOneRtwo", tex_num(k_one.R2, 3))
    define("KOneGpa", tex_num(k_one.MAE_GPa, 2))
    define("KOneRel", tex_num(k_one.rel_error_pct, 1))

    define("GEnsMae", tex_num(g_ens.MAE_log10))
    define("GEnsRtwo", tex_num(g_ens.R2, 3))
    define("GEnsGpa", tex_num(g_ens.MAE_GPa, 2))
    define("GEnsRel", tex_num(g_ens.rel_error_pct, 1))

    define("GOneMae", tex_num(g_one.MAE_log10))
    define("GOneRtwo", tex_num(g_one.R2, 3))
    define("GOneGpa", tex_num(g_one.MAE_GPa, 2))
    define("GOneRel", tex_num(g_one.rel_error_pct, 1))

    define("KTrainMae", tex_num(k_train.MAE_log10))
    define("EnsembleGainPct", tex_num(gain, 0))
    define("KFactor", tex_num(10 ** k_ens.MAE_log10, 3))

    define("TOneN", t1["n"])
    define("TOneNTrain", t1["n_train"])
    define("TOneNUnseen", t1["n_unseen"])
    define("TOneOursCloser", t1["ours_closer"])
    define("TOnePaperCloser", t1["paper_closer"])
    define("TOneMaeOurs", tex_num(t1["mae_ours"], 3))
    define("TOneMaePaper", tex_num(t1["mae_paper"], 3))
    define("TOneMaeCorrected", tex_num(t1["mae_corrected"], 3))
    define("TOneRG", tex_num(t1["r_g"], 3))
    define("TOneRVs", tex_num(t1["r_vs"], 3))
    define("TOneGammaOffset", tex_num(t1["gamma_offset"], 2))

    define("GnCandidates", f"{gn['n_candidates']:,}".replace(",", "{,}"))
    define("GnOverlap", f"{gn['n_overlap']:,}".replace(",", "{,}"))
    define("GnPctOfPaper", tex_num(gn["pct_of_paper"], 1))
    define("GnPctOfOurs", tex_num(gn["pct_of_ours"], 1))
    define("GnTight", f"{gn['n_tight']:,}".replace(",", "{,}"))
    define("GnModerate", f"{gn['n_moderate']:,}".replace(",", "{,}"))
    define("GnWide", f"{gn['n_wide']:,}".replace(",", "{,}"))

    define("STwoN", s2["n"])
    define("STwoR", tex_num(s2["r"], 3))
    define("STwoRho", tex_num(s2["rho"], 3))
    define("STwoMae", tex_num(s2["mae"], 3))
    define("STwoRel", tex_num(s2["rel_pct"], 1))
    define("STwoOverlap", s2["overlap20"])

    define("AlignnKMae", tex_num(an["KMae"], 4))
    define("AlignnKR2", tex_num(an["KR2"], 3))
    define("AlignnKGpa", tex_num(an["KGpa"], 2))
    define("AlignnKRel", tex_num(an["KRel"], 1))
    define("AlignnGMae", tex_num(an["GMae"], 4))
    define("AlignnGR2", tex_num(an["GR2"], 3))
    define("AlignnGGpa", tex_num(an["GGpa"], 2))
    define("AlignnGRel", tex_num(an["GRel"], 1))

    define("AlignnKappaN", ak["n"])
    define("AlignnKappaR", tex_num(ak["r"], 3))
    define("AlignnKappaRho", tex_num(ak["rho"], 3))
    define("AlignnKappaMae", tex_num(ak["mae"], 3))
    define("AlignnKappaRel", tex_num(ak["rel_pct"], 1))
    define("AlignnKappaOverlap", ak["overlap20"])

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    metrics_path = os.path.join(RESULTS, "metrics_summary.csv")
    if not os.path.exists(metrics_path):
        raise SystemExit(f"No {metrics_path} - run scripts/cgcnn/06_summarise.py first.")
    for needed in ("pink_kappa_predictions.csv", "pink_reference_kappa.csv",
                  "table1_kappa_predictions.csv", "table1_reference.csv",
                  "gnome_screen_candidates.csv", "gnome_overlap.csv"):
        if not os.path.exists(os.path.join(RESULTS, needed)):
            raise SystemExit(f"No results/cgcnn/{needed} - run scripts/cgcnn/08_compare_kappa.py, "
                            f"scripts/cgcnn/10_validate_table1.py, or "
                            f"scripts/cgcnn/13/14_*gnome*.py first (see docs/method.tex).")
    for needed in ("alignn_bulk_modulus_kv/Test_results.json",
                  "alignn_shear_modulus_gv/Test_results.json",
                  "alignn_kappa_predictions.csv"):
        if not os.path.exists(os.path.join(RESULTS_ALIGNN, needed)):
            raise SystemExit(f"No results/alignn/{needed} - run "
                            f"kaggle/run_alignn_kernel.py --fetch and "
                            f"scripts/alignn/16_alignn_predict_kappa.py first "
                            f"(see docs/method.tex).")

    allrows = pd.read_csv(metrics_path)
    test = allrows[allrows.split == "test"].set_index("tag")

    figure_nacl_crystal(os.path.join(FIGS, "nacl_crystal.png"))
    print("wrote figures/nacl_crystal.png")
    figure_nacl_graph(os.path.join(FIGS, "nacl_graph.png"))
    print("wrote figures/nacl_graph.png")
    figure_literature(test, os.path.join(FIGS, "literature.png"))
    print("wrote figures/literature.png")
    figure_ensemble_gain(test, os.path.join(FIGS, "ensemble.png"))
    print("wrote figures/ensemble.png")

    s2 = stage2_metrics()
    t1 = table1_metrics()
    gn = gnome_metrics()
    an = alignn_metrics()
    ak = alignn_kappa_metrics()
    write_metrics_tex(allrows, test, t1, gn, s2, an, ak, os.path.join(HERE, "metrics.tex"))
    print("wrote metrics.tex")


if __name__ == "__main__":
    main()
