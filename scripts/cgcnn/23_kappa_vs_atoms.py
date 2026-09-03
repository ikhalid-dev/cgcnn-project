#!/usr/bin/env python3
"""
STEP 23 - How predicted Kappa_cal changes with primitive-cell size.
================================================================================

    python scripts/cgcnn/23_kappa_vs_atoms.py

WHAT THIS IS
-------------
19_small_cells.py and 21_feature_importance.py both touched on this from
different angles - the small-cell filter found atom count matters, and the
feature-importance ranking found it is the 3rd-most-important predictor of
Kappa_cal (after the two moduli). This script draws the actual relationship:
every one of the 33,323 screened candidates, Kappa_cal (log scale - it spans
nearly 5 orders of magnitude, 0.003 to 81 W/m/K) against primitive-cell atom
count, plus the per-atom-count MEDIAN as a trend line (median, not mean, so
a handful of high-kappa outliers at any given atom count cannot distort the
line - the raw scatter cloud is still shown underneath for anyone who wants
the full spread, not just the summary).

WHY A DOWNWARD TREND IS EXPECTED, NOT SURPRISING
------------------------------------------------------
The Slack-model formula itself (docs/oxide_screen_columns.tex, Eq. 4) has
Kappa_cal directly proportional to 1/N_atoms, and Volume grows roughly
linearly with N_atoms for chemically similar materials, so V^(1/3)/N_atoms
falls off roughly as N_atoms^(-2/3) even before any change in chemistry is
considered. So part of this trend is built into the physics, not a new
empirical discovery - what the plot actually shows beyond that is how much
SPREAD remains at each atom count (chemistry still matters a lot - see the
halogen-enrichment finding in 18_element_frequency.py) and where the trend
flattens out.

WHY A SECOND, LOG-LOG PLOT
-------------------------------
A power law (Kappa ~ N^(-2/3)) is only a STRAIGHT LINE on a log-log plot -
on the first plot's linear-x/log-y axes, even a perfect N^(-2/3) relationship
would show up as a downward-curving line, not a straight one. So a second
figure (gnome_kappa_vs_atoms_loglog.png) redraws the same data with atom
count also on a log axis, with a fitted line (ordinary least squares in
log-log space, all 33,323 points) overlaid to show the empirical slope
directly. Checked once, not asserted: the fitted slope comes out to -0.705,
close to the pure-geometry -2/3 prediction - but the correlation behind that
fit is weak (r=-0.21 on log-log Pearson), because atom count is a real but
minor lever compared to chemistry (log(G_VRH_pred) alone correlates with
log(Kappa_cal) at r=0.95 - see 21_feature_importance.py). The fitted line
describes the real central tendency; the wide scatter around it is real
chemistry variation, not noise to be explained away.
"""

import os  # path joining and filesystem path handling

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - lets this run headless with no display attached
import matplotlib.pyplot as plt  # plotting API used throughout this script
import numpy as np  # log10, polyfit, corrcoef - the log-log fit machinery
import pandas as pd  # CSV reading and the groupby/aggregate used to build the per-atom-count table

# walk up three directories from this file (scripts/cgcnn/23_...py -> scripts/cgcnn -> scripts -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")
SCREEN_CSV = os.path.join(RESULTS, "13_gnome_screen_all.csv")  # written by 13_screen_gnome.py - the full scored candidate table
OUT_PNG = os.path.join(RESULTS, "23_gnome_kappa_vs_atoms.png")  # linear-x/log-y scatter + median trend
OUT_PNG_LOGLOG = os.path.join(RESULTS, "23_gnome_kappa_vs_atoms_loglog.png")  # log-log version with the fitted power law
OUT_CSV = os.path.join(RESULTS, "23_gnome_kappa_vs_atoms_by_atomcount.csv")  # per-atom-count n/median/p25/p75 table

# hex colors used consistently across every plot in this script
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# global matplotlib style overrides so every figure this script draws shares one look
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_loglog(df, reliable, path):
    """Same data, both axes logged - a power-law relationship (the
    theoretical Kappa ~ N^(-2/3) from pure cell-size geometry) is a straight
    line here, unlike on the linear-x version. Overlays the fitted OLS line
    (log-log, all 33,323 points) so the empirical slope is visible directly
    rather than asserted in prose."""
    logN = np.log10(df["Number of Atoms"].values)  # log10 of atom count for every screened candidate
    logK = np.log10(df["Kappa_cal (W m-1 K-1)"].values)  # log10 of predicted kappa for every screened candidate
    slope, intercept = np.polyfit(logN, logK, 1)  # ordinary least-squares fit of a degree-1 polynomial (a line) in log-log space
    r = np.corrcoef(logN, logK)[0, 1]  # Pearson correlation between logN and logK; [0,1] pulls the off-diagonal entry out of the 2x2 correlation matrix

    fig, ax = plt.subplots(figsize=(9.5, 5.5))  # one figure/axes pair, 9.5x5.5 inches
    ax.scatter(df["Number of Atoms"], df["Kappa_cal (W m-1 K-1)"],
              s=6, color=BLUE, alpha=0.12, edgecolor="none",
              label=f"All screened candidates (n={len(df)})")  # every candidate as a small, mostly-transparent dot so overlapping points show density
    ax.plot(reliable.index, reliable["median"], "o", color=ORANGE, markersize=4,
           label=f"Median Kappa_cal per atom count (n >= 20)")  # one orange marker per atom count that has a reliable (n>=20) sample

    x_line = np.array([df["Number of Atoms"].min(), df["Number of Atoms"].max()])  # the two x-endpoints spanning the full data range
    y_line = 10 ** (intercept + slope * np.log10(x_line))  # the fitted line evaluated at those two endpoints, converted back out of log space
    ax.plot(x_line, y_line, color=INK, linewidth=1.6, linestyle="--",
           label=f"Fitted trend: Kappa $\\propto$ N$^{{{slope:.2f}}}$ (r={r:.2f})")  # dashed line connecting the two fitted endpoints

    # Anchored to the fitted line's own value at the left edge, so the two
    # lines are directly comparable - only their SLOPES differ (-2/3 exact
    # geometry vs. the real fitted -0.70ish), not their vertical placement.
    y0 = 10 ** (intercept + slope * np.log10(x_line[0]))  # the fitted line's height at the smallest atom count
    y_theory = y0 * (x_line / x_line[0]) ** (-2 / 3)  # a pure N^(-2/3) power law starting from that same height
    ax.plot(x_line, y_theory, color=MUTED, linewidth=1.2, linestyle=":",
           label="Pure-geometry prediction: Kappa $\\propto$ N$^{-2/3}$")  # dotted reference line for the theoretical slope

    ax.set_xscale("log")  # log x-axis (atom count)
    ax.set_yscale("log")  # log y-axis (kappa)
    ax.set_xlabel("Primitive-cell atom count, log scale")
    ax.set_ylabel("Predicted Kappa_cal (W m$^{-1}$ K$^{-1}$), log scale")
    ax.set_title("Kappa_cal vs. cell size, log-log")
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()  # auto-adjust margins so labels/titles are not clipped
    fig.savefig(path, dpi=160)  # write the PNG to disk
    plt.close(fig)  # free the figure's memory now that it has been saved


def main():
    print("=== Kappa_cal vs. primitive-cell atom count ===\n")
    df = pd.read_csv(SCREEN_CSV)  # load the full scored candidate table
    print(f"Loaded {len(df)} scored candidates from {SCREEN_CSV}")

    # group rows by atom count, then compute four summary stats of Kappa_cal within each group
    by_atoms = df.groupby("Number of Atoms")["Kappa_cal (W m-1 K-1)"].agg(
        n="size", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
    by_atoms.to_csv(OUT_CSV)  # write the per-atom-count summary table, indexed by atom count
    print(f"Wrote {OUT_CSV} ({len(by_atoms)} distinct atom counts, "
         f"n/median/p25/p75 of Kappa_cal at each)")

    # Past ~40 atoms the screen thins out fast (down to single-digit or n=1
    # counts - e.g. exactly one candidate at 49 atoms), so a per-atom-count
    # median there is not a trend, it is that one candidate's value. Drawing
    # a trend line through it would show a spike that is pure sampling noise,
    # not a real physical effect - so the line/band only cover atom counts
    # with a reasonably sized sample (n >= MIN_N), same standard a real
    # trend estimate would apply. The raw scatter still shows every point,
    # sparse tail included.
    MIN_N = 20
    reliable = by_atoms[by_atoms["n"] >= MIN_N]  # keep only atom counts with at least MIN_N candidates behind their median
    n_dropped = len(by_atoms) - len(reliable)  # how many atom counts were excluded from the trend line for being too sparse
    print(f"Trend line drawn for the {len(reliable)} atom counts with n >= {MIN_N} "
         f"({n_dropped} sparser atom counts excluded from the line, e.g. the single "
         f"candidate at 49 atoms - still shown as raw scatter points)")

    print(f"\nMedian Kappa_cal: {reliable['median'].iloc[0]:.2f} W/m/K at "
         f"{reliable.index[0]} atoms -> {reliable['median'].iloc[-1]:.2f} W/m/K at "
         f"{reliable.index[-1]} atoms")  # first vs. last reliable atom count's median, read via positional indexing

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(df["Number of Atoms"], df["Kappa_cal (W m-1 K-1)"],
              s=6, color=BLUE, alpha=0.12, edgecolor="none",
              label=f"All screened candidates (n={len(df)})")  # raw scatter, every candidate
    ax.plot(reliable.index, reliable["median"], color=ORANGE, linewidth=2.0,
           label=f"Median Kappa_cal per atom count (n >= {MIN_N})")  # solid median trend line, reliable atom counts only
    ax.fill_between(reliable.index, reliable["p25"], reliable["p75"],
                    color=ORANGE, alpha=0.15, linewidth=0, label="25th-75th percentile band")  # shaded interquartile band around the median line
    ax.axhline(1.0, color=INK_SOFT, linewidth=0.9, linestyle=":")  # horizontal reference line at the project's kappa_L=1 threshold
    ax.text(df["Number of Atoms"].max() * 0.99, 1.0, "kappa_L = 1 W/m/K threshold ",
           fontsize=8.5, color=INK_SOFT, va="bottom", ha="right")  # label for that reference line, anchored near the right edge

    ax.set_yscale("log")  # log y-axis; x stays linear (this is the non-loglog figure)
    ax.set_xlabel("Primitive-cell atom count")
    ax.set_ylabel("Predicted Kappa_cal (W m$^{-1}$ K$^{-1}$), log scale")
    ax.set_title("Predicted lattice thermal conductivity vs. cell size")
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)  # write the linear-x figure to disk
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")

    plot_loglog(df, reliable, OUT_PNG_LOGLOG)  # build and write the second, log-log figure
    print(f"Wrote {OUT_PNG_LOGLOG}")


if __name__ == "__main__":  # only run main() when executed as a script, not when imported as a module
    main()
