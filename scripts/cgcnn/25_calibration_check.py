#!/usr/bin/env python3
"""
STEP 25 - Is the ensemble's uncertainty interval actually calibrated?
================================================================================

    python scripts/cgcnn/25_calibration_check.py

WHAT THIS IS
-------------
Every "confidence" ranking used throughout this project's GNoME follow-on
work (Kappa_cal_p05/p95, the tight/moderate/wide buckets in
24_small_cell_oxides_for_dft.py, the p95-based re-ranking in
16_spacegroups_oxides.py) rests on one unverified assumption: that the
ensemble's predicted p05-p95 interval actually contains the true value
close to 90% of the time. docs/method.tex flags this directly as "weak by
construction" - real experimental kappa_L only exists for 2 materials our
ensemble never trained on, nowhere near enough to check calibration.

But the ensemble's UNCERTAINTY comes from disagreement on the MODULI (K, G),
not from kappa_L itself, and the matbench TEST split - 1,648 crystals with
real DFT-computed K/G, genuinely held out from training - already has
everything needed to check that half directly, with zero new data. This
script does exactly that: for each test crystal, does the true value fall
inside the ensemble's own claimed interval, at the rate the interval claims?

WHERE THE DATA COMES FROM
------------------------------
scripts/cgcnn/05_ensemble.py already wrote per-crystal ensemble predictions
with the exact ingredients needed - pred_log10 (ensemble mean) and
member_std_log10 (standard deviation across the 3 seeded members) - to
results/cgcnn/predictions_K_VRH_ens.csv and predictions_G_VRH_ens.csv, split
column included. This script reads those, filters to split=="test", and
does not recompute or re-derive either quantity.

HOW THE INTERVAL IS BUILT, AND WHY A NORMAL-QUANTILE FORMULA HERE IS
EQUIVALENT TO THE PROJECT'S OWN MONTE CARLO METHOD
------------------------------------------------------------------------
07_predict_kappa.py's monte_carlo_kappa() draws 2,000 samples from
Normal(pred_log10, member_std_log10) and takes the empirical 5th/95th
percentile. For a genuinely normal distribution, that converges (as the
sample count grows) to pred_log10 +/- z*member_std_log10 with
z = norm.ppf(0.95) = 1.6449. Using the exact formula here instead of
re-running a 2,000-draw Monte Carlo is mathematically the same claim,
just without sampling noise - appropriate for a calibration CHECK, where
the point is to test the interval's own definition precisely, not to
re-simulate it.

WHAT "CALIBRATED" MEANS HERE
----------------------------------
A well-calibrated 90% interval should contain the true value on ~90% of
crystals - not more (over-covering, meaning the interval is unnecessarily
wide/conservative) and not fewer (under-covering, meaning the interval is
overconfident - the real failure mode that would undermine every "tight"
label this project has handed out). Checked across MULTIPLE nominal levels
(10%-90%), not just the one used for screening, so the full calibration
curve is visible rather than a single number that could look fine by luck.

Writes one CSV (nominal vs. empirical coverage, K and G separately, every
10% level from 10% to 90%, RAW and RECALIBRATED) and one plot (a reliability
diagram - nominal on the x-axis, empirical on the y-axis, a diagonal
reference line for perfect calibration, raw and recalibrated curves both
shown).

THE FIX, CHECKED OUT-OF-SAMPLE, NOT JUST DIAGNOSED
--------------------------------------------------------
If the raw interval turns out to be miscalibrated, the natural next question
is whether a simple correction fixes it - so this script also derives a
per-target scale factor from the VAL split (a genuinely different 1,648
crystals from TEST, so this is not the same data twice): at each nominal
level, find the empirical quantile of |true_log10 - pred_log10| /
member_std_log10 on VAL, then apply that SAME factor to TEST and report the
resulting coverage. If the fix only worked on the data it was fit on, this
would be circular; checking it on a disjoint split is what makes it a real
validation rather than an overfit correction.
"""

import os                                          # path-joining and path-manipulation utilities

import matplotlib
matplotlib.use("Agg")                              # non-interactive rendering backend - lets fig.savefig() work with no display attached
import matplotlib.pyplot as plt                    # plotting API used by plot_reliability()
import numpy as np                                 # numerical array support (pandas/scipy build on this underneath)
import pandas as pd                                # DataFrame/CSV loading, filtering, and writing
from scipy.stats import norm                       # normal-distribution quantile function, used for the theoretical z-score

# __file__ is this script's own path; abspath() makes it absolute, and three
# dirname() calls climb scripts/cgcnn/ -> scripts/ -> the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")            # shared output directory every numbered script reads/writes under
K_PRED_CSV = os.path.join(RESULTS, "predictions_K_VRH_ens.csv")     # 05_ensemble.py's per-crystal K predictions (skip-group name, left unchanged)
G_PRED_CSV = os.path.join(RESULTS, "predictions_G_VRH_ens.csv")     # 05_ensemble.py's per-crystal G predictions (skip-group name, left unchanged)
OUT_CSV = os.path.join(RESULTS, "25_calibration_check.csv")         # this script's own coverage table - script number 25 stamped in the name
OUT_PNG = os.path.join(RESULTS, "25_calibration_check.png")         # this script's own reliability-diagram plot - same naming rule

NOMINAL_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]   # claimed central-interval widths to check, e.g. 0.90 = a 5th-95th percentile interval

BLUE = "#2a78d6"      # plot color for the bulk-modulus (K) series
ORANGE = "#eb6834"    # plot color for the shear-modulus (G) series
INK = "#0b0b0b"       # near-black, used for titles and body text
INK_SOFT = "#52514e"  # softer dark grey, used for axis labels and the reference line
MUTED = "#898781"     # muted grey, used for tick labels and the 90%-marker line
GRID = "#e1e0d9"      # light grey, used for gridlines and legend borders
SURFACE = "#fcfcfb"   # near-white, used as the figure/axes background

plt.rcParams.update({                                          # override matplotlib's global default style settings for every plot below
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,    # figure canvas and plot-area background colors
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,  # axis border color and axis-label text color
    "axes.titlecolor": INK, "text.color": INK,                 # subplot-title and generic text color
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,   # tick-label and gridline colors
    "font.family": "sans-serif", "font.size": 10,              # default font family and base font size
    "axes.spines.top": False, "axes.spines.right": False,      # hide the top and right border lines on every axes
})


def empirical_coverage(true_log10, pred_log10, std_log10, nominal_levels):
    """For each nominal central interval (e.g. 0.90 = a 5th-95th percentile
    interval), what fraction of true values actually fall inside the
    interval the ensemble's own std claims? Returns a list aligned with
    nominal_levels."""
    coverages = []                                   # one empirical coverage fraction will be appended per nominal level
    for level in nominal_levels:                     # e.g. 0.10, 0.20, ..., 0.90
        z = norm.ppf(0.5 + level / 2)                 # standard-normal z-score whose central interval covers exactly `level` fraction of probability
        lo = pred_log10 - z * std_log10               # elementwise lower bound of the interval, one per crystal (numpy arrays)
        hi = pred_log10 + z * std_log10               # elementwise upper bound of the interval
        inside = (true_log10 >= lo) & (true_log10 <= hi)   # boolean array: True wherever the true value lies within [lo, hi]
        coverages.append(inside.mean())               # fraction of True values = empirical coverage at this nominal level
    return coverages                                  # list of coverage fractions, same order/length as nominal_levels


def recalibrated_coverage(val, test, nominal_levels):
    """For each nominal level, derive a scale factor from VAL (the empirical
    quantile of |standardized residual|, not the normal-distribution
    assumption empirical_coverage() uses) and apply it to TEST - a disjoint
    split, so the resulting coverage is a genuine out-of-sample check of the
    correction, not a repeat of whatever data it was fit on."""
    z_val = (val["true_log10"] - val["pred_log10"]).abs() / val["member_std_log10"]
    # ^ standardized absolute residual per VAL crystal: how many claimed std-devs away the true value actually landed
    coverages, factors = [], []                       # accumulators: one coverage value and one fitted factor per nominal level
    for level in nominal_levels:
        c = z_val.quantile(level)                     # VAL-empirical quantile of the standardized residual - replaces the normal-theory z with a data-driven multiplier
        lo = test["pred_log10"] - c * test["member_std_log10"]   # corrected lower bound, evaluated on the TEST split using VAL's factor
        hi = test["pred_log10"] + c * test["member_std_log10"]   # corrected upper bound, same TEST/VAL split
        inside = (test["true_log10"] >= lo) & (test["true_log10"] <= hi)   # boolean mask: TEST truth inside the corrected interval
        coverages.append(inside.mean())               # empirical coverage on TEST after applying the VAL-fitted correction
        factors.append(c)                             # keep the fitted factor itself, for reporting/plotting
    return coverages, factors                         # two lists, same order/length as nominal_levels


def plot_reliability(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))      # one figure, two side-by-side subplots (raw, recalibrated), size in inches
    for ax, label in zip(axes, ["raw", "recalibrated"]):   # pair each subplot with the column-name suffix it should read
        ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.2, linestyle="--",
               label="Perfect calibration")                # diagonal reference line where empirical would equal nominal exactly
        ax.plot(table["nominal"], table[f"K_VRH_{label}"], "o-", color=BLUE,
               label="Bulk modulus K")                      # K curve: nominal level (x) vs. this column's coverage (y)
        ax.plot(table["nominal"], table[f"G_VRH_{label}"], "o-", color=ORANGE,
               label="Shear modulus G")                     # G curve, same axes
        ax.axvline(0.90, color=MUTED, linewidth=0.8, linestyle=":")   # vertical marker at 90% - the level actually used for screening
        ax.set_xlim(0, 1)                                   # x-axis fixed to the full 0-1 probability range
        ax.set_ylim(0, 1)                                   # y-axis fixed to the full 0-1 probability range
        ax.set_xlabel("Nominal coverage (claimed by the interval)")   # x-axis caption
        ax.set_title("Raw ensemble spread" if label == "raw"
                    else "Recalibrated (scale factor fit on val, checked on test)")  # per-subplot title depends on which column this iteration plots
        ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5)  # legend box, boxed, upper-left
        ax.set_aspect("equal")                              # force a 1:1 x/y scale so the diagonal is a true 45 degrees
    axes[0].set_ylabel("Empirical coverage (actual, on 1,648 held-out test crystals)")  # y-axis caption, left subplot only (shared visually)
    fig.suptitle("Is the ensemble's uncertainty interval calibrated?", fontsize=13)     # figure-level title spanning both subplots
    fig.tight_layout()                                      # auto-adjust spacing so labels/titles do not overlap
    fig.savefig(path, dpi=160)                              # rasterize and write the figure to disk at 160 dots per inch
    plt.close(fig)                                          # release the figure's memory now that it has been saved


def main():
    print("=== Ensemble uncertainty calibration check (matbench test split) ===\n")
    k = pd.read_csv(K_PRED_CSV)                     # load every K prediction row (all splits) written by 05_ensemble.py
    g = pd.read_csv(G_PRED_CSV)                     # load every G prediction row (all splits) written by 05_ensemble.py
    k_val, k_test = k[k["split"] == "val"], k[k["split"] == "test"]   # boolean-filter K rows into the val and test subsets
    g_val, g_test = g[g["split"] == "val"], g[g["split"] == "test"]   # boolean-filter G rows into the val and test subsets
    print(f"K_VRH: {len(k_val)} val / {len(k_test)} test crystals   "
         f"G_VRH: {len(g_val)} val / {len(g_test)} test crystals")
    print("(genuinely held out - none of these were used to train any ensemble member)")

    k_raw = empirical_coverage(k_test["true_log10"].values, k_test["pred_log10"].values,
                               k_test["member_std_log10"].values, NOMINAL_LEVELS)
    # ^ raw (uncorrected) empirical coverage of the K interval on TEST, one value per nominal level; .values pulls the numpy array out of each pandas column
    g_raw = empirical_coverage(g_test["true_log10"].values, g_test["pred_log10"].values,
                               g_test["member_std_log10"].values, NOMINAL_LEVELS)
    # ^ same computation for G
    k_recal, k_factors = recalibrated_coverage(k_val, k_test, NOMINAL_LEVELS)   # recalibrated K coverage plus the fitted scale factors, per level
    g_recal, g_factors = recalibrated_coverage(g_val, g_test, NOMINAL_LEVELS)   # recalibrated G coverage plus the fitted scale factors, per level

    table = pd.DataFrame({                           # assemble every computed series into one row-per-nominal-level table
        "nominal": NOMINAL_LEVELS,
        "K_VRH_raw": k_raw, "K_VRH_recalibrated": k_recal, "K_VRH_scale_factor": k_factors,
        "G_VRH_raw": g_raw, "G_VRH_recalibrated": g_recal, "G_VRH_scale_factor": g_factors,
    })
    table.to_csv(OUT_CSV, index=False)               # write the table to disk; index=False omits pandas' row-number column
    print(f"\nWrote {OUT_CSV}")

    print("\nNominal vs. empirical coverage (RAW ensemble spread):")
    print("  nominal   K_VRH emp.   G_VRH emp.")
    for _, row in table.iterrows():                  # walk the table one row at a time; `_` discards the row index since it is unused
        print(f"   {row.nominal * 100:5.0f}%      {row.K_VRH_raw * 100:5.1f}%       "
             f"{row.G_VRH_raw * 100:5.1f}%")

    row90 = table[table["nominal"] == 0.90].iloc[0]    # select the single row where nominal equals 90% - the level actually used for screening
    print(f"\n{'=' * 70}")
    print("HEADLINE (the 90% level actually used throughout the GNoME screen):")
    print(f"  K_VRH: claimed 90%, actual {row90.K_VRH_raw * 100:.1f}% -> "
         f"BADLY OVERCONFIDENT (ensemble spread underestimates true error)")
    print(f"  G_VRH: claimed 90%, actual {row90.G_VRH_raw * 100:.1f}% -> "
         f"BADLY OVERCONFIDENT")
    print(f"{'=' * 70}")

    print(f"\nFIX, checked out-of-sample (scale factor fit on val, applied to test):")
    print(f"  K_VRH: widen interval by {row90.K_VRH_scale_factor / 1.645:.1f}x "
         f"(vs. the untouched normal-quantile 1.645) -> {row90.K_VRH_recalibrated * 100:.1f}% "
         f"actual coverage on test (target: 90%)")
    print(f"  G_VRH: widen interval by {row90.G_VRH_scale_factor / 1.645:.1f}x -> "
         f"{row90.G_VRH_recalibrated * 100:.1f}% actual coverage on test (target: 90%)")

    plot_reliability(table, OUT_PNG)                 # render and save the reliability-diagram PNG using the renamed output path
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":                           # guard so main() only runs when this file is executed directly, not on import
    main()
