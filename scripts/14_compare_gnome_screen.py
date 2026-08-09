#!/usr/bin/env python3
"""
STEP 14 - Compare our GNoME screen against the paper's own published candidates.
======================================================================================

    python scripts/14_compare_gnome_screen.py

WHAT THIS COMPARES
-------------------
scripts/13_screen_gnome.py produced our own low-kappa_L candidate list from a
current GNoME snapshot. The paper published its own screening result -
external/AI4Kappa/JMI_Supporting_Information/Nature-filtered-low-kappa.csv,
11,869 candidates from their (older, smaller) snapshot. This script compares
the two: candidate-set overlap by reduced formula, kappa_L distribution
shape, and - the part a point-estimate pipeline like the paper's cannot do
at all - which of our OVERLAPPING candidates carry a wide Monte Carlo
interval and would be worth a DFT check before trusting, versus which are
confidently low-kappa.

WHY OVERLAP IS CHECKED BY FORMULA, NOT BY ID
------------------------------------------------
The paper's published CSV has no material ID column at all - not GNoME's own
ID, not an MP-id, nothing except the reduced formula (and even that has a
handful of duplicates: 3 of 11,869 rows repeat a formula). So "overlap" here
means "the same reduced formula appears in both candidate lists," which is
the only comparison the paper's own release actually supports - not proof
that the exact same specific structure/polymorph was matched, which would
need shared IDs neither list provides.

SNAPSHOT DRIFT, AGAIN
-----------------------
Our screen ran on 554,054 GNoME materials; the paper's ran on 377,221. Some
of what does NOT overlap is not disagreement - it is the current snapshot
containing structures the paper's screen never had the chance to consider at
all, and vice versa (structures in their older snapshot that have since been
superseded/merged in the live database, the same class of drift already
documented for Table 1's mp-3490/GaP).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
PAPER_CSV = os.path.join(PROJECT_ROOT, "external", "AI4Kappa",
                        "JMI_Supporting_Information", "Nature-filtered-low-kappa.csv")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3f9142"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_kappa_distributions(ours, paper, threshold, path):
    """Both candidate lists' kappa_L distributions, overlaid."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bins = np.linspace(0, threshold, 41)

    ax.hist(paper["Kappa_cal (W m-1 K-1)"], bins=bins, color=ORANGE, alpha=0.45,
           label=f"PINK's published candidates (n={len(paper)})", density=True,
           edgecolor="none")
    ax.hist(ours["Kappa_cal (W m-1 K-1)"], bins=bins, color=BLUE, alpha=0.45,
           label=f"Ours (n={len(ours)})", density=True, edgecolor="none")

    ax.set_xlabel("Predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("Density")
    ax.set_title("Low-$\\kappa_L$ candidate distribution shape: ours vs. the paper's",
                fontsize=12, pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_overlap_confidence(overlap_df, path):
    """Of the candidates BOTH screens flag, which does our own uncertainty
    estimate trust and which does it flag as needing a DFT check? Something
    the paper's point-estimate pipeline has no equivalent of at all.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    width = overlap_df["Kappa_cal_p95"] / overlap_df["Kappa_cal_p05"].clip(lower=1e-6)
    order = np.argsort(width.values)
    x = np.arange(len(overlap_df))

    colors = [GREEN if w <= 2 else (MUTED if w <= 5 else ORANGE) for w in width.values[order]]
    ax.bar(x, width.values[order], color=colors, width=1.0, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel(f"Overlapping candidates (n={len(overlap_df)}), sorted by interval width")
    ax.set_ylabel("Monte Carlo interval width (p95 / p05 ratio)")
    ax.set_title("Which shared candidates do we actually trust?", fontsize=12, pad=12)
    ax.axhline(2, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(5, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(0.02, 0.97, "green: tight (<2x) - confidently low-$\\kappa$\n"
                       "grey: moderate (2-5x)\n"
                       "orange: wide (>5x) - worth a DFT check before trusting",
           transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK_SOFT,
           bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8))
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", default=os.path.join(RESULTS, "gnome_screen_candidates.csv"))
    parser.add_argument("--paper", default=PAPER_CSV)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--out-dist-fig", default=os.path.join(RESULTS, "gnome_kappa_distributions.png"))
    parser.add_argument("--out-conf-fig", default=os.path.join(RESULTS, "gnome_overlap_confidence.png"))
    parser.add_argument("--out-csv", default=os.path.join(RESULTS, "gnome_overlap.csv"))
    args = parser.parse_args()

    print("=== Phase 3: comparing our GNoME screen against the paper's published candidates ===\n")
    ours = pd.read_csv(args.ours)
    paper = pd.read_csv(args.paper)
    print(f"Ours: {len(ours)} candidates (kappa_L <= {args.threshold})")
    print(f"Paper's published candidates: {len(paper)}")

    ours_formulas = set(ours.formula)
    paper_formulas = set(paper["Reduced Formula"])
    overlap_formulas = ours_formulas & paper_formulas

    print(f"\nOverlap by reduced formula: {len(overlap_formulas)} formulas appear in both lists")
    print(f"  {len(overlap_formulas) / len(ours_formulas) * 100:.1f}% of our unique candidate "
         f"formulas also appear in the paper's list")
    print(f"  {len(overlap_formulas) / len(paper_formulas) * 100:.1f}% of the paper's unique "
         f"candidate formulas also appear in ours")

    overlap_df = ours[ours.formula.isin(overlap_formulas)].copy()
    overlap_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv} ({len(overlap_df)} of our rows whose formula "
         f"also appears in the paper's list)")

    width = overlap_df["Kappa_cal_p95"] / overlap_df["Kappa_cal_p05"].clip(lower=1e-6)
    n_tight = int((width <= 2).sum())
    n_moderate = int(((width > 2) & (width <= 5)).sum())
    n_wide = int((width > 5).sum())
    print(f"\nOf the {len(overlap_df)} overlapping candidates, by our own Monte Carlo "
         f"interval width:")
    print(f"  {n_tight} confidently low-kappa (<2x p95/p05)")
    print(f"  {n_moderate} moderate confidence (2-5x)")
    print(f"  {n_wide} wide interval (>5x) - worth a DFT check before trusting, "
         f"something the paper's own point-estimate pipeline cannot flag at all")

    plot_kappa_distributions(ours, paper, args.threshold, args.out_dist_fig)
    plot_overlap_confidence(overlap_df, args.out_conf_fig)
    print(f"\nWrote {args.out_dist_fig}")
    print(f"Wrote {args.out_conf_fig}")


if __name__ == "__main__":
    main()
