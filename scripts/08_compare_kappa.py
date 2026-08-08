#!/usr/bin/env python3
"""
STEP 8 - Validate our kappa_L against the paper's own pipeline.
=================================================================

    python scripts/08_compare_kappa.py

There is no ground-truth kappa_L to check against - real lattice thermal
conductivity measurements are rare and expensive, which is the entire reason
PINK predicts it instead of measuring it. So the only honest check available
is the one the README already promised: compare our from-scratch model's
predictions against the PAPER's OWN pre-trained model, run through the
identical Slack-model physics, on the identical 1,213 crystals.

Inputs (both produced earlier in the pipeline):
    results/pink_kappa_predictions.csv   ours   (scripts/07_predict_kappa.py)
    results/pink_reference_kappa.csv     paper  (pink_predict.py, its own weights)

WHY RANK CORRELATION MATTERS AS MUCH AS THE ABSOLUTE VALUE
------------------------------------------------------------
PINK's actual use case is SCREENING: given 1,213 (or 377,221) candidates, find
the ones with the lowest kappa_L. For that job, getting the ORDER right matters
more than matching the paper's number exactly - two models that agree on "these
are the 20 softest candidates" but disagree on the third decimal place are
equally useful for screening. So this script reports Spearman rank correlation
alongside the usual log-space MAE and Pearson correlation, and neither
substitutes for the other.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")

# Same palette as scripts/03_evaluate.py, so this figure sits comfortably next
# to the parity_*.png plots already in results/.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def load_and_merge(ours_path, reference_path):
    """Join our predictions to the paper's reference run, on material_id.

    An inner join: only crystals present in BOTH runs are compared. The two
    pipelines can drop different crystals (a CIF that fails our featuriser
    might parse fine for the paper's, or vice versa), so this is not
    necessarily all 1,213 on either side - the row count printed makes that
    explicit rather than silently comparing mismatched sets.
    """
    ours = pd.read_csv(ours_path)
    reference = pd.read_csv(reference_path).rename(columns={"Material": "material_id"})
    merged = ours.merge(reference, on="material_id", suffixes=("_ours", "_ref"))
    return merged


def summarise(merged):
    """Pearson (log space) and Spearman rank correlation, plus log-space MAE.

    Everything is computed in log10 space because kappa_L spans several
    orders of magnitude - a Pearson correlation on raw values would be
    dominated by the handful of stiffest, highest-kappa crystals and say
    nothing about agreement in the ultralow regime PINK actually cares about.
    """
    log_ours = np.log10(merged["Kappa_cal (W m-1 K-1)_ours"])
    log_ref = np.log10(merged["Kappa_cal (W m-1 K-1)_ref"])

    pearson_r, _ = stats.pearsonr(log_ours, log_ref)
    spearman_r, _ = stats.spearmanr(merged["Kappa_cal (W m-1 K-1)_ours"],
                                    merged["Kappa_cal (W m-1 K-1)_ref"])
    mae_log = (log_ours - log_ref).abs().mean()

    return {"n": len(merged), "pearson_r_log": pearson_r,
           "spearman_r": spearman_r, "mae_log10": mae_log,
           "rel_error_pct": (10 ** mae_log - 1) * 100}


def screening_overlap(merged, top_n=20):
    """Of the top-N lowest-kappa crystals by each model, how many agree?

    The number that matters for screening: if two models were run
    independently and still nominate mostly the same ultralow-kappa
    candidates, that agreement is the real evidence the screen is picking up
    a physical signal rather than model-specific noise.
    """
    ours_top = set(merged.nsmallest(top_n, "Kappa_cal (W m-1 K-1)_ours").material_id)
    ref_top = set(merged.nsmallest(top_n, "Kappa_cal (W m-1 K-1)_ref").material_id)
    return len(ours_top & ref_top)


def uncertainty_calibration(merged):
    """Does our reported uncertainty predict disagreement with the reference?

    We have no ground truth for kappa_L, so we cannot check calibration the
    usual way (does the true value fall in the interval X% of the time).
    What we CAN check: a crystal our ensemble is unsure about (wide
    Kappa_cal_p05-p95 interval) should, on average, be a crystal where we
    disagree MORE with an independently-trained reference model - if the
    uncertainty estimate is measuring something real rather than being
    decorative. This is necessarily a weaker check than true calibration, but
    it is the only one available, and a strong positive correlation here is
    good evidence the ensemble spread is doing its job.
    """
    log_width = np.log10(merged["Kappa_cal_p95"] / merged["Kappa_cal_p05"])
    disagreement = (np.log10(merged["Kappa_cal (W m-1 K-1)_ours"]) -
                    np.log10(merged["Kappa_cal (W m-1 K-1)_ref"])).abs()
    r, _ = stats.spearmanr(log_width, disagreement)
    return r


def plot_parity(merged, stats_summary, path):
    """Our kappa_cal vs the paper's, log-log, in the house parity-plot style."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    x = merged["Kappa_cal (W m-1 K-1)_ref"]
    y = merged["Kappa_cal (W m-1 K-1)_ours"]
    lo = min(x.min(), y.min()) * 0.6
    hi = max(x.max(), y.max()) * 1.4

    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                    color=BLUE, alpha=0.07, linewidth=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)

    n = len(merged)
    size, alpha = (30, 0.75) if n <= 300 else (12, 0.35)
    ax.scatter(x, y, s=size, color=BLUE, alpha=alpha, linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("$\\kappa_L$ from the paper's pre-trained CGCNN (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("$\\kappa_L$ from our from-scratch CGCNN (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Stage 2 validation: same physics, two independently-trained models",
                fontsize=12, pad=12)

    ax.text(0.03, 0.97,
           f"n = {stats_summary['n']}\n"
           f"Pearson $r$ (log) = {stats_summary['pearson_r_log']:.3f}\n"
           f"Spearman $\\rho$ = {stats_summary['spearman_r']:.3f}\n"
           f"MAE = {stats_summary['mae_log10']:.3f} log$_{{10}}$",
           transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK_SOFT,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID,
                    linewidth=0.8))

    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", default=os.path.join(RESULTS, "pink_kappa_predictions.csv"))
    parser.add_argument("--reference", default=os.path.join(RESULTS, "pink_reference_kappa.csv"))
    parser.add_argument("--out-fig", default=os.path.join(RESULTS, "kappa_validation.png"))
    parser.add_argument("--out-csv", default=os.path.join(RESULTS, "kappa_comparison.csv"))
    args = parser.parse_args()

    print("=== Stage 2 validation: our kappa_L vs the paper's own pipeline ===\n")
    merged = load_and_merge(args.ours, args.reference)
    print(f"Compared on {len(merged)} crystals present in both runs "
         f"(of 1,213 attempted by each)")

    summary = summarise(merged)
    print(f"\nPearson r (log10 space)  : {summary['pearson_r_log']:.4f}")
    print(f"Spearman rank correlation: {summary['spearman_r']:.4f}")
    print(f"MAE (log10 W/m/K)        : {summary['mae_log10']:.4f}  "
         f"({summary['rel_error_pct']:.1f}% typical multiplicative error)")

    overlap = screening_overlap(merged, top_n=20)
    print(f"\nOverlap in the 20 lowest-kappa_L candidates: {overlap}/20 "
         f"nominated by both models")

    calibration_r = uncertainty_calibration(merged)
    print(f"\nUncertainty calibration check: Spearman r = {calibration_r:.3f}"
         f" between our MC interval width and disagreement with the reference")
    print("  (no ground-truth kappa_L exists to check calibration directly - "
         "this is the closest available proxy: crystals we flag as "
         "uncertain should be crystals we disagree with the reference "
         "model about more, on average)")

    plot_parity(merged, summary, args.out_fig)
    merged.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_fig}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
