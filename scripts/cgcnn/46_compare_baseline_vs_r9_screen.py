#!/usr/bin/env python3
"""
STEP 46 - Where the tree baseline and the round-9 CGCNN disagree on GNoME,
and which of the two the disagreement can actually adjudicate.
================================================================================

    python scripts/cgcnn/46_compare_baseline_vs_r9_screen.py

WHY THIS SCRIPT EXISTS
------------------------
45_screen_gnome_baseline_tree.py scored all 33,118 GNoME candidates with the
composition-only tree baseline and found a large, systematic split:

    median kappa, round-9 CGCNN  2.14 W/m/K      candidates <=1:  7,271
    median kappa, tree baseline  0.94 W/m/K      candidates <=1: 17,099

That is a factor of 2.3 on the median crystal and 2.4x as many candidates
clearing the screen's threshold - far too big to leave as one line of stdout.
This script decomposes it: WHICH predicted quantity carries the difference,
whether it is uniform or concentrated, and whether anything in this project's
data can say which pipeline is right on GNoME (it cannot - and saying so
explicitly is the point, see the CAVEAT section below).

THE DECOMPOSITION
-------------------
Slack kappa is a product of a prefactor in G and the sound velocity, times
exp(-gamma). The two pipelines share EXACTLY the same structural constants
(Volume, Number of Atoms, Density are read off the same table by both), so
every difference in kappa has to come through K, G or gamma and nothing else.
Splitting log10(kappa_tree) - log10(kappa_r9) into those channels says which
head is responsible instead of leaving it as one aggregate number.

THE CAVEAT THAT GOVERNS HOW THIS IS REPORTED
-----------------------------------------------
GNoME has NO ground-truth elastic moduli - that is the entire reason it is a
screening target. So nothing here measures accuracy on GNoME; the only
accuracy either model has is on AFLOW's 835-crystal held-out test set, where
the tree wins (K 0.0592 vs 0.1255, G 0.1010 vs 0.1521, log10 MAE). That win
does NOT transfer automatically: a tree can only ever predict a weighted
average of training-leaf values, so on chemistry unlike AFLOW's it reverts
toward what it has seen, while the CGCNN can extrapolate (correctly or not).
This script therefore reports the disagreement and the ONE thing that is
decidable - how much of the existing DFT shortlist the two pipelines agree
on - and does not declare a winner.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    "scored_csv": "results/cgcnn/45_gnome_screen_all_baseline.csv",   # 45's own output, both pipelines side by side
    "dft_shortlist_csv": "/Users/mac/Desktop/dft_candidates.csv",      # the 218 crystals actually queued for DFT
    "kappa_threshold": 1.0,   # W/m/K, identical to 13/39/45
    "csv_dir": "results/cgcnn/baseline_tree/csv",   # same csv/png split 43 and 44 use
    "png_dir": "results/cgcnn/baseline_tree/png",
}
# =============================================================================

import os                 # path joining/creation
import sys                 # sys.exit() on a missing input

os.environ.setdefault("OMP_NUM_THREADS", "1")        # see 43's comment / cgcnn-pink-environment memory
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401   # torch first - MKL/OpenMP import-order guard
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend - this script never opens a window
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same palette as 44_baseline_tree_analysis.py, so the two scripts' figures
# sit together without a colour clash.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def main():
    path = os.path.join(PROJECT_ROOT, CONFIG["scored_csv"])
    if not os.path.exists(path):
        sys.exit(f"No scored table at {path} - run 45_screen_gnome_baseline_tree.py first")
    df = pd.read_csv(path, dtype={"material_id": str}, low_memory=False)

    # Keep only rows both pipelines produced a physical kappa for - a handful
    # can be NaN/non-positive from a non-physical predicted modulus, and log10
    # of those is meaningless.
    valid = df.dropna(subset=["Kappa_r9_gamma", "Kappa_baseline"])
    valid = valid[(valid["Kappa_r9_gamma"] > 0) & (valid["Kappa_baseline"] > 0)].copy()

    # ---- decompose the log10 kappa gap into its three channels --------------
    # Both pipelines use the SAME Volume/Number of Atoms/Density, so those
    # cancel exactly and every term below is a pure model difference.
    valid["d_log_kappa"] = np.log10(valid["Kappa_baseline"]) - np.log10(valid["Kappa_r9_gamma"])
    valid["d_log_K"] = np.log10(valid["K_baseline_rf"]) - np.log10(valid["K_r9_pred"])
    valid["d_log_G"] = np.log10(valid["G_baseline_rf"]) - np.log10(valid["G_r9_pred"])
    valid["d_gamma"] = valid["gamma_baseline_rf"] - valid["gamma_r9_pred"]
    # The anharmonic channel's contribution to log10(kappa) is -(d_gamma)/ln(10):
    # kappa carries exp(-gamma), so a gamma difference moves log10(kappa) by
    # -d_gamma * log10(e).
    valid["d_log_kappa_from_gamma"] = -valid["d_gamma"] * np.log10(np.e)
    valid["d_log_kappa_from_moduli"] = valid["d_log_kappa"] - valid["d_log_kappa_from_gamma"]

    rows = []
    for name, series in [
        ("log10 kappa gap (tree - r9)", valid["d_log_kappa"]),
        ("  ...from the moduli (K, G)", valid["d_log_kappa_from_moduli"]),
        ("  ...from gamma", valid["d_log_kappa_from_gamma"]),
        ("log10 K gap (tree - r9)", valid["d_log_K"]),
        ("log10 G gap (tree - r9)", valid["d_log_G"]),
        ("gamma gap (tree - r9)", valid["d_gamma"]),
    ]:
        rows.append({"quantity": name.strip(), "median": series.median(), "mean": series.mean(),
                     "sd": series.std(), "p05": series.quantile(0.05), "p95": series.quantile(0.95),
                     "median_as_factor": 10 ** series.median() if "gamma gap" not in name else np.nan})
    summary = pd.DataFrame(rows)

    print("=" * 78)
    print("  STEP 46 - decomposing the tree-vs-CGCNN disagreement on GNoME")
    print("=" * 78)
    print(f"\n{len(valid)} GNoME crystals scored by both pipelines\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nREADING: a negative log10 gap means the TREE predicts the smaller number.")
    print("Both pipelines use identical Volume/atom-count/density, so these are")
    print("pure model differences - nothing structural leaks in.")

    # ---- the one question the data CAN answer -------------------------------
    thr = CONFIG["kappa_threshold"]
    r9_low = valid["Kappa_r9_gamma"] <= thr
    tree_low = valid["Kappa_baseline"] <= thr
    confusion = pd.DataFrame({
        "tree: low": [int((r9_low & tree_low).sum()), int((~r9_low & tree_low).sum())],
        "tree: high": [int((r9_low & ~tree_low).sum()), int((~r9_low & ~tree_low).sum())],
    }, index=["r9: low", "r9: high"])
    print(f"\nAgreement on the screen decision (kappa <= {thr} W/m/K), all {len(valid)} crystals:")
    print(confusion.to_string())
    agree = int((r9_low == tree_low).sum())
    print(f"  the two pipelines make the same call on {agree} crystals ({100 * agree / len(valid):.1f}%)")

    # Recall of round-9's own candidate set is the screening-relevant number:
    # of the crystals round 9 flagged, how many would the tree also have found?
    recall = int((r9_low & tree_low).sum()) / max(1, int(r9_low.sum()))
    print(f"  of round-9's {int(r9_low.sum())} candidates, the tree also flags "
          f"{int((r9_low & tree_low).sum())} ({100 * recall:.1f}%)")

    # ---- and the actionable end of the list ----------------------------------
    if os.path.exists(CONFIG["dft_shortlist_csv"]):
        dft = pd.read_csv(CONFIG["dft_shortlist_csv"], dtype={"material_id": str})
        sub = valid[valid["material_id"].isin(set(dft["material_id"]))]
        n_low = int((sub["Kappa_baseline"] <= thr).sum())
        print(f"\nOn the {len(sub)} crystals already shortlisted for DFT, the tree baseline")
        print(f"  agrees {n_low} are low-kappa ({100 * n_low / max(1, len(sub)):.0f}%), median "
              f"kappa {sub['Kappa_baseline'].median():.3f} vs round-9's {sub['Kappa_r9_gamma'].median():.3f} W/m/K")

    # ---- figure --------------------------------------------------------------
    png_dir = os.path.join(PROJECT_ROOT, CONFIG["png_dir"])
    csv_dir = os.path.join(PROJECT_ROOT, CONFIG["csv_dir"])
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.patch.set_facecolor("white")

    # Panel 1: kappa vs kappa. A 2D histogram, not a scatter - 33k points as
    # dots is a solid blob that hides the density structure entirely.
    ax = axes[0]
    x, y = np.log10(valid["Kappa_r9_gamma"]), np.log10(valid["Kappa_baseline"])
    ax.hexbin(x, y, gridsize=60, cmap="Blues", mincnt=1, linewidths=0)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color=INK_SOFT, lw=1.2, ls="--", label="perfect agreement")
    ax.axvline(0, color=ORANGE, lw=1.2, ls=":")   # log10(1.0) = 0, the screen threshold
    ax.axhline(0, color=ORANGE, lw=1.2, ls=":")
    ax.set_xlabel("round-9 CGCNN   log$_{10}$ $\\kappa_L$ (W m$^{-1}$K$^{-1}$)", color=INK_SOFT)
    ax.set_ylabel("tree baseline   log$_{10}$ $\\kappa_L$", color=INK_SOFT)
    ax.set_title(f"33,118 GNoME candidates\nr = {np.corrcoef(x, y)[0, 1]:.3f}", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel 2: which channel carries the gap.
    ax = axes[1]
    ax.hist(valid["d_log_kappa_from_moduli"], bins=80, color=BLUE, alpha=0.85,
            label=f"moduli (K, G)   median {valid['d_log_kappa_from_moduli'].median():+.3f}")
    ax.hist(valid["d_log_kappa_from_gamma"], bins=80, color=GREEN, alpha=0.85,
            label=f"gamma   median {valid['d_log_kappa_from_gamma'].median():+.3f}")
    ax.axvline(0, color=INK_SOFT, lw=1.0)
    ax.set_xlabel("contribution to log$_{10}$($\\kappa_{tree}$/$\\kappa_{r9}$)", color=INK_SOFT)
    ax.set_ylabel("crystals", color=INK_SOFT)
    ax.set_title("The gap is entirely in the moduli\ngamma contributes ~nothing", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    # Panel 3: the screen decision itself.
    ax = axes[2]
    labels = ["both\nlow", "r9 only", "tree only", "both\nhigh"]
    values = [int((r9_low & tree_low).sum()), int((r9_low & ~tree_low).sum()),
              int((~r9_low & tree_low).sum()), int((~r9_low & ~tree_low).sum())]
    bars = ax.bar(labels, values, color=[GREEN, ORANGE, BLUE, MUTED], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("crystals", color=INK_SOFT)
    ax.set_title(f"Screen decision at $\\kappa_L\\leq${thr} W m$^{{-1}}$K$^{{-1}}$\n"
                 f"same call on {100 * agree / len(valid):.0f}%", color=INK, fontsize=11)

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    png_path = os.path.join(png_dir, "46_baseline_vs_r9_gnome_screen.png")
    fig.savefig(png_path, dpi=160, facecolor="white")
    plt.close(fig)

    summary_path = os.path.join(csv_dir, "46_baseline_vs_r9_gap_decomposition.csv")
    summary.to_csv(summary_path, index=False)
    confusion_path = os.path.join(csv_dir, "46_baseline_vs_r9_screen_confusion.csv")
    confusion.to_csv(confusion_path)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
