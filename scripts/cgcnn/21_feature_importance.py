#!/usr/bin/env python3
"""
STEP 21 - What drives the predicted Kappa_cal, feature by feature.
================================================================================

    python scripts/cgcnn/21_feature_importance.py

WHAT THIS IS
-------------
Trains a RandomForestRegressor to predict log10(Kappa_cal) from six
independent inputs, then reports permutation importance (primary) and
impurity-based (MDI) importance (secondary) - answering "which of the
things we actually measure or predict has the most leverage over the
screening outcome" for a general audience, without requiring anyone to
trace through the closed-form Slack-model algebra by hand.

WHY THESE SIX FEATURES, AND NOT THE OTHER COLUMNS IN THE SCREEN CSV
------------------------------------------------------------------------
Kappa_cal is not a black-box prediction - it is an exact, closed-form
function of a few quantities (see 07_predict_kappa.py's slack_physics()):

    Kappa_cal = G_VRH_pred * v_sound * V^(1/3) / (N_atoms * 300) * exp(-gamma)
    v_sound, gamma <- derived from K_VRH_pred, G_VRH_pred, Density

Poisson ratio, Gruneisen parameter and the sound velocities are
INTERMEDIATE quantities computed from K_VRH_pred/G_VRH_pred/Density inside
that same formula - including them as model features would just recover the
algebra circularly (e.g. "how important is x^2 for predicting x^2*y" has
the trivial answer "entirely"), not tell you anything new. So the feature
set here is restricted to the underlying INDEPENDENT quantities the formula
actually consumes: the two predicted moduli, plus the three structural
quantities read straight from the primitive cell (Number of Atoms,
Volume, Density) and Atomic mass, which is not part of the kappa_L formula
at all but is included since it correlates with which elements are present
(heavier elements -> lower kappa_L, this project's own halogen-enrichment
finding from 18_element_frequency.py).

Because the target is a genuine physics formula rather than noisy labelled
data, the model's job here is closer to a SENSITIVITY ANALYSIS than a
prediction task - the R^2 reported below should be very high (the six
inputs really do determine the output almost completely, modulo how well a
random forest approximates the closed-form algebra) - that is expected and
is not being claimed as a novel ML result, just a way to rank which inputs
matter most across the REAL, correlated distribution of this dataset,
which is not obvious from the formula alone (algebraically G_VRH_pred
appears twice and N_atoms appears once, but that says nothing about which
one actually varies most / matters most across 33,323 real candidates).

WHY BOTH MDI AND PERMUTATION IMPORTANCE
--------------------------------------------
Impurity-based (MDI) importance is fast but biased toward high-cardinality
continuous features (it can rank a feature as "important" just because it
offers more possible split points, not because it truly drives the
target) - a real risk here since Number of Atoms is a low-cardinality
integer while Volume/Density are continuous. Permutation importance
(shuffle one column, measure how much test R^2 drops) does not have this
bias and is reported as the primary ranking; MDI is kept alongside as a
secondary, standard cross-check.
"""

import os                                            # path joining

import matplotlib
matplotlib.use("Agg")                                # non-interactive backend - this script only saves a PNG, never shows a window
import matplotlib.pyplot as plt                      # figure/axes creation and saving
import numpy as np                                   # np.log10, np.arange
import pandas as pd                                  # CSV I/O and DataFrame column selection
from sklearn.ensemble import RandomForestRegressor   # the sensitivity-analysis model itself
from sklearn.inspection import permutation_importance  # shuffle-one-column-and-remeasure importance
from sklearn.metrics import r2_score                 # goodness-of-fit metric on the held-out test split
from sklearn.model_selection import train_test_split  # random train/test split

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 13_screen_gnome.py's full, unfiltered scored-candidate table
SCREEN_CSV = os.path.join(RESULTS, "13_gnome_screen_all.csv")
# outputs: this script's own importance table and chart, "21_" prefix (both terminal)
OUT_CSV = os.path.join(RESULTS, "21_gnome_screen_feature_importance.csv")
OUT_PNG = os.path.join(RESULTS, "21_gnome_screen_feature_importance.png")

FEATURES = ["Number of Atoms", "Volume (A3)", "Density (g cm-3)",
           "Atomic mass (amu)", "K_VRH_pred", "G_VRH_pred"]  # the six independent inputs fed to the random forest
FEATURE_LABELS = {  # human-readable names for plot/console labels, keyed by the raw column name
    "Number of Atoms": "Number of atoms",
    "Volume (A3)": "Volume",
    "Density (g cm-3)": "Density",
    "Atomic mass (amu)": "Atomic mass",
    "K_VRH_pred": "Bulk modulus K",
    "G_VRH_pred": "Shear modulus G",
}

BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({  # global matplotlib style overrides, applied to every figure created after this point
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,  # drop the top/right plot border for a cleaner look
})


def plot_importance(table, path):
    table = table.sort_values("permutation_importance_mean")  # ascending, so the barh reads most-important-at-top after plotting
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(table))  # integer y-position for each bar
    ax.barh(y, table["permutation_importance_mean"],
           xerr=table["permutation_importance_std"],  # error bars: spread across the 20 permutation repeats
           color=BLUE, alpha=0.75, error_kw=dict(ecolor=INK_SOFT, capsize=3, linewidth=1))
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in table["feature"]])  # human-readable feature names on the y-axis
    ax.set_xlabel("Permutation importance (drop in test $R^2$ when shuffled)")
    ax.set_title("What drives predicted Kappa_cal")
    fig.tight_layout()       # shrink margins so labels don't get clipped
    fig.savefig(path, dpi=160)
    plt.close(fig)           # free the figure's memory


def main():
    print("=== Feature importance for predicted Kappa_cal ===\n")
    df = pd.read_csv(SCREEN_CSV).dropna(subset=FEATURES + ["Kappa_cal (W m-1 K-1)"])  # drop rows missing any of the model's inputs or the target
    print(f"Loaded {len(df)} scored candidates with complete features from {SCREEN_CSV}")

    X = df[FEATURES].values                                    # feature matrix, shape (n_rows, 6)
    y = np.log10(df["Kappa_cal (W m-1 K-1)"].values)            # target: log10(kappa), matching this project's convention everywhere else

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 80/20 split, fixed seed for reproducibility

    model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)  # 300 trees, all CPU cores, fixed seed
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))  # how well the forest's predictions match the held-out targets
    print(f"\nRandomForest fit on log10(Kappa_cal), {len(FEATURES)} features: test R^2 = {r2:.4f}")
    print("(expected to be very high - Kappa_cal is a closed-form function of these inputs, "
         "not noisy labelled data; see module docstring)")

    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0, n_jobs=-1)  # shuffle each feature 20 times, measure the R^2 drop
    mdi = model.feature_importances_  # the forest's own built-in (impurity-based) importance, for comparison

    table = pd.DataFrame({
        "feature": FEATURES,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
        "mdi_importance": mdi,
    }).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)  # most important (by the primary ranking) first

    print("\nPermutation importance (primary ranking) vs. MDI (secondary cross-check):")
    for _, row in table.iterrows():  # one console line per feature, ranked
        print(f"  {FEATURE_LABELS[row.feature]:16s} permutation={row.permutation_importance_mean:.4f} "
             f"(+/-{row.permutation_importance_std:.4f})   MDI={row.mdi_importance:.4f}")

    table.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV}")

    plot_importance(table, OUT_PNG)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
