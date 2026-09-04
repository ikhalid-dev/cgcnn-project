#!/usr/bin/env python3
"""
STEP 44 - Correlation, confusion-matrix and parity report for the baseline
tree models (43_baseline_tree_models.py), plus a reference table for the
92-dim atom_init.json embedding those tree models are built on.
================================================================================

    python scripts/cgcnn/44_baseline_tree_analysis.py

WHY THIS SCRIPT EXISTS
------------------------
43_baseline_tree_models.py answers "how accurate is composition alone" with
one number per (source, target, model): MAE. That number hides three things
this script makes visible:

  1. CORRELATION - do the three tree models (and the CGCNN they're being
     compared against) actually agree with each other and with the truth, or
     does a similar MAE hide very different per-crystal predictions? Also:
     are the 184 input features themselves independent, or is the
     composition-weighted-mean/std encoding highly redundant (it should be -
     see the atom reference below)?
  2. CONFUSION MATRIX - this project's actual use case is not "predict K/G
     accurately", it's "flag low-kappa candidates for a screen" (see
     scripts/cgcnn/13_screen_gnome.py). A model can have a mediocre MAE and
     still be a fine screening tool, or a good MAE and still miss the
     candidates that matter (see [[cgcnn-training-status]]'s Q1-quintile
     finding in the kappa-ml-experiments skill: MAE improvements historically
     have NOT moved recall on the bottom decile). This turns the AFLOW tree
     baseline's K/G predictions into the same low-kappa/not-low-kappa call
     the real screen makes, via the identical slack_physics() used
     everywhere else in this project, and checks it against AFLOW's own
     real kappa_agl label (not another model's prediction - genuine ground
     truth, the same column 43_baseline_tree_models.py trained "gamma"
     against).
  3. THE ATOM FEATURE TABLE ITSELF - cgcnn_scratch/atom_init.json's 92
     numbers per element have never had a reference in this project beyond
     one docstring line ("group, period, electronegativity, ..."). Verified
     here computationally (not just asserted) against the standard published
     CGCNN/Magpie-style block layout: 9 named property blocks, mostly
     one-hot within each block (a handful of elements are 0 in a block when
     that property is undefined for them, e.g. noble gases and
     electronegativity - see the verification note in atom_feature_reference()).

WHY PARITY PLOTS TOO
------------------------
The project's own convention (03_evaluate.py's plot_parity, mirrored for
ALIGNN in scripts/alignn/17_alignn_diagnostics.py) is a log-log true-vs-
predicted scatter with a 1.5x tolerance band. Reusing that convention here
- one figure per (source, target), all three tree models overlaid - makes
this baseline directly visually comparable to every other model's figures
in results/cgcnn and results/alignn, not just comparable by a table of MAE
numbers.

OUTPUT LAYOUT
------------------------
results/cgcnn/baseline_tree/csv/   - every table this script writes
results/cgcnn/baseline_tree/png/   - every figure this script writes
results/cgcnn/atom_init_feature_reference.csv  - project-wide reference, not
  baseline-tree-specific, so it lives one level up from the baseline_tree/
  folder (same place 43's own scores/predictions CSVs used to live).
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    "predictions_csv": "results/cgcnn/baseline_tree/csv/43_baseline_tree_predictions.csv",
    "aflow_labels_csv": "data_full/gamma_labels.csv",
    "matbench_labels_csv": "data_full/labels.csv",
    "matbench_graphs_pt": "data_full/graphs.pt",
    "aflow_graphs_pt": "data_full/gamma_graphs.pt",
    "atom_init_json": "cgcnn_scratch/atom_init.json",

    "csv_out_dir": "results/cgcnn/baseline_tree/csv",
    "png_out_dir": "results/cgcnn/baseline_tree/png",
    "atom_reference_csv": "results/cgcnn/atom_init_feature_reference.csv",

    # low-kappa screening threshold - identical to 13_screen_gnome.py's own
    # cutoff, so the confusion matrix answers the same question that script
    # asks of the real GNoME screen.
    "kappa_threshold_w_mk": 1.0,
    # which of the three tree models feeds the confusion matrix - xgboost has
    # the best AFLOW K_VRH test MAE (see 43's own printed comparison table);
    # using one consistent model for both K and G matters more here than
    # picking G's own marginally-better random_forest, since the confusion
    # matrix is meant to show one model's real screening behaviour, not the
    # best cell of a table.
    "kappa_model": "xgboost",

    "models": ["decision_tree", "random_forest", "xgboost"],
}
# =============================================================================

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")   # see 43_baseline_tree_models.py - avoids the same xgboost segfault, harmless here since this script doesn't refit anything but keeps the two scripts' env identical
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401   # torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from scipy.constants import h, k
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ---- reuse the project's own plot palette (03_evaluate.py) -----------------
BLUE = "#2a78d6"      # decision_tree
ORANGE = "#eb6834"    # random_forest
GREEN = "#3f9142"     # xgboost - this project's established 3rd categorical colour (10_validate_table1.py, 14_compare_gnome_screen.py)
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
MODEL_COLOR = {"decision_tree": BLUE, "random_forest": ORANGE, "xgboost": GREEN}


def out_paths():
    csv_dir = os.path.join(PROJECT_ROOT, CONFIG["csv_out_dir"])
    png_dir = os.path.join(PROJECT_ROOT, CONFIG["png_out_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    return csv_dir, png_dir


# =============================================================================
#  1. Atom feature reference - what the 92 dims of atom_init.json actually are
# =============================================================================

# The 9 property blocks of atom_init.json's 92-dim vector, each one-hot binned.
#
# These widths were DERIVED FROM THE FILE, not copied from the CGCNN paper -
# and the two differ, so do not "correct" them back. Deriving them is a
# 5-line greedy scan: extend a block while every element still has at most one
# active bit in it, and the boundaries fall out uniquely. Each block's meaning
# was then confirmed by looking at which elements light up which column:
#
#   group_number     19 wide, NOT 18. Column 0 is the 26 lanthanides+actinides
#                    (no standard group number); columns 1-18 are groups 1-18
#                    (col 1 = H/Li/Na/K/Rb/Cs/Fr, col 17 = the halogens,
#                    col 18 = the noble gases). Assuming 18 here silently
#                    shifts every later block by one.
#   period            7 wide, NOT 8 - periods 1-7, exactly as the periodic
#                    table has (col 19 = H,He; col 22 = the 18 period-4
#                    elements; col 24 = the 32 period-6 elements).
#   electronegativity ordered low -> high (col 26 = K,Rb,Cs,Fr at ~0.8;
#                    col 35 = F alone at 3.98).
#   block_spdf        s, p, d, f in that order (col 78 = H,He,Li,Be,...;
#                    col 79 = B,C,N,O,F,...; col 80 = Sc,Ti,V,...;
#                    col 81 = the f-block).
#
# Three blocks have elements with NO bit set anywhere in the block - that is
# the source table having no value for that property for that element (15 for
# electronegativity, 26 for electron affinity, 9 for atomic volume), not a
# layout error. The reference CSV records that count per block.
ATOM_FEATURE_BLOCKS = [
    ("group_number", 19),
    ("period", 7),
    ("electronegativity", 10),
    ("covalent_radius", 10),
    ("valence_electrons", 12),
    ("first_ionization_energy", 10),
    ("electron_affinity", 10),
    ("block_spdf", 4),
    ("atomic_volume", 10),
]


def atom_feature_reference(ari_path, csv_dir):
    """One row per dimension of atom_init.json's 92-long vector: which named
    property it belongs to, and its position within that property's one-hot
    block. Also verifies the block layout against the real file (prints a
    warning rather than silently trusting the hardcoded table above)."""
    import json
    from pymatgen.core import Element
    d = json.load(open(ari_path))
    z_order = sorted(d, key=int)
    arr = np.array([d[key] for key in z_order])   # (100 elements, 92 dims)
    symbols = [Element.from_Z(int(z)).symbol for z in z_order]
    assert arr.shape[1] == 92, f"atom_init.json has {arr.shape[1]} dims, expected 92 - block layout below is stale"

    rows = []
    start = 0
    for block_name, width in ATOM_FEATURE_BLOCKS:
        seg = arr[:, start:start + width]
        ones_per_row = seg.sum(axis=1)
        off_one_hot = int(((ones_per_row != 1)).sum())   # elements where this block isn't exactly one 1 (0 = property undefined for that element; >1 would be a real layout error)
        if (ones_per_row > 1).any():
            print(f"  WARNING: block '{block_name}' has a row with >1 active bit - "
                  f"block boundaries in ATOM_FEATURE_BLOCKS may be off by one")
        for i in range(width):
            # Which elements actually light up this bin - the only self-documenting
            # thing available, since atom_init.json stores no bin edges, only bits.
            members = [symbols[r] for r in np.where(arr[:, start + i] == 1)[0]]
            rows.append({
                "feature_index": start + i,
                "property_block": block_name,
                "index_within_block": i,
                "block_width": width,
                "n_elements_in_bin": len(members),
                "example_elements": ", ".join(members[:6]) + ("..." if len(members) > 6 else ""),
                "elements_with_property_undefined": off_one_hot,   # same value repeated per row in the block, for convenience
            })
        start += width

    ref = pd.DataFrame(rows)
    path = os.path.join(PROJECT_ROOT, CONFIG["atom_reference_csv"])
    ref.to_csv(path, index=False)
    print(f"Wrote {path} ({len(ref)} rows, {len(ATOM_FEATURE_BLOCKS)} property blocks)")
    return ref


# =============================================================================
#  2. Correlation: how much do the tree models actually agree, per target
# =============================================================================

def model_correlation(preds, csv_dir, png_dir):
    """Pairwise Pearson r among [true, decision_tree, random_forest, xgboost]
    on the TEST split, one matrix per (source, target). This is the same
    "do independent models agree" check the rest of this project already
    uses for CGCNN-vs-ALIGNN and CGCNN-vs-paper (see [[cgcnn-training-status]]
    Phase 2b) - here applied to three tree variants against each other and
    against the truth."""
    summary_rows = []
    targets_by_source = {"matbench": ["K_VRH", "G_VRH"], "aflow": ["K_VRH", "G_VRH", "gamma"]}

    for source, targets in targets_by_source.items():
        sub = preds[(preds.source == source) & (preds.split == "test")]
        for target in targets:
            cols = {"true": sub[f"{target}_true_log10"].values}
            for model in CONFIG["models"]:
                cols[model] = sub[f"{target}_pred_log10_{model}"].values
            mat = pd.DataFrame(cols)
            corr = mat.corr(method="pearson")

            csv_path = os.path.join(csv_dir, f"44_correlation_matrix_{source}_{target}.csv")
            corr.to_csv(csv_path)

            for row_name in corr.index:
                for col_name in corr.columns:
                    if row_name == "true":
                        summary_rows.append({"source": source, "target": target,
                                             "against": col_name, "pearson_r": corr.loc[row_name, col_name]})

            _plot_correlation_heatmap(corr, f"{source} / {target} - test split", os.path.join(png_dir, f"44_correlation_heatmap_{source}_{target}.png"))

    summary = pd.DataFrame(summary_rows)
    summary = summary[summary.against != "true"]   # drop the trivial true-vs-true row
    summary_path = os.path.join(csv_dir, "44_correlation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} rows) and one heatmap per source/target")
    return summary


def _plot_correlation_heatmap(corr, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(corr.values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=9, color=INK_SOFT)
    ax.set_yticklabels(corr.index, fontsize=9, color=INK_SOFT)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=INK if val < 0.65 else "white")
    ax.set_title(title, fontsize=11, pad=10, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
#  3. Feature correlation: are the 184 composition-encoded dims redundant?
# =============================================================================

def feature_correlation(csv_dir, png_dir, ari):
    """Full 184x184 Pearson correlation of the composition-weighted-mean/std
    feature vector across every crystal (matbench + AFLOW combined - the
    features live in the same 92-dim atom_init.json space regardless of
    source). Dense by design: the point is to SEE the 9 property blocks as
    correlated clusters (visible as block-diagonal structure in the heatmap),
    which is the actual answer to "are these features doing 184 independent
    things or fewer" - a summary number would hide exactly that structure.
    """
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline43", os.path.join(PROJECT_ROOT, "scripts", "cgcnn", "43_baseline_tree_models.py"))
    b43 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b43)

    X_mb, _ = b43.build_feature_table(CONFIG["matbench_labels_csv"], CONFIG["matbench_graphs_pt"], "mb_id", ari)
    X_af, _ = b43.build_feature_table(CONFIG["aflow_labels_csv"], CONFIG["aflow_graphs_pt"], "gid", ari)
    X = np.concatenate([X_mb, X_af], axis=0)   # (n_crystals, 184) - both sources share the same 184-dim space

    col_names = [f"mean_{name}_{i}" for name, w in ATOM_FEATURE_BLOCKS for i in range(w)] + \
        [f"std_{name}_{i}" for name, w in ATOM_FEATURE_BLOCKS for i in range(w)]
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)   # a handful of columns are constant (all-zero, see 43's own note) - corrcoef gives NaN there, not a real correlation

    csv_path = os.path.join(csv_dir, "44_feature_correlation_184dim.csv")
    pd.DataFrame(corr, index=col_names, columns=col_names).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    # block boundary gridlines - mean block (0-92) then std block (92-184), 9 named sub-blocks in each
    boundaries = []
    pos = 0
    for _, w in ATOM_FEATURE_BLOCKS:
        pos += w
        boundaries.append(pos)
    for b in boundaries:
        ax.axvline(b - 0.5, color=MUTED, linewidth=0.4, alpha=0.6)
        ax.axhline(b - 0.5, color=MUTED, linewidth=0.4, alpha=0.6)
        ax.axvline(b - 0.5 + 92, color=MUTED, linewidth=0.4, alpha=0.6)
        ax.axhline(b - 0.5 + 92, color=MUTED, linewidth=0.4, alpha=0.6)
    ax.axvline(91.5, color=INK_SOFT, linewidth=1.0)   # mean/std boundary, drawn heavier
    ax.axhline(91.5, color=INK_SOFT, linewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Feature correlation, 184-dim composition encoding\n"
                "first half: weighted mean, second half: weighted std\n"
                "thin lines = the 9 property blocks, thick line = mean | std",
                fontsize=10, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    png_path = os.path.join(png_dir, "44_feature_correlation_184dim.png")
    fig.savefig(png_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {csv_path} and {png_path}")


# =============================================================================
#  4. Confusion matrix: low-kappa screening call, tree baseline vs real AFLOW label
# =============================================================================

def slack_physics(k_gpa, g_gpa, volume_a3, density, mass_amu, n_atoms):
    """Copied from scripts/cgcnn/07_predict_kappa.py rather than imported -
    that module's top-level argparse setup and self_test() are not needed
    here, and importing it would pull in its CLI argument parser. Formulas
    are byte-identical; see that file for the physics derivation comments."""
    v_long = ((k_gpa + 4 * g_gpa / 3) / density) ** 0.5 * 1000
    v_trans = (g_gpa / density) ** 0.5 * 1000
    v_sound = ((1 / v_long ** 3 + 2 / v_trans ** 3) / 3) ** (-1 / 3)
    debye_temp = (h / k * np.power(3 / (4 * np.pi * volume_a3), 1 / 3) * v_sound * 1e10)
    ratio = v_long / v_trans
    poisson = (ratio ** 2 - 2) / (2 * ratio ** 2 - 2)
    gruneisen = 3 * (1 + poisson) / (2 * (2 - 3 * poisson))
    kappa_cal = (g_gpa * 1e9 * v_sound * (volume_a3 * 1e-30) ** (1 / 3)
                / (n_atoms * 300) * np.exp(-gruneisen))
    return kappa_cal


def confusion_matrix_report(preds, csv_dir, png_dir):
    """Turns the AFLOW tree baseline's K/G predictions into the same
    low-kappa/not-low-kappa call 13_screen_gnome.py makes for real, and
    checks it against AFLOW's own true kappa_agl - genuine DFT-derived
    ground truth (AGL method), not another model's prediction."""
    model = CONFIG["kappa_model"]
    thr = CONFIG["kappa_threshold_w_mk"]

    sub = preds[(preds.source == "aflow") & (preds.split == "test")].copy()
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["aflow_labels_csv"])).set_index("gid")
    sub = sub.set_index("material_id").join(labels[["n_sites", "density_g_cm3", "volume_m3", "formula"]], rsuffix="_lbl")

    mass_amu = np.array([Composition(f).weight for f in sub["formula"]])
    k_pred = 10 ** sub[f"K_VRH_pred_log10_{model}"].values
    g_pred = 10 ** sub[f"G_VRH_pred_log10_{model}"].values
    volume_a3 = sub["volume_m3"].values * 1e30
    density = sub["density_g_cm3"].values
    n_atoms = sub["n_sites"].values

    kappa_pred = slack_physics(k_pred, g_pred, volume_a3, density, mass_amu, n_atoms)
    kappa_true = labels.loc[sub.index, "kappa_agl"].values   # AFLOW's own AGL-computed kappa - real DFT-derived ground truth, not a model prediction

    valid = np.isfinite(kappa_pred) & np.isfinite(kappa_true) & (kappa_pred > 0)
    n_dropped = int((~valid).sum())
    kappa_pred, kappa_true = kappa_pred[valid], kappa_true[valid]

    true_label = (kappa_true <= thr).astype(int)   # 1 = low-kappa candidate
    pred_label = (kappa_pred <= thr).astype(int)

    cm = confusion_matrix(true_label, pred_label, labels=[1, 0])   # positive class (low-kappa) first, matches how a screen reads "did we flag it"
    cm_df = pd.DataFrame(cm, index=["true: low-kappa", "true: not low-kappa"],
                         columns=["pred: low-kappa", "pred: not low-kappa"])
    csv_path = os.path.join(csv_dir, "44_confusion_matrix_aflow_low_kappa.csv")
    cm_df.to_csv(csv_path)

    precision = precision_score(true_label, pred_label, zero_division=0)
    recall = recall_score(true_label, pred_label, zero_division=0)
    f1 = f1_score(true_label, pred_label, zero_division=0)
    metrics_path = os.path.join(csv_dir, "44_confusion_matrix_aflow_low_kappa_metrics.csv")
    pd.DataFrame([{
        "model": model, "kappa_threshold_w_mk": thr, "n_test_crystals": len(true_label),
        "n_dropped_non_finite": n_dropped, "n_true_low_kappa": int(true_label.sum()),
        "n_pred_low_kappa": int(pred_label.sum()),
        "precision": precision, "recall": recall, "f1": f1,
    }]).to_csv(metrics_path, index=False)
    print(f"Wrote {csv_path} and {metrics_path}")
    print(f"  AFLOW test, kappa<={thr} screen: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
         f"(n={len(true_label)}, {n_dropped} dropped as non-finite)")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["low-kappa", "not low-kappa"], fontsize=9, color=INK_SOFT)
    ax.set_yticklabels(["low-kappa", "not low-kappa"], fontsize=9, color=INK_SOFT)
    ax.set_xlabel(f"{model} predicted (via slack physics)", fontsize=9, color=INK_SOFT)
    ax.set_ylabel("true (AFLOW kappa_agl)", fontsize=9, color=INK_SOFT)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > vmax * 0.5 else INK)
    ax.set_title(f"Low-kappa screen (<= {thr} W/m/K)\nAFLOW test set\n"
                f"precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}", fontsize=10, color=INK)
    fig.tight_layout()
    png_path = os.path.join(png_dir, "44_confusion_matrix_aflow_low_kappa.png")
    fig.savefig(png_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
#  5. Parity plots: true vs predicted, all three tree models overlaid
# =============================================================================

def parity_plots(preds, png_dir):
    targets_by_source = {"matbench": ["K_VRH", "G_VRH"], "aflow": ["K_VRH", "G_VRH", "gamma"]}
    for source, targets in targets_by_source.items():
        for target in targets:
            sub = preds[preds.source == source]
            train_val = sub[sub.split != "test"]
            test = sub[sub.split == "test"]

            fig, ax = plt.subplots(figsize=(6, 6))
            true_all = 10 ** sub[f"{target}_true_log10"].values
            lo = true_all.min() * 0.7
            hi = true_all.max() * 1.3
            ax.fill_between([lo, hi], [lo / 1.5, hi / 1.5], [lo * 1.5, hi * 1.5],
                            color=BLUE, alpha=0.06, linewidth=0, zorder=0)
            ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)

            for model in CONFIG["models"]:
                y_true = 10 ** test[f"{target}_true_log10"].values
                y_pred = 10 ** test[f"{target}_pred_log10_{model}"].values
                ax.scatter(y_true, y_pred, color=MODEL_COLOR[model], s=14, alpha=0.55,
                          linewidth=0, zorder=3, label=model)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            unit = "GPa" if target != "gamma" else "(dimensionless)"
            ax.set_xlabel(f"true {target} {unit}", color=INK_SOFT)
            ax.set_ylabel(f"tree-baseline predicted {target} {unit}", color=INK_SOFT)
            ax.set_title(f"{source} / {target} - composition-only baselines, test split", fontsize=11, pad=10, color=INK)
            ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SOFT)
            ax.grid(True, which="major", linewidth=0.6, alpha=0.7, color=GRID)
            ax.set_axisbelow(True)
            fig.tight_layout()
            path = os.path.join(png_dir, f"44_parity_{source}_{target}.png")
            fig.savefig(path, dpi=160, facecolor=SURFACE)
            plt.close(fig)
    print(f"Wrote 5 parity plots to {png_dir}")


def main():
    print("=" * 78)
    print("  STEP 44 - baseline tree correlation / confusion-matrix / parity report")
    print("=" * 78)

    csv_dir, png_dir = out_paths()
    preds = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["predictions_csv"]))

    print("\n[1/5] Atom feature reference (92-dim atom_init.json)...")
    from cgcnn_scratch.data import AtomFeaturiser
    ari_path = os.path.join(PROJECT_ROOT, CONFIG["atom_init_json"])
    atom_feature_reference(ari_path, csv_dir)
    ari = AtomFeaturiser(ari_path)

    print("\n[2/5] Model-vs-model-vs-truth correlation...")
    model_correlation(preds, csv_dir, png_dir)

    print("\n[3/5] 184-dim feature correlation...")
    feature_correlation(csv_dir, png_dir, ari)

    print("\n[4/5] Confusion matrix - low-kappa screening call (AFLOW test)...")
    confusion_matrix_report(preds, csv_dir, png_dir)

    print("\n[5/5] Parity plots...")
    parity_plots(preds, png_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
