#!/usr/bin/env python3
"""
STEP 54 - The tree control for the matbench subset size test.
================================================================================

    python scripts/cgcnn/54_matbench_subset_trees.py

WHY THIS EXISTS
-----------------
The Kaggle run behind step 55 trains the CGCNN on a random 3,894-crystal subset
of matbench - AFLOW's exact training-set size - to ask whether the CGCNN's loss
to a composition-only tree on AFLOW is caused by DATA QUANTITY or by the AFLOW
DATA ITSELF.

That question is only answerable if the TREE is measured at the same two sizes.
Comparing a 3,894-trained CGCNN against a 7,691-trained tree would confound the
very variable under test. This script supplies the missing half, and it needs no
GPU - it is sklearn on composition features, so it runs on the laptop while the
Kaggle job is still queued.

THE SUBSET MUST BE THE SAME CRYSTALS
--------------------------------------
`02_train.py --train-subsample N` draws its subset with
`np.random.RandomState(split_seed).permutation(len(train_idx))[:N]`. That exact
line is reproduced below rather than approximated, so the tree and the network
are fitted on the identical 3,894 crystals. If the two ever diverge the
comparison silently stops being a controlled one, so the reproduction is
verified by regenerating the full split the same way `02_train.py` does.

WHAT TO EXPECT
----------------
On the full matbench training set, step 43 measured the best tree at 0.0868
(K) and 0.1062 (G) against the CGCNN's 0.0630/0.0781 - the CGCNN wins by ~27%.
A random forest loses far less to a halved training set than a graph network
typically does, so the interesting outcome is not "the tree gets worse" but
"how much the GAP moves".
"""

# =============================================================================
#  CONFIG - every path, size and hyperparameter lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "labels_csv": "data_full/labels.csv",
    "graphs_pt": "data_full/graphs.pt",
    "id_col": "mb_id",
    "atom_init": "cgcnn_scratch/atom_init.json",

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/underfitting/csv",
    "png_dir": "results/cgcnn/underfitting/png",

    # ---- the split, identical to 02_train.py's ------------------------------
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "split_seed": 42,
    # AFLOW's training-set size. None = the full training split.
    "subset_sizes": [3894, None],

    # ---- the models, identical to step 43's ---------------------------------
    "random_forest_n_estimators": 300,
    "xgboost_n_estimators": 300,
    "xgboost_max_depth": 6,
    # tree_method="hist" (xgboost's default) segfaults on this machine even at
    # n_jobs=1 with no torch imported. "exact" is stable. See step 43's config
    # note - this is a machine quirk, not a modelling choice.
    "xgboost_tree_method": "exact",
    "targets": ["K_VRH", "G_VRH"],
}
# =============================================================================

import os                    # paths and directory creation
import sys                   # sys.path manipulation and early exit
from importlib import import_module   # reusing step 43's feature builder

# torch first: the graph cache is a torch pickle, and MKL's duplicate libiomp5
# aborts the process if numpy/pandas load first.
import torch  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

_base = import_module("43_baseline_tree_models")   # build_feature_table, split_indices

BLUE, GREEN, ORANGE = "#2a78d6", "#3f9142", "#eb6834"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def subset_train(train_idx, n, split_seed):
    """Reproduce 02_train.py's --train-subsample selection EXACTLY.

    Copied line-for-line from that script rather than reimplemented, because
    the entire experiment depends on the tree and the network seeing the same
    crystals. A different RNG call, or the same call in a different order,
    would silently break the control.
    """
    if not n:
        return list(train_idx)
    rng = np.random.RandomState(split_seed)
    return [train_idx[i] for i in rng.permutation(len(train_idx))[:n]]


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 54 - tree control at AFLOW's training-set size, on matbench")
    print("=" * 78)
    print()

    ari = _base.AtomFeaturiser(os.path.join(PROJECT_ROOT, cfg["atom_init"]))
    X, meta = _base.build_feature_table(cfg["labels_csv"], cfg["graphs_pt"],
                                        cfg["id_col"], ari)
    n_total = len(meta)
    tr, va, te = _base.split_indices(n_total, cfg["train_ratio"],
                                     cfg["val_ratio"], cfg["split_seed"])
    print(f"  matbench: {n_total} crystals -> {len(tr)} train / {len(va)} val / "
          f"{len(te)} test  (split_seed={cfg['split_seed']})")
    print()

    rows = []
    for target in cfg["targets"]:
        y = np.log10(meta[target].to_numpy(dtype=float))
        ok = np.isfinite(y)          # a non-positive modulus has no log; step 43 drops these too

        for n_sub in cfg["subset_sizes"]:
            idx = subset_train(tr, n_sub, cfg["split_seed"])
            idx = [i for i in idx if ok[i]]
            test_idx = [i for i in te if ok[i]]
            size_label = "sub" if n_sub else "full"

            models = {
                "random_forest": RandomForestRegressor(
                    n_estimators=cfg["random_forest_n_estimators"],
                    random_state=cfg["split_seed"], n_jobs=1),
                "xgboost": XGBRegressor(
                    n_estimators=cfg["xgboost_n_estimators"],
                    max_depth=cfg["xgboost_max_depth"],
                    tree_method=cfg["xgboost_tree_method"],
                    random_state=cfg["split_seed"], n_jobs=1),
            }
            for name, model in models.items():
                model.fit(X[idx], y[idx])
                tr_mae = float(np.mean(np.abs(model.predict(X[idx]) - y[idx])))
                te_mae = float(np.mean(np.abs(model.predict(X[test_idx]) - y[test_idx])))
                rows.append({"target": target, "size": size_label,
                             "n_train": len(idx), "model": name,
                             "train_mae": tr_mae, "test_mae": te_mae,
                             "test_over_train": te_mae / max(tr_mae, 1e-9)})
                print(f"  {target:<6} {size_label:<5} n={len(idx):<5} "
                      f"{name:<14} train {tr_mae:.4f}  test {te_mae:.4f}")

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(csv_dir, "54_matbench_subset_trees.csv"), index=False)

    print()
    print("  " + "-" * 74)
    print("  HOW MUCH DOES HALVING THE TRAINING SET COST THE TREE?")
    print("  " + "-" * 74)
    for target in cfg["targets"]:
        sub = scores[scores.target == target]
        best_full = sub[sub["size"] == "full"]["test_mae"].min()
        best_sub = sub[sub["size"] == "sub"]["test_mae"].min()
        print(f"    {target:<6} full {best_full:.4f} -> subset {best_sub:.4f}   "
              f"({100 * (best_sub - best_full) / best_full:+.1f}%)")
    print()
    print("  The CGCNN half of this comparison comes from the Kaggle run; "
          "step 55 joins them.")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(11.0, 4.8))
    if len(cfg["targets"]) == 1:
        axes = [axes]
    for ax, target in zip(axes, cfg["targets"]):
        sub = scores[scores.target == target]
        for i, model in enumerate(sub.model.unique()):
            m = sub[sub.model == model].sort_values("n_train")
            ax.plot(m["n_train"], m["test_mae"], "o-",
                    color=[GREEN, ORANGE][i % 2], label=f"tree: {model}")
        ax.set_xlabel("training crystals", color=INK_SOFT)
        ax.set_ylabel("test MAE log10", color=INK_SOFT)
        ax.set_title(f"{target} on matbench", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Tree control: what halving the training set costs",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "54_matbench_subset_trees.png"), dpi=150,
                facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    print(f"    {cfg['csv_dir']}/54_matbench_subset_trees.csv")
    print(f"    {cfg['png_dir']}/54_matbench_subset_trees.png")


if __name__ == "__main__":
    main()
