#!/usr/bin/env python3
"""
STEP 53 - Did the matbench recipe close the tree gap on AFLOW? (No.)
================================================================================

    python scripts/cgcnn/53_underfitting_verdict.py

THE QUESTION
--------------
Steps 43/44 found a composition-only decision tree beats the CGCNN on AFLOW,
and diagnosed UNDERFITTING rather than a real advantage. The evidence was one
number: round 9's TRAINING error on AFLOW K (0.109) is worse than the tree's
TEST error (0.059). A model that cannot fit data it has already seen has not
been out-generalised - it never learned.

Three suspects, all regularisation: dropout 0.10, huber loss, and three heads
competing on one shared trunk with only 3,894 training crystals. Step 52 +
the Kaggle kernel removed all three at once by training the ORIGINAL
single-target CGCNN with the MATBENCH recipe - no dropout, MSE, one target per
model, batch 32, lr 0.02, 200 epochs, plateau schedule, identical widths.

This script reads what came back and answers one question: is the train error
now below the tree's test error?

THE ANSWER, AND WHY IT IS NOT THE ONE THE README PRE-REGISTERED
----------------------------------------------------------------
No. The recipe helped - train error on K fell 0.117 -> 0.088, a genuine 25%
improvement - and it is still ABOVE the tree's 0.059 test error. The network,
with every brake released, still cannot fit crystals it has already seen as
well as a composition-only tree predicts crystals it has never seen.

The direct-kappa README pre-registered two branches for "the gap persists":
one where train error drops below the tree's (composition genuinely carries
the signal) and one where the model loses outright (underfitting). The actual
outcome is a THIRD state that table did not name - the gap persists AND the
model still underfits - which means the composition-vs-structure question is
still open, and the recipe was not the cause. Recording that the
pre-registration was incomplete matters more than pretending it covered this.

WHAT THIS RULES OUT, AND WHAT IS LEFT
---------------------------------------
Ruled out: dropout, huber, and multi-head competition. Removing all three
bought ~25% of training error and left the gap intact.

Left, and separable by one cheap experiment: on matbench the SAME architecture
with the SAME recipe BEATS the tree (0.0630 vs 0.0868); on AFLOW it loses.
Three things differ - training-set size (7,691 vs 3,894), graph source
(matminer structures vs AFLOW POSCARs), and label source (matbench VRH vs
AFLOW AEL). Training on a random 3,894-crystal subset of matbench would split
size from source in a single run.

NOTHING IS TRAINED HERE. This reads the summaries and history files the
Kaggle run already produced.
"""

# =============================================================================
#  CONFIG - every path and reference number lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    # Per-epoch history from 02_train.py: the ONLY place train MAE is recorded
    # (its summary JSON carries test and val but not train).
    "history_glob": "kaggle/output_aflow_recipe/predictions/history_aflow_mbrecipe_{target}_s*.csv",
    "recipe_summary": "kaggle/output_aflow_recipe/aflow_mbrecipe_summary.csv",
    "round9_glob": "results/cgcnn/summary_37_r9_full_s*.json",
    "tree_scores": "results/cgcnn/baseline_tree/csv/43_baseline_tree_scores.csv",

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/underfitting/csv",
    "png_dir": "results/cgcnn/underfitting/png",

    # ---- what to compare ----------------------------------------------------
    "targets": ["K_VRH", "G_VRH"],
    "tree_models": ["random_forest", "xgboost"],
    # matbench reference, where the SAME architecture and recipe BEAT the tree.
    # This is the contrast that makes the AFLOW result interesting rather than
    # just disappointing.
    "matbench_cgcnn": {"K_VRH": 0.0630, "G_VRH": 0.0781},   # 3-model ensemble
    "matbench_tree": {"K_VRH": 0.0868, "G_VRH": 0.1062},    # best tree, step 43
    "matbench_n_train": 7691,
    "aflow_n_train": 3894,
    # This project's own underfitting diagnostic. Healthy matbench baseline is
    # 2.36; round 9 sat at 1.08-1.23.
    "healthy_test_over_train": 2.36,
}
# =============================================================================

import glob                  # expanding the history/summary globs
import json                  # reading round 9's summaries
import os                    # paths and directory creation
import sys                   # early exit when an input is missing

import torch  # noqa: F401   # import order rule: torch before numpy/pandas
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def recipe_errors(cfg, target):
    """(train, test) MAE per seed for the matbench-recipe runs on one target.

    Train comes from the LAST ROW of the per-epoch history - 02_train.py's
    summary JSON records test and best-val but not train, and train is the
    whole point of this script.

    A caveat worth stating: the reported test MAE is from the BEST-VAL
    checkpoint, while this train MAE is from the FINAL epoch. With a plateau
    schedule that has annealed the LR to ~1e-3 by epoch 199 the two are
    essentially the same model, but they are not identical, so the ratio below
    is approximate. The conclusion does not turn on the third decimal.
    """
    pattern = os.path.join(PROJECT_ROOT, cfg["history_glob"].format(target=target))
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"ERROR: no history files matched {pattern}\n"
                 "Fetch the Kaggle run first:\n"
                 "  python kaggle/run_aflow_recipe_kernel.py --fetch")

    summary = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["recipe_summary"]))
    rows = []
    for path in paths:
        tag = os.path.basename(path).replace("history_", "").replace(".csv", "")
        hist = pd.read_csv(path)
        train_mae = float(hist["train_mae"].iloc[-1])
        match = summary[summary["tag"] == tag]
        test_mae = float(match["top_test_mae"].iloc[0]) if len(match) else np.nan
        rows.append({"tag": tag, "train": train_mae, "test": test_mae})
    return pd.DataFrame(rows)


def round9_errors(cfg, target):
    """(train, test) MAE per seed for round 9, from its own summaries."""
    key = "mae_log_K" if target == "K_VRH" else "mae_log_G"
    rows = []
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["round9_glob"]))):
        with open(path) as fh:
            res = json.load(fh)["results"]
        rows.append({"tag": os.path.basename(path),
                     "train": res["train"][key], "test": res["test"][key]})
    return pd.DataFrame(rows)


def tree_errors(cfg, target):
    """(train, test) MAE for each tree model on AFLOW, from step 43's scores."""
    scores = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["tree_scores"]))
    sub = scores[(scores["source"] == "aflow") & (scores["target"] == target)]
    rows = []
    for model in cfg["tree_models"]:
        m = sub[sub["model"] == model]
        rows.append({
            "tag": model,
            "train": float(m[m["split"] == "train"]["mae_log10"].iloc[0]),
            "test": float(m[m["split"] == "test"]["mae_log10"].iloc[0]),
        })
    return pd.DataFrame(rows)


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 53 - did the matbench recipe close the tree gap on AFLOW?")
    print("=" * 78)
    print()

    all_rows = []
    for target in cfg["targets"]:
        recipe = recipe_errors(cfg, target)
        r9 = round9_errors(cfg, target)
        trees = tree_errors(cfg, target)

        best_tree = trees.loc[trees["test"].idxmin()]

        print("  " + "-" * 74)
        print(f"  {target} on AFLOW - MAE log10, SINGLE models throughout")
        print("  " + "-" * 74)
        print(f"  {'model':<34}{'train':>9}{'test':>9}{'test/train':>12}")

        for label, frame, colour in (
                ("round 9 (dropout, huber, 3 heads)", r9, "r9"),
                ("matbench recipe (none of those)", recipe, "recipe")):
            tr, te = frame["train"].mean(), frame["test"].mean()
            print(f"  {label:<34}{tr:>9.4f}{te:>9.4f}{te / tr:>12.2f}")
            all_rows.append({"target": target, "arm": colour, "train": tr,
                             "test": te, "test_over_train": te / tr,
                             "train_sd": frame["train"].std(),
                             "test_sd": frame["test"].std()})
        for _, t in trees.iterrows():
            print(f"  {'tree: ' + t['tag']:<34}{t['train']:>9.4f}{t['test']:>9.4f}"
                  f"{t['test'] / t['train']:>12.2f}")
            all_rows.append({"target": target, "arm": f"tree_{t['tag']}",
                             "train": t["train"], "test": t["test"],
                             "test_over_train": t["test"] / t["train"],
                             "train_sd": np.nan, "test_sd": np.nan})

        # ---- the one comparison the whole experiment turns on ---------------
        recipe_train = recipe["train"].mean()
        print()
        print(f"    THE TEST: is the network's TRAIN error below the tree's TEST error?")
        print(f"      network train : {recipe_train:.4f}")
        print(f"      tree    test  : {best_tree['test']:.4f}  ({best_tree['tag']})")
        if recipe_train < best_tree["test"]:
            print(f"      -> YES. The network can fit its data; the tree's win on "
                  f"{target} is real.")
        else:
            print(f"      -> NO, by {recipe_train - best_tree['test']:+.4f}. The "
                  f"network STILL cannot fit crystals")
            print(f"         it has SEEN as well as the tree predicts crystals it "
                  f"has NOT. Still underfitting.")

        r9_train = r9["train"].mean()
        print(f"    recipe change moved train error {r9_train:.4f} -> "
              f"{recipe_train:.4f} "
              f"({100 * (r9_train - recipe_train) / r9_train:+.0f}%)")
        print()

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(csv_dir, "53_underfitting_summary.csv"), index=False)

    # ---- the contrast that makes this interesting ---------------------------
    print("  " + "=" * 74)
    print("  THE CONTRAST: same architecture, same recipe, two datasets")
    print("  " + "=" * 74)
    print(f"  {'':<10}{'CGCNN':>10}{'tree':>10}   verdict")
    for target in cfg["targets"]:
        mb_c, mb_t = cfg["matbench_cgcnn"][target], cfg["matbench_tree"][target]
        af_c = summary[(summary.target == target) & (summary.arm == "recipe")]["test"].iloc[0]
        af_t = summary[(summary.target == target)
                       & (summary.arm.str.startswith("tree"))]["test"].min()
        print(f"  matbench {target[:5]:<5}{mb_c:>9.4f}{mb_t:>10.4f}   "
              f"CGCNN wins by {100 * (mb_t - mb_c) / mb_t:.0f}%")
        print(f"  AFLOW    {target[:5]:<5}{af_c:>9.4f}{af_t:>10.4f}   "
              f"TREE wins by {100 * (af_c - af_t) / af_c:.0f}%")
    print()
    print(f"  Three things differ between those datasets:")
    print(f"    training crystals   {cfg['matbench_n_train']} vs {cfg['aflow_n_train']}")
    print(f"    graph source        matminer structures vs AFLOW POSCARs")
    print(f"    label source        matbench VRH vs AFLOW AEL")
    print(f"  Training on a random {cfg['aflow_n_train']}-crystal subset of matbench")
    print(f"  would separate SIZE from SOURCE in one run.")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(11.5, 5.2))
    if len(cfg["targets"]) == 1:
        axes = [axes]
    for ax, target in zip(axes, cfg["targets"]):
        sub = summary[summary.target == target]
        labels = ["round 9", "mb recipe"] + [
            a.replace("tree_", "tree\n") for a in sub.arm if a.startswith("tree")]
        colours = [MUTED, BLUE] + [GREEN] * (len(sub) - 2)
        x = np.arange(len(sub))
        ax.bar(x - 0.19, sub["train"], width=0.36, color=colours, alpha=0.55,
               label="train", edgecolor="none")
        ax.bar(x + 0.19, sub["test"], width=0.36, color=colours,
               label="test", edgecolor="none")
        best_tree_test = sub[sub.arm.str.startswith("tree")]["test"].min()
        ax.axhline(best_tree_test, color=GREEN, lw=1.2, ls="--")
        ax.annotate("best tree TEST error", xy=(0.02, best_tree_test),
                    xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8, color=GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=INK_SOFT)
        ax.set_ylabel("MAE log10", color=INK_SOFT)
        ax.set_title(f"{target} on AFLOW", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Light bars = train, solid = test. The network's TRAIN error is "
                 "still above the tree's TEST error.", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "53_underfitting_verdict.png"), dpi=150,
                facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    print(f"    {cfg['csv_dir']}/53_underfitting_summary.csv")
    print(f"    {cfg['png_dir']}/53_underfitting_verdict.png")


if __name__ == "__main__":
    main()
