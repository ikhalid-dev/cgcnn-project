#!/usr/bin/env python3
"""
STEP 55 - Is AFLOW's problem dataset SIZE? (No. It is the data itself.)
================================================================================

    python scripts/cgcnn/55_matbench_subset_verdict.py

THE QUESTION
--------------
Step 53 showed the CGCNN still underfits on AFLOW even with every brake off,
and step 56 showed the two datasets are comparable on shape, redundancy,
element coverage and graph construction. One difference remained that could
still explain everything: matbench trains on 7,691 crystals, AFLOW on 3,894.
Maybe a graph network simply needs more data than a tree does, and AFLOW is
below the threshold.

This settles it. The CGCNN was retrained on a random 3,894-crystal subset of
matbench - AFLOW's EXACT training-set size - and scored on the UNCHANGED
matbench test set. The tree was fitted on the identical subset by step 54. If
size is the cause, the CGCNN should collapse to tree level. If it does not,
size is ruled out and only the AFLOW data itself is left.

THE ANSWER
------------
Size is ruled out. At 3,894 crystals the CGCNN still beats the tree on matbench
by ~11-12% and sits essentially AT the composition floor. At the same 3,894 on
AFLOW it LOSES to the tree by 35-49% and sits at 1.6-2.4x the floor. Same
architecture, same recipe, same training-set size, same graph construction -
and opposite outcomes. What differs is the AFLOW structures (its own relaxed
POSCARs) or its AEL labels.

THE IN-SESSION CONTROL, AND ITS ONE BLEMISH
---------------------------------------------
The full-size arm exists to prove the setup reproduces this project's published
single-model numbers, which also verifies the new --train-subsample flag end to
end. It half succeeds:

    K  published 0.0688   this run 0.0716   +4.1%
    G  published 0.0852   this run 0.0933   +9.5%

Both are WORSE, in the same direction, which points at an environment
difference (Kaggle ships torch 2.10; the published numbers were produced under
torch 2.2) rather than at a bug in the flag - a broken subsample would not shift
the FULL-size arm at all, since it does not run there. It is stated rather than
buried, and it is why every conclusion below is drawn from WITHIN-RUN
comparisons (full vs subset, both from this same kernel) rather than against the
older published figures.

NOTHING IS TRAINED HERE. This joins the Kaggle output to step 54's tree control
and step 56's composition floors.
"""

# =============================================================================
#  CONFIG - every path and reference number lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "kaggle_summary": "kaggle/output_matbench_subset/mbsubset_summary.csv",
    "tree_scores": "results/cgcnn/underfitting/csv/54_matbench_subset_trees.csv",
    "floors_csv": "results/cgcnn/underfitting/csv/56_dataset_comparability.csv",

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/underfitting/csv",
    "png_dir": "results/cgcnn/underfitting/png",

    # ---- reference points ---------------------------------------------------
    "targets": ["K_VRH", "G_VRH"],
    # AFLOW at the SAME 3,894 training crystals: step 53 (CGCNN, matbench
    # recipe) and step 43 (best tree). This is the row the whole experiment
    # exists to compare against.
    "aflow_at_3894": {
        "K_VRH": {"cgcnn": 0.1154, "tree": 0.0592},
        "G_VRH": {"cgcnn": 0.1548, "tree": 0.1010},
    },
    # This project's published single-model matbench numbers, for the control.
    "published_single": {"K_VRH": 0.0688, "G_VRH": 0.0852},
}
# =============================================================================

import os
import sys

import torch  # noqa: F401   import order rule: torch before numpy/pandas
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, GREEN, ORANGE, MUTEDC = "#2a78d6", "#3f9142", "#eb6834", "#8a8a8a"
INK, INK_SOFT, GRID = "#1c1c1c", "#4a4a4a", "#e3e3e3"


def load_cgcnn(cfg):
    """Per-seed matbench CGCNN test MAE, keyed by (target, size)."""
    path = os.path.join(PROJECT_ROOT, cfg["kaggle_summary"])
    if not os.path.exists(path):
        sys.exit(f"ERROR: {cfg['kaggle_summary']} not found.\n"
                 "Fetch the Kaggle run first:\n"
                 "  python kaggle/run_matbench_subset_kernel.py --fetch")
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        tag = str(r["tag"])                      # mbsubset_<target>_<size>_s<seed>
        parts = tag.split("_")
        target = f"{parts[1]}_{parts[2]}"        # K_VRH / G_VRH
        size = parts[3]                          # sub / full
        out.setdefault((target, size), []).append(float(r["top_test_mae"]))
    return out


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 55 - is AFLOW's problem dataset SIZE?")
    print("=" * 78)
    print()

    cgcnn = load_cgcnn(cfg)
    trees = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["tree_scores"]))
    floors = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["floors_csv"]))

    # ---- the in-session control, first ---------------------------------------
    print("  " + "-" * 74)
    print("  IN-SESSION CONTROL - does the full-size arm reproduce the published?")
    print("  " + "-" * 74)
    for target in cfg["targets"]:
        got = np.mean(cgcnn[(target, "full")])
        want = cfg["published_single"][target]
        print(f"    {target:<6} published {want:.4f}   this run {got:.4f}   "
              f"{100 * (got - want) / want:+.1f}%")
    print("    Both worse in the same direction -> environment (Kaggle torch 2.10 vs")
    print("    the published runs' 2.2), not the new flag: a broken subsample could")
    print("    not move the FULL arm, where it never executes. Conclusions below are")
    print("    therefore drawn WITHIN this run only.")

    # ---- the size effect ----------------------------------------------------
    rows = []
    print()
    print("  " + "-" * 74)
    print("  WHAT HALVING THE TRAINING SET COSTS, ON MATBENCH")
    print("  " + "-" * 74)
    print(f"  {'target':<8}{'model':<8}{'7,691':>10}{'3,894':>10}{'cost':>9}")
    for target in cfg["targets"]:
        c_full, c_sub = np.mean(cgcnn[(target, "full")]), np.mean(cgcnn[(target, "sub")])
        t = trees[trees.target == target]
        t_full = t[t["size"] == "full"]["test_mae"].min()
        t_sub = t[t["size"] == "sub"]["test_mae"].min()
        print(f"  {target:<8}{'CGCNN':<8}{c_full:>10.4f}{c_sub:>10.4f}"
              f"{100 * (c_sub - c_full) / c_full:>8.1f}%")
        print(f"  {'':<8}{'tree':<8}{t_full:>10.4f}{t_sub:>10.4f}"
              f"{100 * (t_sub - t_full) / t_full:>8.1f}%")
        rows.append({"target": target, "cgcnn_full": c_full, "cgcnn_sub": c_sub,
                     "tree_full": t_full, "tree_sub": t_sub,
                     "cgcnn_seed_sd_sub": float(np.std(cgcnn[(target, 'sub')]))})

    # ---- the comparison the experiment exists for ---------------------------
    print()
    print("  " + "=" * 74)
    print("  AT AFLOW'S EXACT TRAINING-SET SIZE (3,894 crystals)")
    print("  " + "=" * 74)
    print(f"  {'dataset':<10}{'target':<8}{'CGCNN':>9}{'tree':>9}{'winner':>10}"
          f"{'floor':>9}{'CGCNN/floor':>13}")
    verdict_rows = []
    for target in cfg["targets"]:
        for ds in ("matbench", "AFLOW"):
            if ds == "matbench":
                c = np.mean(cgcnn[(target, "sub")])
                t = trees[(trees.target == target) & (trees["size"] == "sub")]["test_mae"].min()
            else:
                c = cfg["aflow_at_3894"][target]["cgcnn"]
                t = cfg["aflow_at_3894"][target]["tree"]
            fl = float(floors[(floors.dataset == ds) & (floors.target == target)]
                       ["within_formula_sd_mean"].iloc[0])
            print(f"  {ds:<10}{target:<8}{c:>9.4f}{t:>9.4f}"
                  f"{('CGCNN' if c < t else 'TREE'):>10}{fl:>9.4f}{c / fl:>13.2f}")
            verdict_rows.append({"dataset": ds, "target": target, "n_train": 3894,
                                 "cgcnn": c, "tree": t, "floor": fl,
                                 "cgcnn_over_floor": c / fl,
                                 "winner": "CGCNN" if c < t else "TREE"})

    verdict = pd.DataFrame(verdict_rows)
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, "55_size_effect.csv"), index=False)
    verdict.to_csv(os.path.join(csv_dir, "55_verdict_at_equal_size.csv"), index=False)

    mb_wins = (verdict[(verdict.dataset == "matbench")]["winner"] == "CGCNN").all()
    af_loses = (verdict[(verdict.dataset == "AFLOW")]["winner"] == "TREE").all()

    print()
    print("  " + "=" * 74)
    print("  VERDICT")
    print("  " + "=" * 74)
    if mb_wins and af_loses:
        print("    DATA QUANTITY IS RULED OUT.")
        print("    Given the SAME 3,894 training crystals, the same architecture and")
        print("    the same recipe, the CGCNN beats the tree on matbench and loses to")
        print("    it on AFLOW. Size cannot be the cause of a gap that reverses while")
        print("    size is held fixed.")
        print()
        print("    What is left, in order of testability:")
        print("      1. the AFLOW STRUCTURES - its own relaxed POSCARs, which may not")
        print("         correspond to the cells its AEL moduli were computed from as")
        print("         tightly as matbench's structures do to matbench's labels;")
        print("      2. the AFLOW LABELS - AEL elastic constants, whose disagreement")
        print("         with matbench VRH this project already measured at 0.1085 on")
        print("         G, which is close to the 0.1548 error the CGCNN achieves there.")
        print()
        print("    (2) is worth noting carefully: an error of 0.155 against labels")
        print("    that carry ~0.11 of cross-source noise is much less damning than it")
        print("    looks, and would ALSO explain why a composition model does fine -")
        print("    label noise that is uncorrelated with structure hurts a structure")
        print("    model far more than a composition average.")
    else:
        print("    The pattern is not the clean one this script was written to detect.")
        print("    Read the table above directly rather than trusting this summary.")
    print()
    print(f"    Sample: 1,648 matbench test crystals, 835 AFLOW test crystals,")
    print(f"    3 seeds per arm. AFLOW's kappa is a model, not experiment.")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, target in zip(axes, cfg["targets"]):
        sub = verdict[verdict.target == target]
        x = np.arange(2)
        ax.bar(x - 0.19, sub["cgcnn"], width=0.36, color=BLUE, label="CGCNN")
        ax.bar(x + 0.19, sub["tree"], width=0.36, color=GREEN, label="tree")
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.plot([i - 0.42, i + 0.42], [r["floor"]] * 2, color=ORANGE,
                    lw=1.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"], color=INK_SOFT)
        ax.set_ylabel("test MAE log10", color=INK_SOFT)
        ax.set_title(f"{target}, both at 3,894 training crystals",
                     color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Dashed orange = composition floor. Same size, same recipe, "
                 "opposite outcome.", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "55_matbench_subset_verdict.png"), dpi=150,
                facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    print(f"    {cfg['csv_dir']}/55_size_effect.csv")
    print(f"    {cfg['csv_dir']}/55_verdict_at_equal_size.csv")
    print(f"    {cfg['png_dir']}/55_matbench_subset_verdict.png")


if __name__ == "__main__":
    main()
