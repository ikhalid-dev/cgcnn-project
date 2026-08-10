#!/usr/bin/env python3
"""
STEP 17 - Diagnostic figures for the trained ALIGNN models.
=============================================================

    python scripts/alignn/17_alignn_diagnostics.py

WHAT THIS IS FOR
------------------
scripts/cgcnn/03_evaluate.py produces a parity plot, a training-history plot
and a residuals plot for every CGCNN run. results/alignn/ never got the
equivalent - ALIGNN's own training loop (scripts/alignn/12_train_alignn.py)
writes its raw JSON/CSV output straight to disk without plotting any of it,
so the data has always been there, just never turned into a figure.

This script reuses scripts/cgcnn/03_evaluate.py's own plot_parity(),
plot_residuals() and metrics() - via the same importlib pattern
scripts/alignn/15_alignn_predict_moduli.py and 16_alignn_predict_kappa.py
already use to reuse 07_predict_kappa.py's physics - so these figures are
pixel-for-pixel the same style as the CGCNN ones, not a lookalike
reimplementation that could quietly drift from it.

WHY THE TRAINING-CURVE PLOT IS NOT ALSO REUSED VERBATIM
------------------------------------------------------------
03_evaluate.py's plot_training() expects CGCNN's own history_<tag>.csv shape:
train/val LOSS and train/val MAE, both in log10 space, because CGCNN trains
on log10(GPa) (see 02_train.py). ALIGNN's history_train.json/history_val.json
record something different - raw-GPa MSE only, no log-space or MAE series at
all (ALIGNN regresses raw GPa directly; see 15_alignn_predict_moduli.py's own
note on this). Relabelling ALIGNN's numbers as if they were CGCNN's would
misdescribe them, so plot_training_alignn() below is its own function: MSE
(GPa^2) on the left, exactly what history_*.json records; RMSE (GPa) =
sqrt(MSE) on the right, an honestly-labelled derived quantity rather than a
fabricated MAE nobody tracked. Colours and rcParams are imported from
03_evaluate.py, not redefined, so it still reads as the same figure family.

WHERE THE DATA COMES FROM (nothing here is recomputed)
------------------------------------------------------------
    results/alignn/alignn_<target>/prediction_results_test_set.csv
        one row per test crystal: id, target (raw GPa), prediction (raw GPa)
    results/alignn/alignn_<target>/prediction_results_train_set.csv
        one row per train crystal: target, prediction - no id column, this
        version of the alignn library simply doesn't write one for train
    results/alignn/alignn_<target>/Val_results.json
        the validation set, but written as ~25 whole BATCHES from the run's
        last improving epoch (see alignn/train.py), not one row per crystal -
        flattened below rather than treated as 25 data points
    results/alignn/alignn_<target>/history_train.json / history_val.json
        [running_loss, running_loss1(=graphwise loss), 0, 0, 0, 0] per epoch;
        confirmed by reading alignn/train.py directly - columns 0 and 1 are
        identical here because loss2..5 (atomwise/force/stress/dos) are
        always 0 for a single scalar-target run like this one, and the
        criterion (config.json: "mse") is applied to the RAW-GPa target
        (Test_results.json's target_out values match prediction_results_*'s
        raw-GPa target column exactly - not normalised).

OUTPUT
------
    results/alignn/parity_<K|G>_VRH_alignn.png
    results/alignn/residuals_<K|G>_VRH_alignn.png
    results/alignn/training_<K|G>_VRH_alignn.png
"""

import json
import os
import sys
import warnings
from importlib import import_module

# torch first - see cgcnn_scratch/data.py for why (MKL/libiomp5 duplicate
# OpenMP runtime segfault if numpy/pandas import first in this env). This
# script never calls torch directly, but import_module("03_evaluate") below
# does, and by then it would be too late - numpy/pandas must not land first.
import torch  # noqa: F401

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# scripts/cgcnn/, not this file's own scripts/alignn/ - see 11's identical note.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
_evaluate = import_module("03_evaluate")  # plot_parity, plot_residuals, metrics, colours

TARGETS = [("bulk_modulus_kv", "K_VRH"), ("shear_modulus_gv", "G_VRH")]


def load_predictions(model_dir):
    """Build the {material_id, split, true_log10, pred_log10, true_GPa,
    pred_GPa, abs_error_log10, abs_error_GPa} shape 03_evaluate.py's own
    collect_predictions() produces for CGCNN, so its metrics()/plot_parity()/
    plot_residuals() run on ALIGNN's data completely unmodified.
    """
    rows = []

    test = pd.read_csv(os.path.join(model_dir, "prediction_results_test_set.csv"))
    for r in test.itertuples(index=False):
        rows.append({"material_id": r.id, "split": "test",
                    "true_GPa": float(r.target), "pred_GPa": float(r.prediction)})

    train = pd.read_csv(os.path.join(model_dir, "prediction_results_train_set.csv"))
    for r in train.itertuples(index=False):
        rows.append({"material_id": None, "split": "train",
                    "true_GPa": float(r.target), "pred_GPa": float(r.prediction)})

    # Val_results.json is whole batches from the run's last improving epoch
    # (see module docstring), not one row per crystal - flatten instead of
    # treating each dict as a single point.
    with open(os.path.join(model_dir, "Val_results.json")) as fh:
        val_batches = json.load(fh)
    for batch in val_batches:
        for t, p in zip(batch["target_out"], batch["pred_out"]):
            rows.append({"material_id": None, "split": "val",
                        "true_GPa": float(t), "pred_GPa": float(p)})

    df = pd.DataFrame(rows)
    # ALIGNN regresses raw GPa directly; a tiny positive floor guards the
    # log10() below against a rare negative/near-zero regression output -
    # same floor scripts/alignn/15_alignn_predict_moduli.py already applies.
    # (Checked: 0 of ~9,300 predictions here actually need it.)
    df["pred_GPa"] = df.pred_GPa.clip(lower=1e-3)
    df["true_log10"] = np.log10(df.true_GPa)
    df["pred_log10"] = np.log10(df.pred_GPa)
    df["abs_error_log10"] = (df.pred_log10 - df.true_log10).abs()
    df["abs_error_GPa"] = (df.pred_GPa - df.true_GPa).abs()
    return df


def plot_training_alignn(train_hist, val_hist, target, path):
    """MSE (GPa^2) and its square root, RMSE (GPa) - see module docstring for
    why this is a separate function rather than 03_evaluate.py's
    plot_training() relabelled.
    """
    import matplotlib.pyplot as plt
    e = _evaluate  # colours/rcParams already set at import time
    epochs = np.arange(1, len(train_hist) + 1)
    train_mse = np.array([row[0] for row in train_hist])
    val_mse = np.array([row[0] for row in val_hist])

    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, train_mse, color=e.BLUE, linewidth=2, label="Train")
    ax_loss.plot(epochs, val_mse, color=e.ORANGE, linewidth=2, label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE (GPa$^2$)")
    ax_loss.set_title("Loss", fontsize=11, pad=10)
    ax_loss.set_yscale("log")  # loss falls by orders of magnitude early on

    # Right: the same loss, back in interpretable GPa units, plus a marker on
    # the best epoch - mirrors 03_evaluate.py's plot_training() layout, but
    # see the module docstring for why this is RMSE, not MAE.
    train_rmse, val_rmse = np.sqrt(train_mse), np.sqrt(val_mse)
    ax_rmse.plot(epochs, train_rmse, color=e.BLUE, linewidth=2, label="Train")
    ax_rmse.plot(epochs, val_rmse, color=e.ORANGE, linewidth=2, label="Validation")
    best_epoch = int(np.argmin(val_rmse)) + 1
    best_value = float(val_rmse.min())
    ax_rmse.scatter([best_epoch], [best_value], s=60, color=e.ORANGE,
                    edgecolor=e.SURFACE, linewidth=1.5, zorder=5)
    ax_rmse.annotate(f"best: {best_value:.2f} GPa @ epoch {best_epoch}",
                     xy=(best_epoch, best_value),
                     xytext=(0.40, 0.70), textcoords="axes fraction",
                     ha="center", va="bottom", fontsize=9, color=e.INK_SOFT,
                     arrowprops=dict(arrowstyle="-", color=e.MUTED,
                                     linewidth=0.8, shrinkA=2, shrinkB=6))
    ax_rmse.set_xlabel("Epoch")
    ax_rmse.set_ylabel("RMSE (GPa)")
    ax_rmse.set_title("Root mean squared error", fontsize=11, pad=10)

    for ax in (ax_loss, ax_rmse):
        ax.legend(frameon=False, fontsize=9, labelcolor=e.INK_SOFT)
        ax.grid(True, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{target}: training history (ALIGNN)", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results", "alignn")
    print("=== ALIGNN diagnostic figures ===\n")

    for alignn_target, label in TARGETS:
        model_dir = os.path.join(results_dir, f"alignn_{alignn_target}")
        print(f"--- {label} ({model_dir}) ---")

        df = load_predictions(model_dir)
        scores = _evaluate.metrics(df)
        print(scores.round(4).to_string(index=False))

        tag = f"{label}_alignn"
        _evaluate.plot_parity(df, label, os.path.join(results_dir, f"parity_{tag}.png"),
                              model_name="ALIGNN")
        _evaluate.plot_residuals(df, label, os.path.join(results_dir, f"residuals_{tag}.png"))

        with open(os.path.join(model_dir, "history_train.json")) as fh:
            train_hist = json.load(fh)
        with open(os.path.join(model_dir, "history_val.json")) as fh:
            val_hist = json.load(fh)
        plot_training_alignn(train_hist, val_hist, label,
                             os.path.join(results_dir, f"training_{tag}.png"))

        print(f"Wrote parity_{tag}.png, residuals_{tag}.png, training_{tag}.png\n")


if __name__ == "__main__":
    main()
