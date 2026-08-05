#!/usr/bin/env python3
"""
STEP 3 - Evaluate a trained model and produce the figures and CSV.
==================================================================

    python scripts/03_evaluate.py --target K_VRH
    python scripts/03_evaluate.py --target G_VRH

WHAT THIS PRODUCES
------------------
    results/predictions_<target>.csv   per-crystal true vs predicted, with split tag
    results/parity_<target>.png        the headline plot: predicted vs true
    results/training_<target>.png      train/validation curves over epochs
    results/residuals_<target>.png     where and how the errors are distributed

HOW TO READ THE PARITY PLOT
---------------------------
Every point is one crystal: its DFT-computed modulus on the x-axis, our
prediction on the y-axis. A perfect model puts every point exactly on the
diagonal. Points ABOVE the line are over-predictions (we said stiffer than it
is), points BELOW are under-predictions. The shaded band marks the "within a
factor of 1.5" region - a useful tolerance for screening work, where we mainly
need the ranking to be right rather than the absolute value.

THE METRICS
-----------
    MAE    mean absolute error in log10(GPa). This is the number the paper
           reports, so it is what we compare against. An MAE of 0.10 means
           typical predictions are within 10^0.10 = 1.26x of the truth.
    R2     fraction of the variance in the true values our predictions explain.
           1.0 is perfect; 0.0 means we do no better than always guessing the
           mean. R2 can go NEGATIVE, which means worse than guessing the mean.
    MAE_GPa mean absolute error back in physical GPa, for intuition.

A NOTE ON WHICH SPLIT TO TRUST
------------------------------
Only the TEST numbers are meaningful as a measure of generalisation. Train
numbers tell you the model has capacity; validation numbers were used to pick
the checkpoint, so they are mildly optimistic. The script reports all three so
the gap between them is visible - a large train/test gap is the signature of
overfitting, which is exactly what we expect to have to manage on 278 samples.
"""

import argparse
import json
import os
import sys
import warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to file, no interactive window needed
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.data import CIFData, Normalizer, collate_pool  # noqa: E402
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402

# ---------------------------------------------------------------------------
# Plot styling.
#
# Colours come from the project's validated categorical palette: slot 1 blue
# and slot 2 orange. Only two series ever appear together in these figures, and
# these two slots are documented as clearing the all-pairs colour-vision-
# deficiency gates, so the pairing stays readable for colourblind readers.
#
# Everything that is TEXT (labels, ticks, annotations) uses the neutral ink
# colours rather than a series colour - the coloured mark beside a label is
# what carries identity, so the text never has to.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # series 1
ORANGE = "#eb6834"    # series 2
INK = "#0b0b0b"       # primary text
INK_SOFT = "#52514e"  # secondary text
MUTED = "#898781"     # axis labels, ticks
GRID = "#e1e0d9"      # hairline gridlines
SURFACE = "#fcfcfb"   # chart background

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,    # recessive chrome: drop the box
    "axes.spines.right": False,
})


def load_model_and_data(target, data_dir, results_dir):
    """Rebuild the dataset, model and normalizer exactly as training left them."""
    ckpt = torch.load(os.path.join(results_dir, f"model_{target}.pth"),
                      map_location="cpu", weights_only=False)
    targs = ckpt["args"]

    # 02_train.py wrote id_prop.csv for this target; regenerate it so that
    # evaluating G_VRH after training K_VRH does not silently read the wrong
    # targets left behind by the previous run.
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    labels = labels[labels[target] > 0]
    pd.DataFrame({"cif_id": labels.material_id + ".cif",
                  "target": np.log10(labels[target])}).to_csv(
        os.path.join(data_dir, "cifs", "id_prop.csv"), index=False)

    dataset = CIFData(os.path.join(data_dir, "cifs"))

    model = CrystalGraphConvNet(
        ckpt["feature_lens"]["orig_atom_fea_len"],
        ckpt["feature_lens"]["nbr_fea_len"],
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normalizer = Normalizer(torch.zeros(1))
    normalizer.load_state_dict(ckpt["normalizer"])

    return model, dataset, normalizer, ckpt


def collect_predictions(model, dataset, normalizer, split):
    """Run the model over every split and return one tidy DataFrame.

    The split dict came from the training checkpoint, so train/val/test here
    mean exactly what they meant during training - no risk of a reshuffle
    quietly moving a training crystal into the test set.
    """
    rows = []
    for split_name, indices in split.items():
        loader = DataLoader(dataset, sampler=SubsetRandomSampler(indices),
                            batch_size=32, collate_fn=collate_pool)
        with torch.no_grad():
            for inputs, target, cif_ids in loader:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                pred_log = normalizer.denorm(output.data).view(-1).numpy()
                true_log = target.view(-1).numpy()
                for cif_id, t, p in zip(cif_ids, true_log, pred_log):
                    rows.append({
                        "material_id": os.path.splitext(cif_id)[0],
                        "split": split_name,
                        "true_log10": float(t),
                        "pred_log10": float(p),
                        "true_GPa": float(10 ** t),
                        "pred_GPa": float(10 ** p),
                    })
    df = pd.DataFrame(rows)
    df["abs_error_log10"] = (df.pred_log10 - df.true_log10).abs()
    df["abs_error_GPa"] = (df.pred_GPa - df.true_GPa).abs()
    return df


def metrics(df):
    """MAE / R2 / MAE-in-GPa per split, as a small DataFrame."""
    out = []
    for split_name in ["train", "val", "test"]:
        sub = df[df.split == split_name]
        if sub.empty:
            continue
        residual_ss = ((sub.true_log10 - sub.pred_log10) ** 2).sum()
        total_ss = ((sub.true_log10 - sub.true_log10.mean()) ** 2).sum()
        out.append({
            "split": split_name,
            "n": len(sub),
            "MAE_log10": sub.abs_error_log10.mean(),
            "R2": 1 - residual_ss / total_ss if total_ss > 0 else float("nan"),
            "MAE_GPa": sub.abs_error_GPa.mean(),
        })
    return pd.DataFrame(out)


def plot_parity(df, target, path):
    """Predicted vs true, test split highlighted against the training cloud."""
    fig, ax = plt.subplots(figsize=(6, 6))

    train_val = df[df.split != "test"]
    test = df[df.split == "test"]

    # Axis limits from the data, padded slightly, shared by both axes so the
    # diagonal is a true 45 degrees and distance from it reads correctly.
    lo = min(df.true_GPa.min(), df.pred_GPa.min()) * 0.7
    hi = max(df.true_GPa.max(), df.pred_GPa.max()) * 1.3

    # "Within a factor of 1.5" tolerance band, drawn first so it sits behind.
    ax.fill_between([lo, hi], [lo / 1.5, hi / 1.5], [lo * 1.5, hi * 1.5],
                    color=BLUE, alpha=0.07, linewidth=0, zorder=0)
    # The perfect-prediction diagonal.
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1,
            linestyle="--", zorder=1)

    # Train/val points are context, so they recede; test points are the result,
    # so they get the saturated colour and a surface-coloured ring to keep
    # overlapping markers legible.
    ax.scatter(train_val.true_GPa, train_val.pred_GPa, s=22, color=MUTED,
               alpha=0.30, linewidth=0, zorder=2, label="Train / validation")
    ax.scatter(test.true_GPa, test.pred_GPa, s=46, color=BLUE, alpha=0.9,
               edgecolor=SURFACE, linewidth=0.8, zorder=3, label="Test")

    # Log scales: the moduli span two orders of magnitude, and the model was
    # trained on log10, so a log axis is the space the errors actually live in.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    test_metrics = metrics(df).query("split == 'test'").iloc[0]
    ax.set_xlabel(f"DFT {target} (GPa)")
    ax.set_ylabel(f"CGCNN predicted {target} (GPa)")
    ax.set_title(f"{target}: predicted vs DFT reference", fontsize=12, pad=12)

    # Metrics as a text block rather than a subtitle - keeps the title short
    # and puts the numbers next to the data they describe.
    ax.text(0.03, 0.97,
            f"Test  n = {int(test_metrics.n)}\n"
            f"MAE = {test_metrics.MAE_log10:.3f} log10(GPa)\n"
            f"R2 = {test_metrics.R2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            color=INK_SOFT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                      edgecolor=GRID, linewidth=0.8))

    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SOFT)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)  # gridlines behind the data

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training(history, target, path):
    """Loss and MAE against epoch, train vs validation."""
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: the normalised MSE the optimiser actually minimises.
    ax_loss.plot(history.epoch, history.train_loss, color=BLUE, linewidth=2,
                 label="Train")
    ax_loss.plot(history.epoch, history.val_loss, color=ORANGE, linewidth=2,
                 label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE (normalised units)")
    ax_loss.set_title("Loss", fontsize=11, pad=10)
    ax_loss.set_yscale("log")  # loss falls by orders of magnitude early on

    # Right: MAE in interpretable units, plus a marker on the best epoch -
    # this is the checkpoint that was actually kept.
    ax_mae.plot(history.epoch, history.train_mae, color=BLUE, linewidth=2,
                label="Train")
    ax_mae.plot(history.epoch, history.val_mae, color=ORANGE, linewidth=2,
                label="Validation")
    best_epoch = int(history.val_mae.idxmin())
    best_value = history.val_mae.min()
    ax_mae.scatter([best_epoch], [best_value], s=60, color=ORANGE,
                   edgecolor=SURFACE, linewidth=1.5, zorder=5)
    # Both curves decay toward the bottom of the axes and the legend owns the
    # top-right, which leaves the upper-middle as the only reliably empty
    # region for this label. A leader line ties it back to the marker.
    ax_mae.annotate(f"best: {best_value:.3f} @ epoch {best_epoch}",
                    xy=(best_epoch, best_value),
                    xytext=(0.40, 0.70), textcoords="axes fraction",
                    ha="center", va="bottom", fontsize=9, color=INK_SOFT,
                    arrowprops=dict(arrowstyle="-", color=MUTED,
                                    linewidth=0.8, shrinkA=2, shrinkB=6))
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE (log10 GPa)")
    ax_mae.set_title("Mean absolute error", fontsize=11, pad=10)

    for ax in (ax_loss, ax_mae):
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
        ax.grid(True, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{target}: training history", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residuals(df, target, path):
    """Error distribution, and whether the error depends on how stiff the material is."""
    test = df[df.split == "test"]
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: signed error histogram. Centred on zero means unbiased; shifted
    # means the model systematically over- or under-predicts.
    signed = test.pred_log10 - test.true_log10
    ax_hist.hist(signed, bins=20, color=BLUE, alpha=0.85, edgecolor=SURFACE,
                 linewidth=0.8)
    ax_hist.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax_hist.axvline(signed.mean(), color=ORANGE, linewidth=2)
    # Corner label rather than one pinned to the mean line - the tallest bars
    # sit near zero, which is exactly where that line falls.
    ax_hist.text(0.97, 0.95, f"mean bias {signed.mean():+.3f}",
                 transform=ax_hist.transAxes, ha="right", va="top",
                 fontsize=9, color=INK_SOFT)
    ax_hist.set_xlabel("Prediction error, log10(GPa)")
    ax_hist.set_ylabel("Test crystals")
    ax_hist.set_title("Error distribution", fontsize=11, pad=10)

    # Right: does accuracy degrade at the extremes? A funnel shape here means
    # the model is only reliable in the middle of the range.
    ax_scatter.scatter(test.true_GPa, signed, s=40, color=BLUE, alpha=0.8,
                       edgecolor=SURFACE, linewidth=0.8)
    ax_scatter.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel(f"DFT {target} (GPa)")
    ax_scatter.set_ylabel("Prediction error, log10(GPa)")
    ax_scatter.set_title("Error vs material stiffness", fontsize=11, pad=10)

    for ax in (ax_hist, ax_scatter):
        ax.grid(True, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{target}: test-set residuals", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], default="K_VRH")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()

    target = args.target
    print(f"=== Evaluating {target} ===\n")

    model, dataset, normalizer, ckpt = load_model_and_data(
        target, args.data_dir, args.results_dir)
    df = collect_predictions(model, dataset, normalizer, ckpt["split"])

    scores = metrics(df)
    print(scores.round(4).to_string(index=False))

    # --- CSV ---------------------------------------------------------------
    pred_path = os.path.join(args.results_dir, f"predictions_{target}.csv")
    df.sort_values(["split", "abs_error_log10"]).to_csv(pred_path, index=False)

    metrics_path = os.path.join(args.results_dir, f"metrics_{target}.csv")
    scores.to_csv(metrics_path, index=False)

    # --- Figures -----------------------------------------------------------
    history = pd.read_csv(os.path.join(args.results_dir, f"history_{target}.csv"))
    plot_parity(df, target, os.path.join(args.results_dir, f"parity_{target}.png"))
    plot_training(history, target, os.path.join(args.results_dir, f"training_{target}.png"))
    plot_residuals(df, target, os.path.join(args.results_dir, f"residuals_{target}.png"))

    print(f"\nWrote:")
    for name in [f"predictions_{target}.csv", f"metrics_{target}.csv",
                 f"parity_{target}.png", f"training_{target}.png",
                 f"residuals_{target}.png"]:
        print(f"  results/{name}")

    # Worst predictions are worth eyeballing - they often reveal a systematic
    # weakness (a whole chemistry the model has never seen) rather than noise.
    worst = df[df.split == "test"].nlargest(5, "abs_error_log10")
    print("\nWorst test predictions:")
    print(worst[["material_id", "true_GPa", "pred_GPa", "abs_error_log10"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
