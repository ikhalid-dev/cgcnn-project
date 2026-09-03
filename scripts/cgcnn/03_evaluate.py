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
overfitting.
"""

import argparse   # command-line argument parser (defines --target/--tag/--data-dir/--results-dir below)
import json       # JSON module (not referenced elsewhere in this file)
import os         # path joining/splitting for building result file paths
import sys        # used below to extend the import search path with PROJECT_ROOT
import warnings   # used below to silence noisy library warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch                                              # tensor library; loads the saved checkpoint and runs the forward pass
from torch.utils.data import DataLoader                   # batches dataset items for the forward pass below
from torch.utils.data.sampler import SubsetRandomSampler  # draws only the indices belonging to one split (train/val/test)

import numpy as np              # array math (quantiles, clipping) used in the plotting helpers
import pandas as pd             # DataFrame construction/CSV I/O for predictions and metrics
import matplotlib                # imported first so the Agg backend below is set before pyplot is imported
matplotlib.use("Agg")  # render to file, no interactive window needed
import matplotlib.pyplot as plt  # the plotting API used by the three plot_* functions

warnings.filterwarnings("ignore")  # suppress warning messages (e.g. pandas/matplotlib deprecation notices) from cluttering output

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# three dirname() calls walk up from this file's absolute path: scripts/cgcnn/03_evaluate.py -> scripts/cgcnn -> scripts -> project root
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of the current working directory

from cgcnn_scratch.data import Normalizer, collate_pool, load_dataset_for  # noqa: E402 - de-normalizer, batch-collation fn, dataset loader
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402 - the CGCNN architecture class used to rebuild the trained model

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


def mark_style(n):
    """Marker size, alpha and ring width appropriate to n points.

    The parity plot carries ~1,648 test points and ~9,300 context points, but
    the same code also renders small subsets. Mark specs that read well sparse
    turn into a solid blob dense, so they scale with n.

    Two things change with n. Marks shrink and go more transparent, so that
    overlap becomes visible as tonal build-up rather than a filled region - at
    high density the shading IS the information. And the 2px surface ring, which
    exists to separate individual overlapping marks, is dropped: past a few
    hundred points it stops separating anything and just floods the plot with
    pale halos that lighten the dense regions most, exactly backwards.
    """
    if n <= 150:                                      # small dataset: sparse points, safe to draw them large and opaque
        return dict(s=46, alpha=0.90, linewidth=0.8)   # marker size 46pt, 90% opaque, 0.8pt outline ring
    if n <= 800:                                       # medium dataset: shrink and fade so overlaps start to blend
        return dict(s=18, alpha=0.45, linewidth=0.0)   # smaller marker, 45% opaque, no outline ring
    return dict(s=7, alpha=0.28, linewidth=0.0)        # large dataset: tiny, mostly-transparent marks so density reads as shading


plt.rcParams.update({                  # sets these style defaults globally for every figure created below
    "figure.facecolor": SURFACE,       # background color of the whole figure canvas
    "axes.facecolor": SURFACE,         # background color inside the plot axes
    "axes.edgecolor": "#c3c2b7",       # color of the axes border box
    "axes.labelcolor": INK_SOFT,       # color of the x/y axis labels
    "axes.titlecolor": INK,            # color of the plot title text
    "text.color": INK,                 # default color for any other text
    "xtick.color": MUTED,              # color of x-axis tick marks/labels
    "ytick.color": MUTED,              # color of y-axis tick marks/labels
    "grid.color": GRID,                # color of gridlines
    "font.family": "sans-serif",       # default font family for all text
    "font.size": 10,                   # default font size in points
    "axes.spines.top": False,    # recessive chrome: drop the box
    "axes.spines.right": False,
})


def load_model_and_data(target, tag, data_dir, results_dir):
    """Rebuild the dataset, model and normalizer exactly as training left them.

    `load_dataset_for` is the same function 02_train.py used, so the dataset is
    rebuilt in identical order - which is what makes the split indices stored in
    the checkpoint mean the same thing here as they did during training.
    """
    # loads the checkpoint dict saved by 02_train.py onto the CPU regardless of
    # what device it was trained on; weights_only=False because the checkpoint
    # also carries plain Python objects (args dict, split indices), not just tensors
    ckpt = torch.load(os.path.join(results_dir, f"model_{tag}.pth"),
                      map_location="cpu", weights_only=False)
    targs = ckpt["args"]                          # the CONFIG/CLI args 02_train.py was run with, needed to rebuild the same architecture

    dataset = load_dataset_for(data_dir, target)   # rebuilds the CGCNN graph dataset for this target, same order as training

    model = CrystalGraphConvNet(                   # re-instantiate the model with the exact hyperparameters training used
        ckpt["feature_lens"]["orig_atom_fea_len"],   # size of each atom's raw (pre-embedding) feature vector
        ckpt["feature_lens"]["nbr_fea_len"],         # size of each bond's Gaussian-expanded distance feature vector
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)
    model.load_state_dict(ckpt["state_dict"])      # copies the trained weights into the freshly built model
    model.eval()                                   # switches off dropout/batchnorm training behaviour for inference

    normalizer = Normalizer(torch.zeros(1))        # placeholder normalizer, immediately overwritten by load_state_dict below
    normalizer.load_state_dict(ckpt["normalizer"]) # restores the exact mean/std used to de-normalize predictions at training time

    return model, dataset, normalizer, ckpt


def collect_predictions(model, dataset, normalizer, split):
    """Run the model over every split and return one tidy DataFrame.

    The split dict came from the training checkpoint, so train/val/test here
    mean exactly what they meant during training - no risk of a reshuffle
    quietly moving a training crystal into the test set.
    """
    rows = []                                        # collects one dict per crystal across all splits
    for split_name, indices in split.items():        # iterate over {"train": [...], "val": [...], "test": [...]}
        loader = DataLoader(dataset, sampler=SubsetRandomSampler(indices),   # yields only the crystals at these indices, batched
                            batch_size=32, collate_fn=collate_pool)          # collate_pool assembles CGCNN's custom graph-batch format
        with torch.no_grad():                        # disables gradient tracking since this is inference only, saves memory
            for inputs, target, cif_ids in loader:    # one mini-batch: graph tensors, true labels, and crystal ids
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs   # unpack the four CGCNN graph-batch tensors
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # forward pass, normalized-space prediction
                pred_log = normalizer.denorm(output.data).view(-1).numpy()  # undo z-scoring, flatten to 1D, move to numpy
                true_log = target.view(-1).numpy()    # true label (already in log10 space), flattened to numpy
                for cif_id, t, p in zip(cif_ids, true_log, pred_log):   # walk the batch one crystal at a time
                    rows.append({
                        "material_id": os.path.splitext(cif_id)[0],    # strip the ".cif" extension off the id
                        "split": split_name,
                        "true_log10": float(t),
                        "pred_log10": float(p),
                        "true_GPa": float(10 ** t),   # convert back from log10 to physical GPa
                        "pred_GPa": float(10 ** p),
                    })
    df = pd.DataFrame(rows)                                          # one row per crystal, across every split
    df["abs_error_log10"] = (df.pred_log10 - df.true_log10).abs()    # per-crystal absolute error in log space
    df["abs_error_GPa"] = (df.pred_GPa - df.true_GPa).abs()          # per-crystal absolute error in physical GPa
    return df


def metrics(df):
    """MAE / R2 / MAE-in-GPa per split, as a small DataFrame."""
    out = []                                             # one summary row per split
    for split_name in ["train", "val", "test"]:
        sub = df[df.split == split_name]                 # rows belonging to just this split
        if sub.empty:                                    # e.g. this run has no rows for this split
            continue
        residual_ss = ((sub.true_log10 - sub.pred_log10) ** 2).sum()       # sum of squared prediction errors
        total_ss = ((sub.true_log10 - sub.true_log10.mean()) ** 2).sum()   # sum of squared deviations from the mean (baseline)
        out.append({
            "split": split_name,
            "n": len(sub),                               # number of crystals in this split
            "MAE_log10": sub.abs_error_log10.mean(),      # mean absolute error, log10(GPa) units
            "R2": 1 - residual_ss / total_ss if total_ss > 0 else float("nan"),  # coefficient of determination
            "MAE_GPa": sub.abs_error_GPa.mean(),           # mean absolute error, physical GPa units
        })
    return add_relative_columns(pd.DataFrame(out), df)    # attach the two derived percentage columns before returning


def add_relative_columns(scores, df):
    """Add the two error measures that are easier to talk about than log-MAE.

    rel_error_pct   the typical MULTIPLICATIVE error, 10^MAE - 1. An MAE of
                    0.070 log10 means predictions land within a factor of
                    10^0.070 = 1.17 of the truth, i.e. 17%. This is the honest
                    "percent error" for a model trained in log space, and it is
                    the number to compare against the paper.
    pct_of_range    MAE in GPa as a fraction of the full span of the data - the
                    "normalised MAE" convention. Much smaller than
                    rel_error_pct, because the denominator is the whole
                    1-575 GPa range rather than each crystal's own value. Quote
                    it only alongside rel_error_pct; on its own it flatters the
                    model, since a wide data range shrinks it for free.
    """
    span = df.true_GPa.max() - df.true_GPa.min()            # full range of true values across ALL splits (not just one)
    scores = scores.copy()                                   # avoid mutating the caller's DataFrame in place
    scores["rel_error_pct"] = (10 ** scores.MAE_log10 - 1) * 100  # convert log-MAE to an equivalent percent multiplicative error
    scores["pct_of_range"] = scores.MAE_GPa / span * 100     # MAE as a percentage of the full data range
    return scores


def plot_parity(df, target, path, model_name="CGCNN"):
    """Predicted vs true, test split highlighted against the training cloud.

    model_name only changes the y-axis label - kept a parameter (default
    CGCNN, this file's own subject) rather than hardcoded so
    scripts/alignn/17_alignn_diagnostics.py can call this same function for
    ALIGNN's predictions and get an identically-styled figure that still
    correctly says which model made the prediction.
    """
    fig, ax = plt.subplots(figsize=(6, 6))   # one square 6x6 inch figure with a single axes

    train_val = df[df.split != "test"]    # rows from the train and validation splits (the context cloud)
    test = df[df.split == "test"]         # rows from the held-out test split (the result being reported)

    # Axis limits from the data, padded slightly, shared by both axes so the
    # diagonal is a true 45 degrees and distance from it reads correctly.
    lo = min(df.true_GPa.min(), df.pred_GPa.min()) * 0.7   # lower bound, padded 30% below the smallest value
    hi = max(df.true_GPa.max(), df.pred_GPa.max()) * 1.3   # upper bound, padded 30% above the largest value

    # "Within a factor of 1.5" tolerance band, drawn first so it sits behind.
    ax.fill_between([lo, hi], [lo / 1.5, hi / 1.5], [lo * 1.5, hi * 1.5],
                    color=BLUE, alpha=0.07, linewidth=0, zorder=0)   # zorder=0 draws this band first, beneath everything else
    # The perfect-prediction diagonal.
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1,
            linestyle="--", zorder=1)   # a dashed line from (lo,lo) to (hi,hi), i.e. y=x

    # Train/val points are context, so they recede; test points are the result,
    # so they get the saturated colour. Sizes adapt to how many points there are
    # (see mark_style) - the context cloud is ~5x larger than the test set, so
    # it is sized against its own count rather than the test count.
    context = mark_style(len(train_val))    # marker size/alpha/linewidth tuned to the train+val point count
    result = mark_style(len(test))          # marker size/alpha/linewidth tuned to the test point count
    ax.scatter(train_val.true_GPa, train_val.pred_GPa, color=MUTED,   # plot train/val points, x=true, y=predicted
               s=context["s"] * 0.7, alpha=context["alpha"] * 0.55,   # further shrunk/faded relative to their own base style
               linewidth=0, zorder=2, label="Train / validation")
    ax.scatter(test.true_GPa, test.pred_GPa, color=BLUE, s=result["s"],  # plot test points, x=true, y=predicted
               alpha=result["alpha"], edgecolor=SURFACE,
               linewidth=result["linewidth"], zorder=3, label="Test")

    # Log scales: the moduli span two orders of magnitude, and the model was
    # trained on log10, so a log axis is the space the errors actually live in.
    ax.set_xscale("log")      # x-axis drawn in log10 spacing
    ax.set_yscale("log")      # y-axis drawn in log10 spacing
    ax.set_xlim(lo, hi)       # x-axis range set to the padded bounds computed above
    ax.set_ylim(lo, hi)       # y-axis range set to the same padded bounds (so the diagonal is exact)
    ax.set_aspect("equal")    # one data unit is the same physical length on both axes

    test_metrics = metrics(df).query("split == 'test'").iloc[0]   # recompute metrics and pull out just the test-split row
    ax.set_xlabel(f"DFT {target} (GPa)")
    ax.set_ylabel(f"{model_name} predicted {target} (GPa)")
    ax.set_title(f"{target}: predicted vs DFT reference", fontsize=12, pad=12)

    # Metrics as a text block rather than a subtitle - keeps the title short
    # and puts the numbers next to the data they describe.
    ax.text(0.03, 0.97,                                   # position in axes-fraction coords: near the top-left corner
            f"Test  n = {int(test_metrics.n)}\n"
            f"MAE = {test_metrics.MAE_log10:.3f} log10(GPa)\n"
            f"R2 = {test_metrics.R2:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,   # transAxes makes (0.03,0.97) relative to the axes, not data
            color=INK_SOFT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,    # draws a rounded box behind the text
                      edgecolor=GRID, linewidth=0.8))

    # The legend swatch must stay readable even when the plotted marks have
    # shrunk to 7px for density - identity is carried by the legend, so it is
    # the one place the mark is not allowed to become a speck. Scale the swatch
    # back up to a fixed readable size and undo the transparency with it.
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9,   # places the legend in the bottom-right corner, no border box
                       labelcolor=INK_SOFT,
                       markerscale=max(1.0, 40.0 / result["s"]))        # scales the legend swatch back up to a minimum readable size
    for handle in legend.legend_handles:    # each legend swatch (one per scatter series)
        handle.set_alpha(0.95)              # force the swatch to near-opaque regardless of the plotted points' transparency
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)   # draw faint gridlines at major tick positions
    ax.set_axisbelow(True)  # gridlines behind the data

    fig.tight_layout()           # auto-adjust spacing so labels/titles don't get clipped
    fig.savefig(path, dpi=160)   # write the figure to disk at 160 dots per inch
    plt.close(fig)               # free the figure's memory now that it's saved


def plot_training(history, target, path):
    """Loss and MAE against epoch, train vs validation."""
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.2))   # one row of two side-by-side panels, 11x4.2 inches total

    # Left: the normalised MSE the optimiser actually minimises.
    ax_loss.plot(history.epoch, history.train_loss, color=BLUE, linewidth=2,   # line plot: x=epoch number, y=training loss
                 label="Train")
    ax_loss.plot(history.epoch, history.val_loss, color=ORANGE, linewidth=2,   # line plot: x=epoch number, y=validation loss
                 label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE (normalised units)")
    ax_loss.set_title("Loss", fontsize=11, pad=10)
    ax_loss.set_yscale("log")  # loss falls by orders of magnitude early on

    # Right: MAE in interpretable units, plus a marker on the best epoch -
    # this is the checkpoint that was actually kept.
    ax_mae.plot(history.epoch, history.train_mae, color=BLUE, linewidth=2,    # line plot: x=epoch number, y=training MAE
                label="Train")
    ax_mae.plot(history.epoch, history.val_mae, color=ORANGE, linewidth=2,    # line plot: x=epoch number, y=validation MAE
                label="Validation")
    best_epoch = int(history.val_mae.idxmin())   # epoch index where validation MAE is lowest
    best_value = history.val_mae.min()           # that lowest validation MAE value
    ax_mae.scatter([best_epoch], [best_value], s=60, color=ORANGE,   # single highlighted point marking the best epoch
                   edgecolor=SURFACE, linewidth=1.5, zorder=5)
    # Both curves decay toward the bottom of the axes and the legend owns the
    # top-right, which leaves the upper-middle as the only reliably empty
    # region for this label. A leader line ties it back to the marker.
    ax_mae.annotate(f"best: {best_value:.3f} @ epoch {best_epoch}",   # text label with a leader line pointing at the marker
                    xy=(best_epoch, best_value),                       # the point being annotated, in data coordinates
                    xytext=(0.40, 0.70), textcoords="axes fraction",    # where the text itself sits, in axes-fraction coordinates
                    ha="center", va="bottom", fontsize=9, color=INK_SOFT,
                    arrowprops=dict(arrowstyle="-", color=MUTED,        # draws a plain leader line (no arrowhead) from text to point
                                    linewidth=0.8, shrinkA=2, shrinkB=6))
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE (log10 GPa)")
    ax_mae.set_title("Mean absolute error", fontsize=11, pad=10)

    for ax in (ax_loss, ax_mae):                                    # apply the same finishing touches to both panels
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)   # borderless legend, using each panel's own labels
        ax.grid(True, linewidth=0.6, alpha=0.7)                     # faint gridlines
        ax.set_axisbelow(True)                                      # gridlines drawn behind the plotted lines

    fig.suptitle(f"{target}: training history", fontsize=12, y=0.99)   # overall figure title above both panels
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residuals(df, target, path):
    """Error distribution, and whether the error depends on how stiff the material is."""
    test = df[df.split == "test"]     # only the held-out test rows are analysed here
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(11, 4.2))   # two side-by-side panels: histogram + scatter

    # Left: signed error histogram. Centred on zero means unbiased; shifted
    # means the model systematically over- or under-predicts.
    signed = test.pred_log10 - test.true_log10   # per-crystal signed error: positive = over-prediction, negative = under

    # A couple of catastrophic outliers would otherwise stretch the x-axis to
    # +/-1.7 and squash the entire distribution into three bars. Clip the view
    # to where the mass actually is - and say how many crystals fall outside,
    # so tightening the axis hides nothing.
    limit = max(float(np.quantile(np.abs(signed), 0.995)), 0.05) * 1.15   # 99.5th percentile of |error|, floored at 0.05, padded 15%
    outside = int((np.abs(signed) > limit).sum())                         # count of points that fall beyond that clipped range

    # More crystals support more bins without the histogram going spiky.
    bins = np.linspace(-limit, limit, 21 if len(test) <= 300 else 51)   # evenly spaced bin edges; more bins for bigger test sets
    ax_hist.hist(np.clip(signed, -limit, limit), bins=bins, color=BLUE,  # clip outliers into the edge bins rather than dropping them
                 alpha=0.85, edgecolor=SURFACE, linewidth=0.8)
    ax_hist.set_xlim(-limit, limit)                                     # crop the visible x-range to the same clipped window
    ax_hist.axvline(0, color=MUTED, linestyle="--", linewidth=1)        # dashed reference line at zero error
    ax_hist.axvline(signed.mean(), color=ORANGE, linewidth=2)           # solid line at the mean (signed) error
    # Corner label rather than one pinned to the mean line - the tallest bars
    # sit near zero, which is exactly where that line falls.
    note = f"mean bias {signed.mean():+.3f}"                  # text showing the average signed error, with an explicit +/- sign
    if outside:                                                # only mention clipped points if there actually were any
        note += f"\n{outside} crystal{'s' if outside > 1 else ''} beyond axis"   # pluralize "crystal(s)" correctly
    ax_hist.text(0.97, 0.95, note, transform=ax_hist.transAxes, ha="right",   # top-right corner, in axes-fraction coordinates
                 va="top", fontsize=9, color=INK_SOFT)
    ax_hist.set_xlabel("Prediction error, log10(GPa)")
    ax_hist.set_ylabel("Test crystals")
    ax_hist.set_title("Error distribution", fontsize=11, pad=10)

    # Right: does accuracy degrade at the extremes? A funnel shape here means
    # the model is only reliable in the middle of the range.
    style = mark_style(len(test))                               # marker size/alpha/linewidth tuned to the test point count
    ax_scatter.scatter(test.true_GPa, signed, color=BLUE, s=style["s"],   # x=true modulus, y=signed error, one point per crystal
                       alpha=style["alpha"], edgecolor=SURFACE,
                       linewidth=style["linewidth"])
    ax_scatter.axhline(0, color=MUTED, linestyle="--", linewidth=1)   # dashed horizontal reference line at zero error
    ax_scatter.set_xscale("log")                                # x-axis (true modulus) drawn in log10 spacing
    ax_scatter.set_xlabel(f"DFT {target} (GPa)")
    ax_scatter.set_ylabel("Prediction error, log10(GPa)")
    ax_scatter.set_title("Error vs material stiffness", fontsize=11, pad=10)

    for ax in (ax_hist, ax_scatter):    # apply the same finishing touches to both panels
        ax.grid(True, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"{target}: test-set residuals", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,                          # reuses this file's module docstring as --help text
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # preserves the docstring's manual line breaks
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], default="K_VRH")   # which physical property to evaluate
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data_full"))            # where the cached graph dataset lives
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))  # where checkpoints/outputs live
    parser.add_argument("--tag", default=None,
                        help="which run to evaluate; must match 02_train.py's "
                             "--tag. Defaults to --target.")
    args = parser.parse_args()                     # parses sys.argv into the args namespace above

    target = args.target
    tag = args.tag or target                       # fall back to the target name if no explicit tag was given
    print(f"=== Evaluating {tag} ===\n")

    model, dataset, normalizer, ckpt = load_model_and_data(    # rebuild everything needed to reproduce training-time predictions
        target, tag, args.data_dir, args.results_dir)
    df = collect_predictions(model, dataset, normalizer, ckpt["split"])   # run inference over every split, get one tidy DataFrame

    scores = metrics(df)                            # compute MAE/R2/etc. per split
    print(scores.round(4).to_string(index=False))   # print the summary table without the DataFrame index column

    # --- CSV ---------------------------------------------------------------
    pred_path = os.path.join(args.results_dir, f"predictions_{tag}.csv")    # per-crystal predictions output path
    df.sort_values(["split", "abs_error_log10"]).to_csv(pred_path, index=False)   # sorted by split, then by error; no index column

    metrics_path = os.path.join(args.results_dir, f"metrics_{tag}.csv")     # per-split summary metrics output path
    scores.to_csv(metrics_path, index=False)

    # --- Figures -----------------------------------------------------------
    history = pd.read_csv(os.path.join(args.results_dir, f"history_{tag}.csv"))   # per-epoch train/val loss+MAE, from 02_train.py
    plot_parity(df, target, os.path.join(args.results_dir, f"parity_{tag}.png"))
    plot_training(history, target, os.path.join(args.results_dir, f"training_{tag}.png"))
    plot_residuals(df, target, os.path.join(args.results_dir, f"residuals_{tag}.png"))

    print(f"\nWrote:")
    for name in [f"predictions_{tag}.csv", f"metrics_{tag}.csv",     # just the five filenames just written, for the console summary
                 f"parity_{tag}.png", f"training_{tag}.png",
                 f"residuals_{tag}.png"]:
        print(f"  {os.path.join(args.results_dir, name)}")           # print the full path of each

    # Worst predictions are worth eyeballing - they often reveal a systematic
    # weakness (a whole chemistry the model has never seen) rather than noise.
    worst = df[df.split == "test"].nlargest(5, "abs_error_log10")    # the 5 test crystals with the largest absolute error
    print("\nWorst test predictions:")
    print(worst[["material_id", "true_GPa", "pred_GPa", "abs_error_log10"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":   # only run main() when this file is executed directly, not when imported
    main()
