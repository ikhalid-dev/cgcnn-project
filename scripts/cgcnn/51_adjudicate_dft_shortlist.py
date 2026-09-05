#!/usr/bin/env python3
"""
STEP 51 - Which model should the DFT shortlist believe: the tree, or round 9?
================================================================================

    python scripts/cgcnn/51_adjudicate_dft_shortlist.py

THE STANDOFF
--------------
Step 46 left this project with a contradiction it could not settle:

    on the 218 GNoME crystals already queued for DFT
        old pipeline (matbench, derived gamma)   218/218 call kappa <= 1
        composition-only tree baseline (AFLOW)   191/218
        round-9 CGCNN (AFLOW, trained gamma)      19/218

The composition-only tree sides with the shortlist; the model built to improve
on it rejects almost all of it. 46's own docstring says GNoME has no ground
truth, so nothing there can adjudicate - and it correctly refused to pick a
winner. DFT time is the scarcest resource this project has, and it was about
to be spent on a list that one of its own models rejects.

WHAT MAKES THIS DECIDABLE
---------------------------
Both models were trained on AFLOW, on the SAME 70/15/15 split with
split_seed=42, and AFLOW's held-out test crystals carry a real tabulated
kappa (agl_thermal_conductivity_300K) that neither model ever saw. That is
835 crystals of genuine ground truth for exactly the question at issue.

The split identity is ASSERTED here, not assumed: 43 (trees) and 37 (round 9)
each call their own `split_indices`, and if those ever diverge the two models
would be scored on different crystals and the comparison would be silently
meaningless. The script exits rather than report a number it cannot defend.

WHAT IS COMPARED, AND WHY THE SCREENING CALL AND NOT MAE
----------------------------------------------------------
This project screens; it does not regress. So the headline is the low-kappa
CALL - precision, recall and F1 of "kappa <= 1 W/m/K" against AFLOW's own
kappa - not the average error. A model can win on MAE and still be useless at
the only decision the pipeline actually makes. Both are reported, with the
call first.

Three predicted-kappa variants per model, so the gamma question stays visible:

    derived gamma    gamma from predicted K/G via the Poisson relation
    predicted gamma  gamma from the model's own gamma head / tree
    oracle gamma     AFLOW's tabulated gamma, with predicted K and G

The oracle is not a model result. It is the ceiling, and step 49 showed that
AFLOW's gamma is itself badly wrong against literature values - so read the
oracle as "the best this pipeline could do if it nailed AFLOW's gamma", not
as "the truth".

THE VERDICT IS APPLIED, NOT JUST STATED
-----------------------------------------
The last section takes whichever model the 835 crystals support and reports
what the 218-crystal shortlist looks like under it. That is the actual
deliverable: a defensible answer to "should we run this DFT or not".
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "data_dir": "data_full",
    "cache_file": "gamma_graphs.pt",              # the AFLOW crystal graphs
    "labels_file": "gamma_labels.csv",            # gid, K_VRH, G_VRH, gamma, kappa_agl, density, volume
    "r9_checkpoints": "results/cgcnn/model_37_r9_full_s*.pth",   # the 3-seed round-9 ensemble
    "tree_csv": "results/cgcnn/baseline_tree/csv/43_baseline_tree_predictions.csv",
    "tree_model": "random_forest",                # 43 found RF best on AFLOW K; "xgboost" also available
    "dft_shortlist_csv": "/Users/mac/Desktop/dft_candidates.csv",   # the 218 crystals queued for DFT
    "r9_gnome_screen": "results/cgcnn/39_gnome_screen_all_gamma.csv",  # round-9's own GNoME kappa

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/adjudication/csv",
    "png_dir": "results/cgcnn/adjudication/png",

    # ---- the split (must match 37 and 43 exactly - asserted, not trusted) ----
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,

    # ---- the decision under test -------------------------------------------
    "low_kappa_threshold": 1.0,   # W/m/K - the threshold the whole screen is built on
    "deciles": [0.10, 0.20],      # recall@k fractions, reported alongside the call
    "bootstrap_n": 10000,         # paired resamples for the CI on the F1 difference
    "bootstrap_seed": 0,

    # ---- inference ----------------------------------------------------------
    "batch_size": 64,
    "num_workers": 0,             # 0 keeps ordering deterministic and avoids fork overhead
    "device": "cpu",              # 835 crystals; a GPU would be pure setup cost
}
# =============================================================================

import os                    # path joining and directory creation
import sys                   # sys.path manipulation and early exit
import glob                  # expanding the checkpoint glob

import torch                 # first: duplicate libiomp5 aborts if numpy/pandas load MKL first
import torch.nn as nn        # noqa: F401
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

from cgcnn_scratch.data import collate_pool  # noqa: E402
from cgcnn_scratch.joint import (  # noqa: E402
    JointCrystalGraphConvNet, VectorNormalizer, JointGraphCacheData)

from importlib import import_module  # noqa: E402
_gamma = import_module("37_train_gamma")     # split_indices, kappa_full, derived_gamma

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


# -----------------------------------------------------------------------------
#  Round-9 inference
# -----------------------------------------------------------------------------
def r9_predictions(cfg, dataset, test_idx, ids):
    """Run the 3-seed round-9 ensemble over the AFLOW test split.

    Returns (log_K, log_G, log_gamma), each the ensemble MEAN over seeds, in
    the order of `ids`.

    Ensembling in LOG space (averaging the head outputs before exponentiating)
    is what 32_ensemble_joint.py does, and it is the convention every headline
    number in this project is quoted at. Averaging in linear kappa instead
    would be a different estimator and would not be comparable to anything
    already published here.

    A SequentialSampler is used rather than 37's SubsetRandomSampler because
    this needs predictions aligned to crystal ids, not shuffled batches.
    """
    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["r9_checkpoints"])))
    if not paths:
        sys.exit(f"ERROR: no round-9 checkpoints matched {cfg['r9_checkpoints']}")

    # Only the test rows, in a fixed order, so predictions line up with `ids`.
    subset = torch.utils.data.Subset(dataset, list(test_idx))
    loader = DataLoader(subset, batch_size=cfg["batch_size"],
                        sampler=SequentialSampler(subset),
                        collate_fn=collate_pool, num_workers=cfg["num_workers"])

    device = torch.device(cfg["device"])
    per_seed = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mc = ck["config"]
        sample, _, _ = dataset[0]
        model = JointCrystalGraphConvNet(
            sample[0].shape[-1], sample[1].shape[-1],
            atom_fea_len=mc["atom_fea_len"], n_conv=mc["n_conv"],
            h_fea_len=mc["h_fea_len"], n_shared_fc=mc["n_shared_fc"],
            n_head_fc=mc["n_head_fc"], head_fea_len=mc["head_fea_len"],
            dropout=mc["dropout"], n_heads=mc["n_heads"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        # The normalizer was fitted on TRAIN only and saved with the model;
        # rebuilding it from this split would leak test statistics into the
        # denormalisation, so the saved one is restored verbatim.
        norm = VectorNormalizer.__new__(VectorNormalizer)
        norm.mean = ck["normalizer"]["mean"].to(device)
        norm.std = ck["normalizer"]["std"].to(device)

        out = []
        with torch.no_grad():
            for inputs, _target, _bid in loader:
                atom_fea, nbr_fea, nbr_idx, crys_idx = inputs
                pred = model(atom_fea.to(device), nbr_fea.to(device),
                             nbr_idx.to(device), [c.to(device) for c in crys_idx])
                out.append(norm.denorm(pred).cpu().numpy())
        per_seed.append(np.vstack(out))
        print(f"    {os.path.basename(path)}: {per_seed[-1].shape[0]} crystals")

    stacked = np.mean(np.stack(per_seed, axis=0), axis=0)   # (n_test, n_heads)
    log_G = stacked[:, 0]
    log_ratio = stacked[:, 1]                                # log10(K/G)
    log_K = log_ratio + log_G
    log_gamma = stacked[:, 2] if stacked.shape[1] >= 3 else None
    return log_K, log_G, log_gamma, len(paths)


# -----------------------------------------------------------------------------
#  Scoring: the screening call comes first
# -----------------------------------------------------------------------------
def screening_call(pred_kappa, true_kappa, threshold):
    """Precision / recall / F1 of 'this crystal is low-kappa', against truth.

    This is the decision the pipeline actually makes. A confusion matrix is
    returned alongside because precision and recall alone hide whether a model
    is calling almost everything low (high recall, useless) or almost nothing
    (high precision, useless) - and both failure modes have shown up in this
    project already.
    """
    p = np.asarray(pred_kappa) <= threshold
    t = np.asarray(true_kappa) <= threshold
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_called": tp + fp,
            "n_true_low": tp + fn, "precision": precision, "recall": recall, "f1": f1}


def recall_at(true_vals, pred_vals, frac):
    """Fraction of the true bottom `frac` that the predicted bottom `frac` contains."""
    n = max(1, int(round(frac * len(true_vals))))
    truth = set(np.argsort(np.asarray(true_vals))[:n])
    picked = set(np.argsort(np.asarray(pred_vals))[:n])
    return len(truth & picked) / n, n


def rank_scores(pred_kappa, true_kappa, cfg):
    """Spearman, recall@k and the log10 error, for one prediction vector."""
    out = {"spearman": spearmanr(true_kappa, pred_kappa).correlation}
    for frac in cfg["deciles"]:
        r, n_bin = recall_at(true_kappa, pred_kappa, frac)
        out[f"recall{int(frac * 100)}"] = 100.0 * r
        out[f"n_bin{int(frac * 100)}"] = n_bin
    d = np.abs(np.log10(pred_kappa) - np.log10(true_kappa))
    out["mae_log10"] = float(np.mean(d))
    out["median_ae_log10"] = float(np.median(d))
    return out


def bootstrap_f1_gap(pred_a, pred_b, true_kappa, cfg):
    """Paired bootstrap CI on F1(a) - F1(b), the difference that decides this.

    Paired: the same resampled crystals are scored for both models on every
    draw, so the interval reflects disagreement between the models rather than
    the variance of each one separately.
    """
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(true_kappa)
    idx = rng.integers(0, n, (cfg["bootstrap_n"], n))
    thr = cfg["low_kappa_threshold"]
    d = np.empty(cfg["bootstrap_n"], dtype=float)
    for j, row in enumerate(idx):
        t = true_kappa[row]
        d[j] = (screening_call(pred_a[row], t, thr)["f1"]
                - screening_call(pred_b[row], t, thr)["f1"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 51 - adjudicating the DFT shortlist on AFLOW's real kappa")
    print("=" * 78)
    print()

    # ---- data and the split -------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])
    meta = pd.read_csv(os.path.join(data_dir, cfg["labels_file"])).set_index("gid")
    log_K = np.log10(meta.K_VRH.values)
    log_G = np.log10(meta.G_VRH.values)
    log_gamma = np.log10(meta.gamma.values)
    targets = {i: (log_G[k], log_K[k] - log_G[k], log_gamma[k])
               for k, i in enumerate(meta.index)}
    dataset = JointGraphCacheData(os.path.join(data_dir, cfg["cache_file"]), targets)

    tr, va, te = _gamma.split_indices(len(dataset), cfg["train_ratio"],
                                      cfg["val_ratio"], cfg["split_seed"])
    ids = [dataset.ids[i] for i in te]
    print(f"  AFLOW: {len(dataset)} crystals, split "
          f"{len(tr)}/{len(va)}/{len(te)} (seed {cfg['split_seed']})")

    # The assertion that makes this comparison meaningful at all.
    tree = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["tree_csv"]))
    tree_test = tree[(tree["source"] == "aflow") & (tree["split"] == "test")]
    tree_test = tree_test.set_index(tree_test["material_id"].astype(str))
    if set(tree_test.index) != set(ids):
        sys.exit("ERROR: the tree baseline's AFLOW test split and round 9's do not "
                 "match. The two models would be scored on different crystals and "
                 "the comparison would be meaningless. Refusing to continue.")
    print(f"  split identity CHECKED: tree and round 9 share all {len(ids)} test crystals")
    print()

    truth = meta.loc[ids]
    true_kappa = truth["kappa_agl"].to_numpy(dtype=float)
    true_gamma = truth["gamma"].to_numpy(dtype=float)
    volume = truth["volume_m3"].to_numpy(dtype=float)
    natoms = truth["n_sites"].to_numpy(dtype=float)
    density = truth["density_g_cm3"].to_numpy(dtype=float)
    n_true_low = int(np.sum(true_kappa <= cfg["low_kappa_threshold"]))
    print(f"  ground truth: AFLOW kappa on {len(ids)} held-out crystals, "
          f"{n_true_low} of them <= {cfg['low_kappa_threshold']} W/m/K")
    print()

    # ---- round-9 predictions ------------------------------------------------
    print("  running round-9 inference ...")
    r9_lK, r9_lG, r9_lgamma, n_seeds = r9_predictions(cfg, dataset, te, ids)
    print(f"    ensembled {n_seeds} seeds in log space")

    # ---- tree predictions, aligned to the same id order ---------------------
    tm = cfg["tree_model"]
    tt = tree_test.loc[ids]
    tree_lK = tt[f"K_VRH_pred_log10_{tm}"].to_numpy(dtype=float)
    tree_lG = tt[f"G_VRH_pred_log10_{tm}"].to_numpy(dtype=float)
    tree_lgamma = tt[f"gamma_pred_log10_{tm}"].to_numpy(dtype=float)
    print(f"  tree baseline: {tm}, {len(tt)} crystals aligned")
    print()

    # ---- three kappa variants per model -------------------------------------
    def kappas(lK, lG, lgamma):
        """derived / predicted / oracle gamma, all through the same Slack call."""
        out = {}
        out["derived"] = _gamma.kappa_full(lK, lG, _gamma.derived_gamma(lK, lG),
                                           volume, natoms, density)
        if lgamma is not None and np.isfinite(lgamma).all():
            out["predicted"] = _gamma.kappa_full(lK, lG, 10.0 ** lgamma,
                                                 volume, natoms, density)
        out["oracle"] = _gamma.kappa_full(lK, lG, true_gamma, volume, natoms, density)
        return out

    variants = {"round 9 (CGCNN)": kappas(r9_lK, r9_lG, r9_lgamma),
                f"tree ({tm})": kappas(tree_lK, tree_lG, tree_lgamma)}

    # ---- report: the screening call FIRST -----------------------------------
    thr = cfg["low_kappa_threshold"]
    rows = []
    print("  " + "-" * 74)
    print(f"  THE SCREENING CALL: kappa <= {thr} W/m/K, vs AFLOW's own kappa")
    print("  " + "-" * 74)
    print(f"  {'model / gamma source':<34}{'called':>8}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'prec':>7}{'rec':>7}{'F1':>7}")
    for model, vs in variants.items():
        for gname, kap in vs.items():
            c = screening_call(kap, true_kappa, thr)
            r = rank_scores(kap, true_kappa, cfg)
            rows.append({"model": model, "gamma_source": gname, **c, **r})
            print(f"  {model + ' / ' + gname:<34}{c['n_called']:>8}{c['tp']:>6}"
                  f"{c['fp']:>6}{c['fn']:>6}{c['precision']:>7.2f}"
                  f"{c['recall']:>7.2f}{c['f1']:>7.2f}")
    print(f"  (of {len(ids)} crystals, {n_true_low} are genuinely low-kappa)")

    print()
    print("  " + "-" * 74)
    print("  RANKING QUALITY on the same crystals")
    print("  " + "-" * 74)
    print(f"  {'model / gamma source':<34}{'spear':>8}{'rec@10':>8}{'rec@20':>8}"
          f"{'MAE':>8}{'med':>8}")
    for r in rows:
        print(f"  {r['model'] + ' / ' + r['gamma_source']:<34}{r['spearman']:>8.3f}"
              f"{r['recall10']:>8.1f}{r['recall20']:>8.1f}"
              f"{r['mae_log10']:>8.3f}{r['median_ae_log10']:>8.3f}")

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(csv_dir, "51_adjudication_scores.csv"), index=False)

    # ---- the head-to-head that decides it -----------------------------------
    # Each model at its OWN best configuration - which for round 9 means its
    # trained gamma head (the thing it was built for) and for the tree its own
    # gamma column. Comparing round 9's derived-gamma variant against the tree's
    # predicted-gamma one would be a rigged comparison.
    r9_best = "predicted" if "predicted" in variants["round 9 (CGCNN)"] else "derived"
    tr_best = "predicted" if "predicted" in variants[f"tree ({tm})"] else "derived"
    a = variants["round 9 (CGCNN)"][r9_best]
    b = variants[f"tree ({tm})"][tr_best]
    print()
    print("  " + "-" * 74)
    print(f"  HEAD TO HEAD: round 9 ({r9_best} gamma) vs tree ({tr_best} gamma)")
    print("  " + "-" * 74)
    mean_d, lo, hi = bootstrap_f1_gap(a, b, true_kappa, cfg)
    f1_a = screening_call(a, true_kappa, thr)["f1"]
    f1_b = screening_call(b, true_kappa, thr)["f1"]
    print(f"    F1  round 9 {f1_a:.3f}   tree {f1_b:.3f}")
    print(f"    paired bootstrap on F1(round 9) - F1(tree), {cfg['bootstrap_n']} resamples:")
    print(f"      {mean_d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    decisive = (lo > 0) or (hi < 0)
    winner = "round 9" if mean_d > 0 else "the tree baseline"
    if decisive:
        print(f"    -> the interval EXCLUDES zero: {winner} is genuinely better "
              f"at this call.")
    else:
        print("    -> the interval CROSSES zero: on this evidence the two models "
              "are not")
        print("       distinguishable at the low-kappa call, and neither can claim "
              "the shortlist.")

    pd.DataFrame([{"f1_round9": f1_a, "f1_tree": f1_b, "d_f1": mean_d,
                   "ci_lo": lo, "ci_hi": hi, "decisive": decisive,
                   "winner": winner if decisive else "neither",
                   "n_crystals": len(ids), "n_true_low": n_true_low}]).to_csv(
        os.path.join(csv_dir, "51_head_to_head.csv"), index=False)

    # ---- apply the verdict to the 218 ---------------------------------------
    print()
    print("  " + "-" * 74)
    print("  WHAT THIS MEANS FOR THE DFT SHORTLIST")
    print("  " + "-" * 74)
    sl_path = cfg["dft_shortlist_csv"]
    if os.path.exists(sl_path):
        sl = pd.read_csv(sl_path, dtype={"material_id": str})
        n = len(sl)
        print(f"    shortlist: {n} crystals ({os.path.basename(sl_path)})")
        for col, label in (("kappa_CGCNN", "old pipeline (matbench, derived gamma)"),
                           ("Kappa_baseline", "tree baseline (AFLOW)"),
                           ("Kappa_baseline_rf", "tree baseline / RF"),
                           ("Kappa_baseline_xgb", "tree baseline / XGB")):
            if col in sl.columns:
                k = pd.to_numeric(sl[col], errors="coerce")
                print(f"      {label:<42}{int((k <= thr).sum()):>4}/{n} call kappa <= {thr}")
        sl.to_csv(os.path.join(csv_dir, "51_shortlist_snapshot.csv"), index=False)
    else:
        print(f"    shortlist not found at {sl_path} - skipping this section.")

    print()
    if decisive:
        print(f"    VERDICT: on {len(ids)} AFLOW crystals with real kappa, "
              f"{winner} wins the")
        print(f"    low-kappa call with the 95% CI excluding zero. Trust its "
              f"reading of the")
        print("    shortlist over the other's.")
    else:
        print(f"    VERDICT: {len(ids)} crystals cannot separate the two models on "
              f"the low-kappa")
        print("    call. Do NOT spend DFT time on the strength of either one alone; "
              "the")
        print("    defensible list is where they AGREE, which is the intersection "
              "below.")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    ax = axes[0]
    labels = [f"{r['model'].split(' ')[0]}\n{r['gamma_source']}" for r in rows]
    f1s = [r["f1"] for r in rows]
    colors = [BLUE if r["model"].startswith("round") else GREEN for r in rows]
    ax.bar(range(len(rows)), f1s, color=colors, width=0.65)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7.5, color=INK_SOFT)
    ax.set_ylabel(f"F1 of the $\\kappa\\leq{thr}$ call", color=INK_SOFT)
    ax.set_title(f"The decision the pipeline makes ({len(ids)} crystals)",
                 color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    ax.scatter(true_kappa, a, s=14, color=BLUE, alpha=0.55,
               label=f"round 9 ({r9_best})", edgecolor="none")
    ax.scatter(true_kappa, b, s=14, color=GREEN, alpha=0.55,
               label=f"tree ({tr_best})", edgecolor="none")
    lim = [max(1e-3, min(true_kappa.min(), a.min(), b.min())),
           max(true_kappa.max(), a.max(), b.max())]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.axhline(thr, color=ORANGE, lw=1.0, ls=":")
    ax.axvline(thr, color=ORANGE, lw=1.0, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AFLOW $\\kappa_L$ (W/m/K)", color=INK_SOFT)
    ax.set_ylabel("predicted $\\kappa_L$", color=INK_SOFT)
    ax.set_title("Dotted lines mark the screening threshold", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle("Adjudicating the DFT shortlist on AFLOW's real $\\kappa_L$",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "51_adjudication.png"), dpi=150, facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    for f in ("51_adjudication_scores.csv", "51_head_to_head.csv"):
        print(f"    {os.path.join(cfg['csv_dir'], f)}")
    print(f"    {os.path.join(cfg['png_dir'], '51_adjudication.png')}")


if __name__ == "__main__":
    main()
