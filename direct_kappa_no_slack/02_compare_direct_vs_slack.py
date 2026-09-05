#!/usr/bin/env python3
"""
DIRECT KAPPA vs THE SLACK PIPELINE - the comparison, with error bars.
================================================================================

    python direct_kappa_no_slack/02_compare_direct_vs_slack.py

Run this after the Kaggle kernel has been fetched:

    python kaggle/build_direct_kappa_kernel.py
    python kaggle/run_direct_kappa_kernel.py

WHAT IS COMPARED
------------------
Four things, all on the SAME 835 held-out AFLOW crystals, all against AFLOW's
own tabulated kappa, which none of them ever saw:

  direct        this directory's single-head model. Graph -> log10(kappa).
                No moduli, no gamma, no Slack.
  control       37_train_gamma.py retrained in the SAME Kaggle kernel, on the
                same graph build and split. Graph -> G, K/G, gamma -> Slack.
                This is the in-session control, and it is what any claim
                rests on.
  round 9       the pre-existing checkpoints in results/cgcnn/. Same recipe,
                different session. A useful cross-check on whether the
                in-session control reproduced, and nothing more.
  tree          the composition-only random forest from step 43. Step 51
                established it beats round 9 at the low-kappa call
                (F1 0.646 vs 0.400, 95% CI [-0.352, -0.146]), so THIS is the
                bar to clear - not round 9.

THE HEADLINE IS THE SCREENING CALL, NOT THE MAE
--------------------------------------------------
This project screens; it does not regress. A model can win on average error
and still be useless at the only decision the pipeline makes, so precision,
recall and F1 of "kappa <= 1 W/m/K" come first and the error metrics come
after. Both are reported, and the mean and median of the log10 error are both
quoted because this project's error distribution is skewed enough that
quoting one alone has misled before.

TWO ERROR BARS, KEPT DISTINCT
-------------------------------
  seed spread       across the three seeds within one arm. Answers "if I
                    retrain, do I get this again?"
  paired bootstrap  resampling the test crystals, with the same indices
                    applied to both arms. Answers "does this hold on crystals
                    I have not seen?" This is the conservative one and the
                    only one that licenses a generalisation claim.

An effect smaller than the seed spread is not a result. That rule has already
cost this project one retracted headline and it is enforced in the printout.

THE VERDICT IS PRE-REGISTERED
-------------------------------
The decision table in README.md was written before the first run. It is
restated at the bottom of this script's output and the matching row is
selected mechanically from the numbers, so that whatever comes back cannot be
rationalised after the fact.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "data_dir": "data_full",
    "cache_file": "gamma_graphs.pt",
    "labels_file": "gamma_labels.csv",
    "direct_checkpoints": "direct_kappa_no_slack/models/model_direct_kappa_s*.pth",
    "control_checkpoints": "direct_kappa_no_slack/models/model_37_control_s*.pth",
    "round9_checkpoints": "results/cgcnn/model_37_r9_full_s*.pth",
    "tree_csv": "results/cgcnn/baseline_tree/csv/43_baseline_tree_predictions.csv",
    "tree_model": "random_forest",

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "direct_kappa_no_slack/results/csv",
    "png_dir": "direct_kappa_no_slack/results/png",

    # ---- the split -- MUST MATCH 37 and 01 ---------------------------------
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,

    # ---- the decision under test -------------------------------------------
    "low_kappa_threshold": 1.0,
    "deciles": [0.10, 0.20],
    "bootstrap_n": 10000,
    "bootstrap_seed": 0,

    # ---- runtime ------------------------------------------------------------
    "batch_size": 64,
    "num_workers": 0,
    "device": "cpu",
}
# =============================================================================

import glob                  # expanding the checkpoint globs
import os                    # paths and directory creation
import sys                   # sys.path manipulation and early exit

import torch                 # first: duplicate libiomp5 aborts if numpy/pandas load MKL first
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgcnn_scratch.data import collate_pool  # noqa: E402
from cgcnn_scratch.joint import (  # noqa: E402
    JointCrystalGraphConvNet, VectorNormalizer, JointGraphCacheData)

from importlib import import_module  # noqa: E402
_gamma = import_module("37_train_gamma")            # split_indices, kappa_full, derived_gamma
_adj = import_module("51_adjudicate_dft_shortlist")  # screening_call, recall_at, rank_scores
_direct = import_module("01_train_direct_kappa")     # DirectKappaCGCNN, ScalarDataset

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#3f9142", "#8e5bb5"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"

ARM_COLOR = {"direct": ORANGE, "control": BLUE, "round 9": PURPLE, "tree": GREEN}


def sequential_loader(dataset, indices, cfg):
    """A DataLoader that yields the given indices in order, never shuffled.

    Order matters here: every prediction is matched to a crystal id and to
    AFLOW's kappa for that crystal, so a shuffled sampler would silently
    misalign the comparison.
    """
    subset = torch.utils.data.Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=cfg["batch_size"],
                      sampler=SequentialSampler(subset),
                      collate_fn=collate_pool, num_workers=cfg["num_workers"])


def predict_direct(cfg, paths, dataset, test_idx):
    """Per-seed log10(kappa) from the single-head direct models.

    Returns a list of (tag, vector). Seeds are kept SEPARATE rather than
    averaged here, because the seed spread is one of the two error bars this
    script has to report; the ensemble mean is formed afterwards.
    """
    device = torch.device(cfg["device"])
    loader = sequential_loader(dataset, test_idx, cfg)
    out = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mc = ck["config"]
        sample_g, _, _ = dataset[0]
        model = _direct.DirectKappaCGCNN(
            sample_g[0].shape[-1], sample_g[1].shape[-1],
            atom_fea_len=mc["atom_fea_len"], n_conv=mc["n_conv"],
            h_fea_len=mc["h_fea_len"], n_shared_fc=mc["n_shared_fc"],
            n_head_fc=mc["n_head_fc"], head_fea_len=mc["head_fea_len"],
            dropout=mc["dropout"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        mean = ck["normalizer"]["mean"].to(device)
        std = ck["normalizer"]["std"].to(device)

        preds = []
        with torch.no_grad():
            for inputs, _t, _b in loader:
                atom_fea, nbr_fea, nbr_idx, crys_idx = inputs
                p = model(atom_fea.to(device), nbr_fea.to(device),
                          nbr_idx.to(device), [c.to(device) for c in crys_idx])
                preds.append((p * std + mean).cpu().numpy())
        out.append((mc["tag"], np.vstack(preds).ravel()))
    return out


def predict_slack(cfg, paths, dataset, test_idx, meta_arrays):
    """Per-seed Slack kappa from the 3-head models (control / round 9).

    Uses each model's OWN gamma head, which is the configuration those models
    were built for. Scoring them with a derived gamma instead would be a rigged
    comparison against a model that has a gamma head.
    """
    volume, natoms, density = meta_arrays
    device = torch.device(cfg["device"])
    loader = sequential_loader(dataset, test_idx, cfg)
    out = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mc = ck["config"]
        sample_g, _, _ = dataset[0]
        model = JointCrystalGraphConvNet(
            sample_g[0].shape[-1], sample_g[1].shape[-1],
            atom_fea_len=mc["atom_fea_len"], n_conv=mc["n_conv"],
            h_fea_len=mc["h_fea_len"], n_shared_fc=mc["n_shared_fc"],
            n_head_fc=mc["n_head_fc"], head_fea_len=mc["head_fea_len"],
            dropout=mc["dropout"], n_heads=mc["n_heads"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        # Restore the saved normaliser rather than refitting: it was fitted on
        # TRAIN only, and refitting here would leak test statistics.
        norm = VectorNormalizer.__new__(VectorNormalizer)
        norm.mean = ck["normalizer"]["mean"].to(device)
        norm.std = ck["normalizer"]["std"].to(device)

        rows = []
        with torch.no_grad():
            for inputs, _t, _b in loader:
                atom_fea, nbr_fea, nbr_idx, crys_idx = inputs
                p = model(atom_fea.to(device), nbr_fea.to(device),
                          nbr_idx.to(device), [c.to(device) for c in crys_idx])
                rows.append(norm.denorm(p).cpu().numpy())
        stacked = np.vstack(rows)
        log_G = stacked[:, 0]
        log_K = stacked[:, 1] + log_G
        gamma = (10.0 ** stacked[:, 2] if stacked.shape[1] >= 3
                 else _gamma.derived_gamma(log_K, log_G))
        kappa = _gamma.kappa_full(log_K, log_G, gamma, volume, natoms, density)
        out.append((mc["tag"], np.log10(kappa)))       # log10, to match the direct arm
    return out


def summarise(name, log_kappa_vectors, true_kappa, cfg):
    """Score each seed, then the log-space ensemble mean, and report the spread."""
    thr = cfg["low_kappa_threshold"]
    per_seed = []
    for tag, lk in log_kappa_vectors:
        kap = 10.0 ** lk
        call = _adj.screening_call(kap, true_kappa, thr)
        rank = _adj.rank_scores(kap, true_kappa, cfg)
        per_seed.append({"arm": name, "tag": tag, "member": "seed", **call, **rank})

    # Ensembling in LOG space, which is the convention every headline number in
    # this project is quoted at. Averaging linear kappa would be a different
    # estimator and would not be comparable to anything already published here.
    ens_log = np.mean(np.stack([v for _, v in log_kappa_vectors], axis=0), axis=0)
    ens_kappa = 10.0 ** ens_log
    call = _adj.screening_call(ens_kappa, true_kappa, thr)
    rank = _adj.rank_scores(ens_kappa, true_kappa, cfg)
    ens_row = {"arm": name, "tag": f"{name} ensemble", "member": "ensemble",
               **call, **rank}

    spread = {
        "f1_sd": float(np.std([r["f1"] for r in per_seed])),
        "f1_range": float(np.ptp([r["f1"] for r in per_seed])),
        "spearman_sd": float(np.std([r["spearman"] for r in per_seed])),
        "mae_sd": float(np.std([r["mae_log10"] for r in per_seed])),
    }
    return per_seed, ens_row, ens_log, spread


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  DIRECT KAPPA (no Slack)  vs  THE SLACK PIPELINE")
    print("=" * 78)
    print()

    # ---- data and split -----------------------------------------------------
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
    truth = meta.loc[ids]
    true_kappa = truth["kappa_agl"].to_numpy(dtype=float)
    meta_arrays = (truth["volume_m3"].to_numpy(dtype=float),
                   truth["n_sites"].to_numpy(dtype=float),
                   truth["density_g_cm3"].to_numpy(dtype=float))
    thr = cfg["low_kappa_threshold"]
    n_true_low = int(np.sum(true_kappa <= thr))
    print(f"  {len(ids)} held-out AFLOW crystals, {n_true_low} genuinely "
          f"<= {thr} W/m/K (split_seed={cfg['split_seed']})")
    print()

    # ---- gather the arms ----------------------------------------------------
    direct_paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["direct_checkpoints"])))
    if not direct_paths:
        sys.exit("ERROR: no direct-kappa checkpoints found at "
                 f"{cfg['direct_checkpoints']}.\n"
                 "Train them first:\n"
                 "  python kaggle/build_direct_kappa_kernel.py\n"
                 "  python kaggle/run_direct_kappa_kernel.py")
    control_paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["control_checkpoints"])))
    r9_paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["round9_checkpoints"])))
    print(f"  direct  : {len(direct_paths)} checkpoints")
    print(f"  control : {len(control_paths)} checkpoints (in-session)")
    print(f"  round 9 : {len(r9_paths)} checkpoints (previous session, cross-check)")
    if not control_paths:
        print("  WARNING: no in-session control. Round 9 is a cross-check, NOT a")
        print("           substitute - any claim below is weaker without it.")
    print()

    print("  running inference ...")
    arms, ens_logs, spreads = [], {}, {}

    direct_seeds = predict_direct(cfg, direct_paths, dataset, te)
    rows, ens, elog, spread = summarise("direct", direct_seeds, true_kappa, cfg)
    arms += rows + [ens]; ens_logs["direct"] = elog; spreads["direct"] = spread

    for name, paths in (("control", control_paths), ("round 9", r9_paths)):
        if not paths:
            continue
        seeds = predict_slack(cfg, paths, dataset, te, meta_arrays)
        rows, ens, elog, spread = summarise(name, seeds, true_kappa, cfg)
        arms += rows + [ens]; ens_logs[name] = elog; spreads[name] = spread

    # ---- the tree baseline, the bar that actually matters -------------------
    tree = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["tree_csv"]))
    tt = tree[(tree["source"] == "aflow") & (tree["split"] == "test")]
    tt = tt.set_index(tt["material_id"].astype(str))
    if set(tt.index) == set(ids):
        tt = tt.loc[ids]
        tm = cfg["tree_model"]
        t_lK = tt[f"K_VRH_pred_log10_{tm}"].to_numpy(dtype=float)
        t_lG = tt[f"G_VRH_pred_log10_{tm}"].to_numpy(dtype=float)
        t_gam = 10.0 ** tt[f"gamma_pred_log10_{tm}"].to_numpy(dtype=float)
        t_kappa = _gamma.kappa_full(t_lK, t_lG, t_gam, *meta_arrays)
        ens_logs["tree"] = np.log10(t_kappa)
        arms.append({"arm": "tree", "tag": f"tree ({tm})", "member": "ensemble",
                     **_adj.screening_call(t_kappa, true_kappa, thr),
                     **_adj.rank_scores(t_kappa, true_kappa, cfg)})
    else:
        print("  NOTE: tree split differs from this one - tree row omitted.")

    scores = pd.DataFrame(arms)
    scores.to_csv(os.path.join(csv_dir, "02_direct_vs_slack_scores.csv"), index=False)

    # ---- report -------------------------------------------------------------
    print()
    print("  " + "-" * 74)
    print(f"  THE SCREENING CALL: kappa <= {thr} W/m/K on {len(ids)} crystals")
    print("  " + "-" * 74)
    print(f"  {'arm / run':<30}{'called':>8}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'prec':>7}{'rec':>7}{'F1':>7}")
    for _, r in scores.iterrows():
        star = "  <-" if r["member"] == "ensemble" else ""
        print(f"  {r['tag']:<30}{r['n_called']:>8}{r['tp']:>5}{r['fp']:>5}"
              f"{r['fn']:>5}{r['precision']:>7.2f}{r['recall']:>7.2f}"
              f"{r['f1']:>7.2f}{star}")

    print()
    print("  " + "-" * 74)
    print("  RANKING QUALITY")
    print("  " + "-" * 74)
    print(f"  {'arm / run':<30}{'spear':>8}{'rec@10':>8}{'rec@20':>8}{'MAE':>8}{'med':>8}")
    for _, r in scores.iterrows():
        print(f"  {r['tag']:<30}{r['spearman']:>8.3f}{r['recall10']:>8.1f}"
              f"{r['recall20']:>8.1f}{r['mae_log10']:>8.3f}"
              f"{r['median_ae_log10']:>8.3f}")

    print()
    print("  " + "-" * 74)
    print("  SEED SPREAD - an effect smaller than this is not a result")
    print("  " + "-" * 74)
    for arm, s in spreads.items():
        print(f"    {arm:<10} F1 sd {s['f1_sd']:.3f} (range {s['f1_range']:.3f})   "
              f"spearman sd {s['spearman_sd']:.3f}   MAE sd {s['mae_sd']:.3f}")

    # ---- paired bootstrap: direct vs the in-session control -----------------
    comparisons = []
    for other in ("control", "round 9", "tree"):
        if other not in ens_logs:
            continue
        a = 10.0 ** ens_logs["direct"]
        b = 10.0 ** ens_logs[other]
        mean_d, lo, hi = _adj.bootstrap_f1_gap(a, b, true_kappa, cfg)
        decisive = (lo > 0) or (hi < 0)
        comparisons.append({"vs": other, "d_f1": mean_d, "ci_lo": lo, "ci_hi": hi,
                            "decisive": decisive})
    comps = pd.DataFrame(comparisons)
    comps.to_csv(os.path.join(csv_dir, "02_direct_vs_slack_bootstrap.csv"), index=False)

    print()
    print("  " + "-" * 74)
    print(f"  PAIRED BOOTSTRAP on F1, direct minus other ({cfg['bootstrap_n']} resamples)")
    print("  " + "-" * 74)
    for _, c in comps.iterrows():
        verdict = ("EXCLUDES zero" if c["decisive"] else "crosses zero")
        print(f"    vs {c['vs']:<10}{c['d_f1']:>+8.3f}   95% CI "
              f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]   {verdict}")

    # ---- the pre-registered verdict -----------------------------------------
    print()
    print("  " + "=" * 74)
    print("  PRE-REGISTERED VERDICT (decision table from README.md)")
    print("  " + "=" * 74)
    f1 = {r["arm"]: r["f1"] for _, r in scores[scores["member"] == "ensemble"].iterrows()}
    vs_ctrl = comps[comps["vs"] == "control"]
    vs_tree = comps[comps["vs"] == "tree"]

    if "tree" in f1 and f1.get("direct", 0) < f1["tree"] and len(vs_tree) \
            and bool(vs_tree.iloc[0]["decisive"]):
        print("    ROW: 'loses to the tree baseline'")
        print("    A composition-only model with no structure beats the graph network.")
        print("    This is NOT about Slack - it points at underfitting on AFLOW, which")
        print("    steps 43/44 already found for round 9. Fix capacity before drawing")
        print("    any conclusion about the formula.")
    elif len(vs_ctrl) == 0:
        print("    No in-session control was available, so no row applies. Re-run the")
        print("    Kaggle kernel with both arms before claiming anything.")
    elif bool(vs_ctrl.iloc[0]["decisive"]) and vs_ctrl.iloc[0]["d_f1"] > 0:
        print("    ROW: 'beats the control, CI excluding zero'")
        print("    The Slack detour is costing real screening performance. Retire it")
        print("    from the AFLOW branch and re-run the GNoME screen without it.")
    elif vs_ctrl.iloc[0]["d_f1"] > 0:
        print("    ROW: 'beats the control but the CI crosses zero'")
        print(f"    Suggestive, not decisive on {len(ids)} crystals. Report as a")
        print("    mechanism, not a win - the discipline rounds 4-7 were held to.")
    elif abs(vs_ctrl.iloc[0]["d_f1"]) <= spreads.get("direct", {}).get("f1_sd", 0):
        print("    ROW: 'ties the control'")
        print("    Slack is neutral once its inputs are predicted. Keep it: it is")
        print("    interpretable and it extrapolates off-dataset where a direct")
        print("    model cannot - which is what the GNoME screen needs.")
    else:
        print("    ROW: 'loses to the control'")
        print("    The physics carries real information the graph alone does not")
        print("    supply. That is a positive result for the pipeline and closes")
        print("    this line of enquiry.")
    print()
    print(f"    All of the above rests on {len(ids)} crystals with "
          f"{n_true_low} true positives,")
    print("    scored against AFLOW's model-derived kappa, which is not experiment.")

    # ---- figure -------------------------------------------------------------
    ens = scores[scores["member"] == "ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    ax = axes[0]
    ax.bar(range(len(ens)), ens["f1"],
           color=[ARM_COLOR.get(a, MUTED) for a in ens["arm"]], width=0.6)
    ax.set_xticks(range(len(ens)))
    ax.set_xticklabels(ens["arm"], fontsize=9, color=INK_SOFT)
    ax.set_ylabel(f"F1 of the $\\kappa\\leq{thr}$ call", color=INK_SOFT)
    ax.set_title(f"The decision the pipeline makes ({len(ids)} crystals)",
                 color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    for arm, lk in ens_logs.items():
        ax.scatter(true_kappa, 10.0 ** lk, s=12, alpha=0.5,
                   color=ARM_COLOR.get(arm, MUTED), label=arm, edgecolor="none")
    lim = [max(1e-3, true_kappa.min()), true_kappa.max()]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.axhline(thr, color=INK_SOFT, lw=0.9, ls=":")
    ax.axvline(thr, color=INK_SOFT, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AFLOW $\\kappa_L$ (W/m/K)", color=INK_SOFT)
    ax.set_ylabel("predicted $\\kappa_L$", color=INK_SOFT)
    ax.set_title("Dotted lines mark the screening threshold", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle("Predicting $\\kappa_L$ directly vs routing moduli through Slack",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "02_direct_vs_slack.png"), dpi=150,
                facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    print(f"    {cfg['csv_dir']}/02_direct_vs_slack_scores.csv")
    print(f"    {cfg['csv_dir']}/02_direct_vs_slack_bootstrap.csv")
    print(f"    {cfg['png_dir']}/02_direct_vs_slack.png")


if __name__ == "__main__":
    main()
