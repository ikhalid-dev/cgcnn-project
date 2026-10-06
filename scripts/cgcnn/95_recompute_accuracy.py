"""
STEP 95, PART 2a - recompute every ACCURACY metric from the saved predictions,
and check each one against the number that was stored when the run was scored.

WHAT "FROM SCRATCH" MEANS HERE
------------------------------
Every formula below is written again from its textbook definition. None is
imported from the scripts that produced the stored numbers. If an original
script had a bug, the two numbers disagree and the row is flagged. Importing
the old function would have copied the bug and passed it.

The one exception is a rule that a number is DEFINED by, not computed by:
e.g. "quintile edges from np.quantile, ties go to the upper bin". Those
conventions are copied on purpose and named where they are used.

WHAT IS COVERED (part 2a = accuracy of continuous predictions)
-------------------------------------------------------------
    CGCNN moduli, single + 3-ensemble     metrics_summary.csv, metrics_*.csv, summary_*.json
    ALIGNN moduli, 4 singles + 3-ensemble 18_alignn_ensemble_scores.csv (incl. its bootstrap CIs)
    joint CGCNN, 32 runs                  summary_joint_*.json, joint_comparison.csv
    separate-model kappa error budget     60_residual_corr_table.csv, joint_comparison.csv
    tree baselines                        43_baseline_tree_scores.csv
    direct-kappa CGCNN, 6 runs            summary_direct_kappa_*.json, direct_vs_control_summary.csv,
                                          aflow_mbrecipe_summary.csv
    direct ALIGNN (coherent-heat), OOF    25_oof_summary.csv
    ensembles                             "is the ensemble file really the mean of its members?"

Part 2b (next) covers the SCREENING metrics: accuracy / precision / recall /
confusion matrices of steps 47, 48, 50, 63, 65, 82, 87, 89 and direct-kappa 02.

OUTPUT
------
    results/cgcnn/95_recomputed_accuracy.csv   one row per (model, run, split, metric)
    results/cgcnn/95_recomputed_accuracy.png   stored vs recomputed, every compared value

status column:
    match        same number (within float precision, or within the stored rounding)
    close        differs by less than 0.1% - same number, different arithmetic path
    MISMATCH     differs by more than 0.1% - investigate
    new          never stored before (e.g. MSE, R2 or Spearman for a run that only had MAE)

RUN (no torch needed)
---------------------
    OMP_NUM_THREADS=1 ~/miniconda3/envs/infer_env/bin/python scripts/cgcnn/95_recompute_accuracy.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")                    # draw to a file, never open a window
import matplotlib.pyplot as plt

# os.path.dirname(x) = "the folder that holds x"; three of them climb from
# scripts/cgcnn/95_... up to the project folder.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
ALI = os.path.join(ROOT, "results", "alignn")
DK = os.path.join(ROOT, "direct_kappa_no_slack")
KAG = os.path.join(ROOT, "kaggle", "output_aflow_recipe")
COH = os.path.expanduser("~/Desktop/a new project")      # the coherent-heat project
OUT_CSV = os.path.join(RES, "95_recomputed_accuracy.csv")
OUT_PNG = os.path.join(RES, "95_recomputed_accuracy.png")

MB = "matbench split_seed 42 (train 7,691 / val 1,648 / test 1,648)"
AF = "AFLOW split (train 3,894 / val 834 / test 835)"

ROWS = []      # a Python list; every metric value is appended to it as one dict


# =============================================================================
#  1. Recording a value
# =============================================================================
def record(family, model, run, trained_on, test_set, split, target, metric, value,
           n=None, stored=np.nan, stored_in="", decimals=None, note=""):
    """Append one metric value (and, if there is one, the number stored for it).

    decimals = how many decimals the stored table was ROUNDED to, when it was
    rounded (e.g. joint_comparison's baseline rows are 4-decimal). None = work
    it out from the stored number itself.
    """
    # A "dict" is a set of name -> value pairs. ** below would unpack it; here
    # we simply build one per value and add it to the list.
    ROWS.append({"family": family, "model": model, "run": run, "trained_on": trained_on,
                 "test_set": test_set, "split": split, "target": target, "metric": metric,
                 "n": n, "recomputed": float(value), "stored": float(stored),
                 "stored_in": stored_in, "stored_decimals": decimals, "note": note})


def record_many(values, stored=None, stored_in="", decimals=None, **where):
    """Record a whole dict of metrics at once. stored = dict of the same names."""
    stored = stored or {}                # "or {}" = use an empty dict if None was passed
    for metric, value in values.items():
        if metric == "n":
            continue                     # n travels with every row instead
        record(metric=metric, value=value, n=values.get("n"),
               stored=stored.get(metric, np.nan), stored_in=stored_in if metric in stored else "",
               decimals=decimals, **where)


def decimals_of(x):
    """How many decimals a stored number was written with: 0.0696 -> 4.

    repr() gives the SHORTEST text that reads back as the same float, so a
    value that was rounded to 4 decimals before being saved comes back with
    at most 4, while a full-precision value comes back with ~17.
    """
    s = repr(float(x))
    if "e" in s or "." not in s:
        return 17
    return len(s.split(".")[1])


# =============================================================================
#  2. The metrics, written from their definitions
# =============================================================================
def regression(true, pred):
    """Accuracy of continuous predictions. Inputs are numpy arrays, same order.

    err = pred - true, crystal by crystal
        MAE    mean |err|
        MSE    mean err^2
        RMSE   sqrt(MSE)
        R2     1 - sum(err^2) / sum((true - mean(true))^2)
               the share of the spread in the truth that the model explains
        bias   mean err (signed: > 0 means the model over-predicts on average)
    """
    err = pred - true
    ae = np.abs(err)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {"n": len(true),
            "MAE": ae.mean(),
            "MSE": np.mean(err ** 2),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "R2": 1.0 - ss_res / ss_tot,
            "median_AE": np.median(ae),
            "p95_AE": np.percentile(ae, 95),
            "bias": err.mean(),
            "spearman_rho": spearmanr(true, pred).statistic,   # rank agreement
            "pearson_r": pearsonr(true, pred).statistic}       # straight-line agreement


def log_metrics(true_log, pred_log):
    """regression() on log10 values, with the names this project stores them under."""
    r = regression(true_log, pred_log)
    # {new_name: value for ...} builds a new dict, renaming each key on the way
    out = {"n": r.pop("n")}
    out.update({f"{k}_log10" if k not in ("spearman_rho", "pearson_r") else k: v
                for k, v in r.items()})
    out["rel_error_pct"] = (10 ** out["MAE_log10"] - 1) * 100   # MAE_log10 as a "typical x% off"
    return out


def gpa_metrics(true_gpa, pred_gpa, span):
    """The same accuracy in GPa. span = max - min of the true GPa over ALL splits."""
    r = regression(true_gpa, pred_gpa)
    return {"MAE_GPa": r["MAE"], "RMSE_GPa": r["RMSE"], "R2_GPa": r["R2"],
            "pct_of_range": r["MAE"] / span * 100}


def slack_parts(log_K, log_G):
    """The rho- and V-free pieces of the Slack kappa, from log10 moduli.

    Written from the textbook, NOT from cgcnn_scratch/joint.py:
        Poisson ratio   nu    = (3K - 2G) / (2 (3K + G))
        Grueneisen      gamma = 3 (1 + nu) / (2 (2 - 3 nu))
        sound speeds    v_l ~ sqrt(K + 4G/3),  v_t ~ sqrt(G)   (the 1/sqrt(rho) cancels in ratios)
        Debye average   v_s = [ (1/v_l^3 + 2/v_t^3) / 3 ] ^ (-1/3)
        kappa           ~ G v_s * exp(-gamma) * (V^(1/3) / (N T), which also cancels)
    joint.py reaches nu through (v_l/v_t)^2 instead; agreeing with its stored
    numbers therefore checks that algebra too.
    """
    K, G = 10.0 ** log_K, 10.0 ** log_G
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    gamma = 3 * (1 + nu) / (2 * (2 - 3 * nu))
    v_l, v_t = np.sqrt(K + 4 * G / 3), np.sqrt(G)
    v_s = ((1 / v_l ** 3 + 2 / v_t ** 3) / 3) ** (-1 / 3)
    return G * v_s, np.exp(-gamma), gamma


def kappa_budget(tK, tG, pK, pG):
    """How the K and G errors turn into a kappa error (all MAEs in log10 units).

    mae_log_ratio    mean |err_K - err_G|   (= error in log10(K/G), what gamma depends on)
    residual_corr    correlation of err_K with err_G (high = errors cancel in K/G)
    kappa_mae_*      log10 error of the Slack kappa, and of its two factors
    gamma_mae        mean |gamma_pred - gamma_true|
    """
    pre_t, anh_t, gam_t = slack_parts(tK, tG)
    pre_p, anh_p, gam_p = slack_parts(pK, pG)
    # crystals where an extreme K/G makes gamma non-physical are left out of the
    # kappa terms only (same rule as the original; it removes 0 crystals here)
    ok = (np.isfinite(gam_t) & np.isfinite(gam_p) & (anh_t > 0) & (anh_p > 0)
          & (pre_t > 0) & (pre_p > 0))
    eK, eG = pK - tK, pG - tG

    def mae_log(a, b):
        return np.mean(np.abs(np.log10(a[ok]) - np.log10(b[ok])))

    return {"n": int(ok.sum()),
            "mae_log_K": np.mean(np.abs(eK)), "mae_log_G": np.mean(np.abs(eG)),
            "mae_log_ratio": np.mean(np.abs(eK - eG)),
            "residual_corr": np.corrcoef(eK, eG)[0, 1],
            "kappa_mae_total": mae_log(pre_p * anh_p, pre_t * anh_t),
            "kappa_mae_prefactor": mae_log(pre_p, pre_t),
            "kappa_mae_anharmonic": mae_log(anh_p, anh_t),
            "gamma_mae": np.mean(np.abs(gam_p[ok] - gam_t[ok]))}


def kappa_quintiles(tK, tG, pK, pG):
    """kappa error inside each fifth of the TRUE kappa range, + recall of the bottom 10%.

    Conventions copied on purpose (they DEFINE the stored number):
      edges = np.quantile of true kappa at 0, .2, .4, .6, .8, 1; a crystal sitting
      exactly on an edge goes to the upper bin (np.digitize's default).
      recall@10% = overlap of the 10% lowest TRUE and the 10% lowest PREDICTED kappa,
      with n10 = int(0.10 * n) crystals in each.
    """
    pre_t, anh_t, _ = slack_parts(tK, tG)
    pre_p, anh_p, _ = slack_parts(pK, pG)
    kt, kp = pre_t * anh_t, pre_p * anh_p
    err = np.abs(np.log10(kp) - np.log10(kt))
    edges = np.quantile(kt, [0.2, 0.4, 0.6, 0.8])
    q = np.digitize(kt, edges)                      # 0..4 = which fifth each crystal is in
    out = {}
    for b in range(5):
        m = q == b                                  # a True/False "mask": crystals in this fifth
        out[f"Q{b + 1}_n"] = m.sum()
        out[f"Q{b + 1}_kappa_mae"] = err[m].mean()
        out[f"Q{b + 1}_mae_log_K"] = np.abs(pK - tK)[m].mean()
        out[f"Q{b + 1}_mae_log_G"] = np.abs(pG - tG)[m].mean()
        out[f"Q{b + 1}_median_true_G_GPa"] = np.median(10 ** tG[m])
    n10 = max(1, int(0.10 * len(kt)))
    lowest_true = set(np.argsort(kt)[:n10])         # positions of the n10 lowest true kappa
    lowest_pred = set(np.argsort(kp)[:n10])
    out["recall_at_10pct"] = len(lowest_true & lowest_pred) / n10   # & on sets = "in both"
    return out


def flatten_quintile_json(q):
    """The stored quintile dict {'Q1': {'n':..}, 'recall_at_10pct':..} -> flat names."""
    flat = {}
    for key, val in q.items():
        if isinstance(val, dict):
            for k, v in val.items():
                flat[f"{key}_{k}"] = v
        else:
            flat[key] = val
    return flat


# =============================================================================
#  3. CGCNN moduli: single model and 3-ensemble (steps 02-06)
# =============================================================================
def cgcnn_moduli():
    summary = pd.read_csv(os.path.join(RES, "metrics_summary.csv"))
    pairs = {}                                     # kept for the kappa budget in section 6
    for tag, label in [("K_VRH_full", "CGCNN single (seed 42)"), ("G_VRH_full", "CGCNN single (seed 42)"),
                       ("K_VRH_ens", "CGCNN 3-ensemble"), ("G_VRH_ens", "CGCNN 3-ensemble")]:
        d = pd.read_csv(os.path.join(RES, f"predictions_{tag}.csv"))
        target = tag[:5]                           # "K_VRH_full"[:5] = "K_VRH" (first five letters)
        pairs[tag] = d
        span = d.true_GPa.max() - d.true_GPa.min()
        per_tag = pd.read_csv(os.path.join(RES, f"metrics_{tag}.csv")).set_index("split")
        json_path = os.path.join(RES, f"summary_{tag}.json")
        for split in ["train", "val", "test"]:
            s = d[d.split == split]
            vals = log_metrics(s.true_log10.values, s.pred_log10.values)
            vals.update(gpa_metrics(s.true_GPa.values, s.pred_GPa.values, span))
            where = dict(family="CGCNN moduli", model=label, run=tag, trained_on="matbench",
                         test_set=MB, split=split, target=target)
            # stored name -> our name
            names = {"MAE_log10": "MAE_log10", "R2": "R2_log10", "MAE_GPa": "MAE_GPa",
                     "rel_error_pct": "rel_error_pct", "pct_of_range": "pct_of_range"}
            # source 1: the per-tag file, full precision
            row = per_tag.loc[split]
            st = {ours: row[theirs] for theirs, ours in names.items()}
            st["n"] = row["n"]
            record_many(vals, st, f"results/cgcnn/metrics_{tag}.csv", **where)
            # source 2: the 6-decimal summary table - recorded as extra rows on the same values
            row = summary[(summary.tag == tag) & (summary.split == split)].iloc[0]
            for theirs, ours in names.items():
                record(metric=ours, value=vals[ours], n=vals["n"], stored=row[theirs],
                       stored_in="results/cgcnn/metrics_summary.csv", decimals=6, **where)
            record(metric="n_rows", value=vals["n"], stored=row["n"],
                   stored_in="results/cgcnn/metrics_summary.csv", **where)
            # source 3: the training run's own JSON (single models only). It was
            # written during training on the Kaggle GPU; the prediction files come
            # from re-running the saved checkpoint on this laptop's CPU, so a gap
            # in the 4th significant digit is expected - and should be the same
            # size on val and test if that is the only cause.
            key = {"test": "test_mae", "val": "best_val_mae"}.get(split)
            if key and os.path.exists(json_path):
                j = json.load(open(json_path))
                record(metric="MAE_log10", value=vals["MAE_log10"], n=vals["n"],
                       stored=j[key], stored_in=f"results/cgcnn/summary_{tag}.json ({key})",
                       note="logged during GPU training; predictions re-run on CPU", **where)
    return pairs


# =============================================================================
#  4. ALIGNN moduli: 4 single runs per target + the 3-ensemble (steps 12-18)
# =============================================================================
ALIGNN_TARGETS = {"bulk_modulus_kv": "K_VRH", "shear_modulus_gv": "G_VRH"}
ALIGNN_RUNS = {"": "ALIGNN single (existing)", "_s42": "ALIGNN seed 42",
               "_s1": "ALIGNN seed 1", "_s2": "ALIGNN seed 2"}


def read_alignn(folder, which):
    """ALIGNN's own output: 'id, target, prediction' in GPa, with a space after each comma."""
    path = os.path.join(ALI, folder, f"prediction_results_{which}_set.csv")
    d = pd.read_csv(path, skipinitialspace=True, dtype={"id": str})
    return d.rename(columns={"id": "material_id", "target": "true_GPa", "prediction": "pred_GPa"})


def alignn_moduli(cgcnn_pairs):
    stored = pd.read_csv(os.path.join(ALI, "18_alignn_ensemble_scores.csv"))
    ens_file = pd.read_csv(os.path.join(ALI, "18_alignn_ensemble_predictions.csv"))
    rng = np.random.default_rng(0)                 # step 18's bootstrap seed, drawn in its order
    alignn_pairs = {}
    for folder_target, target in ALIGNN_TARGETS.items():
        lbl = target[0]                            # "K" or "G", as step 18 labels its rows
        runs = {}
        for suffix, label in ALIGNN_RUNS.items():
            folder = f"alignn_{folder_target}{suffix}"
            for which in ["test", "train"]:
                d = read_alignn(folder, which)
                if which == "test":
                    runs[suffix] = d
                # ALIGNN can predict a modulus <= 0, which has no log10: drop those
                # rows from the log metrics (as step 18 does) and count them
                pos = d.pred_GPa > 0
                vals = log_metrics(np.log10(d.true_GPa[pos].values), np.log10(d.pred_GPa[pos].values))
                span = d.true_GPa.max() - d.true_GPa.min()     # this split only: ALIGNN has no all-split file
                g = gpa_metrics(d.true_GPa.values, d.pred_GPa.values, span)
                del g["pct_of_range"]                          # needs all splits; not comparable
                vals.update(g)
                vals["n_nonpositive_pred"] = int((~pos).sum())
                st = {}
                if which == "test":
                    row = stored[(stored.target == lbl) & (stored.model == label)]
                    if len(row):
                        st = {"MAE_log10": row.mae_log10.iloc[0], "R2_log10": row.r2.iloc[0]}
                record_many(vals, st, "results/alignn/18_alignn_ensemble_scores.csv",
                            family="ALIGNN moduli", model=label, run=folder, trained_on="matbench",
                            test_set=MB, split=which, target=target,
                            note="" if which == "test" else "7,680 of 7,691 train rows (ALIGNN drops a partial batch)")

        # ---- the ensemble: rebuild it from the three seed files, then score it ----
        base = runs["_s42"]                                   # step 18 orders everything by seed 42's ids
        ids = base.material_id.tolist()
        stack = np.vstack([runs[s].set_index("material_id").loc[ids, "pred_GPa"].values
                           for s in ["_s42", "_s1", "_s2"]])  # 3 rows x 1,648 columns
        with np.errstate(divide="ignore", invalid="ignore"):
            log_stack = np.log10(np.where(stack > 0, stack, np.nan))
        ens_log = np.nanmean(log_stack, axis=0)               # mean over the members, ignoring NaN
        true_log = np.log10(base.true_GPa.values)

        # check 1: the saved ensemble file holds exactly this mean
        saved = ens_file[ens_file.target == lbl].set_index("material_id").loc[ids]
        record(family="ensemble check", model="ALIGNN 3-ensemble", run=f"18 {lbl}",
               trained_on="matbench", test_set=MB, split="test", target=target,
               metric="max |saved - mean of member files| (log10)",
               value=np.nanmax(np.abs(saved.pred_log10.values - ens_log)), n=len(ids),
               stored=0.0, stored_in="(should be 0)", decimals=6)
        vals = log_metrics(true_log, ens_log)
        span = base.true_GPa.max() - base.true_GPa.min()
        vals.update({k: v for k, v in gpa_metrics(base.true_GPa.values, 10 ** ens_log, span).items()
                     if k != "pct_of_range"})
        row = stored[(stored.target == lbl) & (stored.model == "ALIGNN 3-ens")]
        record_many(vals, {"MAE_log10": row.mae_log10.iloc[0], "R2_log10": row.r2.iloc[0]},
                    "results/alignn/18_alignn_ensemble_scores.csv",
                    family="ALIGNN moduli", model="ALIGNN 3-ensemble", run=f"18_alignn_ensemble ({lbl})",
                    trained_on="matbench", test_set=MB, split="test", target=target)
        alignn_pairs[target] = pd.DataFrame({"material_id": ids, "true_log10": true_log,
                                             "pred_log10": ens_log})

        # check 2: step 18's paired bootstraps, redrawn with the same seed in the same order
        ens_err = np.abs(ens_log - true_log)
        valid = np.isfinite(ens_log) & np.isfinite(true_log)
        single = runs[""].set_index("material_id").loc[ids, "pred_GPa"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            single_err = np.abs(np.log10(np.where(single > 0, single, np.nan)) - true_log)
        cg = cgcnn_pairs[f"{target}_ens"]
        cg = cg[cg.split == "test"].set_index("material_id").loc[ids]
        cg_err = np.abs(cg.pred_log10.values - true_log)
        for name, other_err in [("d(ens - single)", single_err),
                                ("d(ALIGNN ens - CGCNN ens)", cg_err)]:
            v = valid & np.isfinite(other_err)       # crystals both models can be scored on
            a, b = ens_err[v], other_err[v]
            idx = rng.integers(0, len(a), (10000, len(a)))   # 10,000 redraws of the crystal positions
            diff = a[idx].mean(axis=1) - b[idx].mean(axis=1)
            row = stored[(stored.target == lbl) & (stored.model == name)].iloc[0]
            for metric, value, st in [("bootstrap mean of d(MAE_log10)", diff.mean(), row.mae_log10),
                                      ("d 95% CI low", np.percentile(diff, 2.5), row.ci_lo),
                                      ("d 95% CI high", np.percentile(diff, 97.5), row.ci_hi)]:
                record(family="ALIGNN moduli", model=name, run=f"18 bootstrap ({lbl})",
                       trained_on="matbench", test_set=MB, split="test", target=target,
                       metric=metric, value=value, n=len(a), stored=st,
                       stored_in="results/alignn/18_alignn_ensemble_scores.csv",
                       note="paired bootstrap, rng seed 0, step 18's draw order")
        # the CGCNN-ens row step 18 also stored
        cg_vals = log_metrics(cg.true_log10.values, cg.pred_log10.values)
        row = stored[(stored.target == lbl) & (stored.model == "CGCNN 3-ens")].iloc[0]
        for ours, theirs in [("MAE_log10", "mae_log10"), ("R2_log10", "r2")]:
            record(family="CGCNN moduli", model="CGCNN 3-ensemble", run=f"{target}_ens",
                   trained_on="matbench", test_set=MB, split="test", target=target,
                   metric=ours, value=cg_vals[ours], n=cg_vals["n"], stored=row[theirs],
                   stored_in="results/alignn/18_alignn_ensemble_scores.csv")
    return alignn_pairs


# =============================================================================
#  5. Joint CGCNN: every saved run (steps 28-34)
# =============================================================================
def joint_runs():
    comparison = pd.read_csv(os.path.join(ROOT, "results", "joint_comparison.csv")).set_index("run")
    resid = pd.read_csv(os.path.join(RES, "60_residual_corr_table.csv")).set_index("run")
    resid_names = {"r4_ens": "joint r4, 3-model ens.", "r5_ensB": "joint r5 ens B",
                   "r5_ensC": "joint r5 ens C"}
    preds = {}
    for path in sorted(glob.glob(os.path.join(RES, "predictions_joint_*.csv"))):
        tag = os.path.basename(path)[len("predictions_joint_"):-len(".csv")]
        d = pd.read_csv(path)
        preds[tag] = d
        j = json.load(open(os.path.join(RES, f"summary_joint_{tag}.json")))
        trained = "matbench + AFLOW" if "_af_" in tag else "matbench"
        model = "joint CGCNN 3-ensemble" if "ens" in tag else "joint CGCNN single"
        for split in ["train", "val", "test"]:
            s = d[d.split == split]
            a = [s[c].values for c in ["true_log10_K", "true_log10_G", "pred_log10_K", "pred_log10_G"]]
            vals = kappa_budget(*a)                 # *a = pass the four arrays as four arguments
            stored = dict(j["results"][split])
            record("joint CGCNN", model, tag, trained, MB, split, "K+G -> kappa", "n_rows",
                   vals["n"], vals["n"], stored["n"], f"results/cgcnn/summary_joint_{tag}.json")
            record_many(vals, stored, f"results/cgcnn/summary_joint_{tag}.json",
                        family="joint CGCNN", model=model, run=tag, trained_on=trained,
                        test_set=MB, split=split, target="K+G -> kappa")
            # the per-modulus accuracy the JSON never stored (R2, MSE, Spearman, ...)
            for t, (tc, pc) in {"K_VRH": a[0::2], "G_VRH": a[1::2]}.items():
                # a[0::2] = items 0 and 2 of the list (true_K, pred_K); a[1::2] = 1 and 3
                lm = {k: v for k, v in log_metrics(tc, pc).items() if k != "MAE_log10"}
                record_many(lm, family="joint CGCNN", model=model, run=tag, trained_on=trained,
                            test_set=MB, split=split, target=t)
            if split != "test":
                continue
            if tag in comparison.index:
                row = comparison.loc[tag]
                record_many({k: vals[k] for k in row.index}, row.to_dict(),
                            "results/joint_comparison.csv", family="joint CGCNN", model=model,
                            run=tag, trained_on=trained, test_set=MB, split=split,
                            target="K+G -> kappa")
            if tag in resid_names:
                row = resid.loc[resid_names[tag]]
                record_many({k: vals[k] for k in ["residual_corr", "mae_log_K", "mae_log_G", "mae_log_ratio"]},
                            row.to_dict(), "results/cgcnn/60_residual_corr_table.csv",
                            family="joint CGCNN", model=model, run=tag, trained_on=trained,
                            test_set=MB, split=split, target="K+G -> kappa")
            if j.get("test_by_kappa_quintile"):
                record_many(kappa_quintiles(*a), flatten_quintile_json(j["test_by_kappa_quintile"]),
                            f"results/cgcnn/summary_joint_{tag}.json", family="joint CGCNN",
                            model=model, run=tag, trained_on=trained, test_set=MB, split=split,
                            target="kappa quintiles")
            else:
                record_many(kappa_quintiles(*a), family="joint CGCNN", model=model, run=tag,
                            trained_on=trained, test_set=MB, split=split, target="kappa quintiles")

    # ensemble check: is each *_ens file the mean of the member files its JSON names?
    for tag in [t for t in preds if "ens" in t]:
        members = json.load(open(os.path.join(RES, f"summary_joint_{tag}.json")))["members"]
        e = preds[tag].set_index(["material_id", "split"])
        worst = 0.0
        for col in ["pred_log10_K", "pred_log10_G"]:
            mean = np.mean([preds[m].set_index(["material_id", "split"]).loc[e.index, col].values
                            for m in members], axis=0)
            worst = max(worst, np.max(np.abs(e[col].values - mean)))
        record("ensemble check", "joint CGCNN 3-ensemble", tag, "", MB, "all", "K and G",
               "max |saved - mean of member files| (log10)", worst, len(e), 0.0,
               "(should be 0)", decimals=6, note="members: " + ", ".join(members))


# =============================================================================
#  6. The separate CGCNN models pushed through the same kappa budget (steps 26, 60)
# =============================================================================
def separate_budget(cgcnn_pairs, alignn_pairs):
    comparison = pd.read_csv(os.path.join(ROOT, "results", "joint_comparison.csv")).set_index("run")
    resid = pd.read_csv(os.path.join(RES, "60_residual_corr_table.csv")).set_index("run")
    for kind, label, cmp_row, resid_row in [
            ("full", "CGCNN separate, 1 model each", "BASELINE 1 model each", "separate, 1 model each"),
            ("ens", "CGCNN separate, 3-ensemble each", "BASELINE 3-model ens.", "separate, 3-model ens.")]:
        K, G = cgcnn_pairs[f"K_VRH_{kind}"], cgcnn_pairs[f"G_VRH_{kind}"]
        # merge = line the two tables up by crystal and split, like a lookup
        m = K.merge(G, on=["material_id", "split"], suffixes=("_K", "_G"))
        for split in ["train", "val", "test"]:
            s = m[m.split == split]
            vals = kappa_budget(s.true_log10_K.values, s.true_log10_G.values,
                                s.pred_log10_K.values, s.pred_log10_G.values)
            where = dict(family="CGCNN separate -> kappa", model=label, run=f"{{K,G}}_VRH_{kind}",
                         trained_on="matbench", test_set=MB, split=split, target="K+G -> kappa")
            if split != "test":
                record_many(vals, **where)
                continue
            row = comparison.loc[cmp_row]
            # joint_comparison's two baseline rows were typed in rounded: 4 decimals, corr 3
            for k in row.index:
                record(metric=k, value=vals[k], n=vals["n"], stored=row[k],
                       stored_in="results/joint_comparison.csv",
                       decimals=3 if k == "residual_corr" else 4, **where)
            row = resid.loc[resid_row]
            st = {k: row[k] for k in ["residual_corr", "mae_log_K", "mae_log_G", "mae_log_ratio"]}
            record_many(vals, st, "results/cgcnn/60_residual_corr_table.csv", **where)
            record_many(kappa_quintiles(s.true_log10_K.values, s.true_log10_G.values,
                                        s.pred_log10_K.values, s.pred_log10_G.values),
                        family=where["family"], model=label, run=where["run"], trained_on="matbench",
                        test_set=MB, split="test", target="kappa quintiles")
    # ALIGNN 3-ensemble through the same budget: never stored, recorded as new
    K, G = alignn_pairs["K_VRH"], alignn_pairs["G_VRH"]
    m = K.merge(G, on="material_id", suffixes=("_K", "_G"))
    a = [m.true_log10_K.values, m.true_log10_G.values, m.pred_log10_K.values, m.pred_log10_G.values]
    for tgt, vals in [("K+G -> kappa", kappa_budget(*a)), ("kappa quintiles", kappa_quintiles(*a))]:
        record_many(vals, family="ALIGNN separate -> kappa", model="ALIGNN 3-ensemble, K and G",
                    run="18_alignn_ensemble", trained_on="matbench", test_set=MB, split="test",
                    target=tgt)


# =============================================================================
#  7. Tree baselines (step 43)
# =============================================================================
def tree_baselines():
    d = pd.read_csv(os.path.join(RES, "baseline_tree", "csv", "43_baseline_tree_predictions.csv"))
    stored = pd.read_csv(os.path.join(RES, "baseline_tree", "csv", "43_baseline_tree_scores.csv"))
    for _, st in stored.iterrows():            # iterrows() walks the table one row at a time
        s = d[(d.source == st.source) & (d.split == st.split)]
        true = s[f"{st.target}_true_log10"].values
        pred = s[f"{st.target}_pred_log10_{st.model}"].values
        vals = log_metrics(true, pred)
        record_many(vals, {"MAE_log10": st.mae_log10, "n": st.n},
                    "results/cgcnn/baseline_tree/csv/43_baseline_tree_scores.csv",
                    family="tree baseline", model=st.model.replace("_", " "),
                    run=f"43 {st.source}", trained_on=st.source,
                    test_set=MB if st.source == "matbench" else AF, split=st.split, target=st.target)
        record("tree baseline", st.model.replace("_", " "), f"43 {st.source}", st.source,
               MB if st.source == "matbench" else AF, st.split, st.target, "n_rows",
               vals["n"], vals["n"], st.n, "results/cgcnn/baseline_tree/csv/43_baseline_tree_scores.csv")


# =============================================================================
#  8. Direct structure -> kappa CGCNN, trained on AFLOW AGL kappa (direct_kappa_no_slack)
# =============================================================================
def direct_kappa():
    names = {"MAE_log10": "mae_log10", "median_AE_log10": "median_ae_log10",
             "p95_AE_log10": "p95_ae_log10", "RMSE_log10": "rmse_log10"}
    control = pd.read_csv(os.path.join(DK, "results", "csv", "direct_vs_control_summary.csv")).set_index("tag")
    recipe = pd.read_csv(os.path.join(KAG, "aflow_mbrecipe_summary.csv")).set_index("tag")
    runs = [(f"s{s}", "direct-kappa CGCNN (own recipe)",
             os.path.join(DK, "results", "csv", f"01_direct_kappa_test_s{s}.csv"),
             os.path.join(DK, "models", f"summary_direct_kappa_s{s}.json"), control,
             "direct_kappa_no_slack/results/csv/direct_vs_control_summary.csv") for s in [42, 1, 2]]
    runs += [(f"mbrecipe_s{s}", "direct-kappa CGCNN (matbench recipe)",
              os.path.join(KAG, "predictions", f"01_direct_kappa_test_mbrecipe_s{s}.csv"),
              os.path.join(KAG, "models", f"summary_direct_kappa_mbrecipe_s{s}.json"), recipe,
              "kaggle/output_aflow_recipe/aflow_mbrecipe_summary.csv") for s in [42, 1, 2]]
    for tag, label, pred_path, json_path, table, table_name in runs:
        d = pd.read_csv(pred_path)
        vals = log_metrics(d.true_log10_kappa.values, d.pred_log10_kappa.values)
        where = dict(family="direct kappa", model=label, run=tag, trained_on="AFLOW AGL kappa (Debye model)",
                     test_set=AF + ", test only", split="test", target="log10 kappa (AGL)")
        j = json.load(open(json_path))["results"]["test"]
        st = {ours: j[theirs] for ours, theirs in names.items()}
        st["n"] = j["n"]
        record_many(vals, st, os.path.relpath(json_path, ROOT), **where)
        record(metric="n_rows", value=vals["n"], n=vals["n"], stored=j["n"],
               stored_in=os.path.relpath(json_path, ROOT), **where)
        row = table.loc[tag]
        record_many({k: vals[k] for k in names},
                    {ours: row[f"test_{theirs}"] for ours, theirs in names.items()},
                    table_name, **where)


# =============================================================================
#  9. Direct ALIGNN structure -> kappa_L (coherent-heat step 25, 5-fold out-of-fold)
# =============================================================================
def direct_alignn_oof():
    d = pd.read_csv(os.path.join(COH, "results", "25_oof_klat_predictions.csv"))
    folds = pd.read_csv(os.path.join(COH, "results", "25_oof_summary.csv")).set_index("fold")
    where = dict(family="direct kappa", model="direct ALIGNN (PhoNIX, 5-fold OOF)",
                 trained_on="PhoNIX phonon-DFT kappa_L", target="log10 kappa_L (PhoNIX)")
    for fold, s in d.groupby("fold"):          # groupby = handle each fold's rows separately
        vals = log_metrics(s.true_log_klat.values, s.oof_pred.values)
        record_many(vals, {"MAE_log10": folds.loc[fold, "held_out_mae"], "n": folds.loc[fold, "n_held_out"]},
                    "[coherent-heat repo] results/25_oof_summary.csv", run=f"fold {fold}",
                    test_set="PhoNIX, fold held out", split="held-out fold", **where)
        record(metric="n_rows", value=vals["n"], n=vals["n"], stored=folds.loc[fold, "n_held_out"],
               stored_in="[coherent-heat repo] results/25_oof_summary.csv", run=f"fold {fold}",
               test_set="PhoNIX, fold held out", split="held-out fold", **where)
    record_many(log_metrics(d.true_log_klat.values, d.oof_pred.values), run="all 5 folds pooled",
                test_set="PhoNIX, every crystal held out once (5,316)", split="out-of-fold", **where)


# =============================================================================
#  10. Compare, flag, save, plot
# =============================================================================
def finish():
    out = pd.DataFrame(ROWS)
    has = out.stored.notna()
    out["abs_diff"] = (out.recomputed - out.stored).abs()
    # relative difference; undefined (NaN) when the stored number is 0 (the ensemble checks)
    out["rel_diff"] = out.abs_diff / out.stored.abs().where(out.stored != 0)
    # the rounding allowance: half a unit in the last decimal the stored number kept
    dec = out.stored_decimals.where(out.stored_decimals.notna(), out.stored.map(
        lambda x: decimals_of(x) if pd.notna(x) else 17)).astype(float)
    rounding = 0.5 * 10.0 ** (-dec.clip(upper=15))
    # float allowance: the prediction files hold float32 numbers (~7 digits), so a
    # sum over thousands of them can move in the 7th significant digit
    allowed = np.maximum(rounding, 1e-6 * out.stored.abs()) + 1e-12
    out["status"] = np.select(
        [~has, out.abs_diff <= allowed, out.rel_diff <= 1e-3],
        ["new", "match", "close"], default="MISMATCH")
    full_precision = dec >= 10            # stored with all its digits, i.e. never rounded
    out = out.drop(columns="stored_decimals")
    out.to_csv(OUT_CSV, index=False)

    print("=" * 100)
    print("STEP 95 PART 2a - accuracy metrics recomputed from the prediction files")
    print("=" * 100)
    print(f"values written: {len(out)}   (to {os.path.relpath(OUT_CSV, ROOT)})\n")
    print(pd.crosstab(out.family, out.status, margins=True, margins_name="total").to_string())
    bad = out[out.status.isin(["close", "MISMATCH"])]
    if len(bad):
        print(f"\n{len(bad)} values that are not an exact match:")
        cols = ["family", "run", "split", "target", "metric", "recomputed", "stored", "rel_diff",
                "status", "stored_in"]
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            print(bad[cols].to_string(index=False))
    cmp = out[has]
    fp = cmp[full_precision[has] & (cmp.status == "match")]
    print(f"\n'match' rows stored at full precision: {len(fp)}, largest relative difference "
          f"{fp.rel_diff.max():.1e} (float32 noise); the rest agree within the stored rounding")

    # ---- figure: every compared value, stored vs recomputed ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    colours = {"match": "#3b6ea8", "close": "#e69f00", "MISMATCH": "#c0392b"}
    pos = cmp[(cmp.stored.abs() > 0) & (cmp.recomputed.abs() > 0)]
    for status, c in colours.items():
        g = pos[pos.status == status]
        ax[0].scatter(g.stored.abs(), g.recomputed.abs(), s=10, color=c,
                      label=f"{status} ({len(cmp[cmp.status == status])})")
    lo, hi = pos.stored.abs().min() / 2, pos.stored.abs().max() * 2
    ax[0].plot([lo, hi], [lo, hi], color="grey", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlabel="stored value (|.|)", ylabel="recomputed value (|.|)",
              title="Every stored accuracy number vs its recomputation")
    ax[0].legend()
    rd = cmp.rel_diff.dropna().clip(lower=1e-17)          # the stored-0 ensemble checks have no ratio
    ax[1].hist(np.log10(rd), bins=60, color="#3b6ea8")
    ax[1].axvline(-3, color="#c0392b", ls="--", label="0.1%: above = MISMATCH")
    ax[1].set(xlabel="log10 relative difference (exact matches piled at -17)", ylabel="values",
              title="How far apart stored and recomputed are")
    ax[1].legend()
    fig.suptitle("Step 95 part 2a - accuracy metrics, recomputed from the saved predictions")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"figure: {os.path.relpath(OUT_PNG, ROOT)}")


def main():
    cgcnn_pairs = cgcnn_moduli()
    alignn_pairs = alignn_moduli(cgcnn_pairs)
    joint_runs()
    separate_budget(cgcnn_pairs, alignn_pairs)
    tree_baselines()
    direct_kappa()
    direct_alignn_oof()
    finish()


if __name__ == "__main__":      # run main() only when this file is run, not when imported
    main()
