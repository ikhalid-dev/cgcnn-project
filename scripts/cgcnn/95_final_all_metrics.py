#!/usr/bin/env python3
"""
STEP 95 PART 3 - ONE final table: every metric, every model, and how each was checked
=====================================================================================

    OMP_NUM_THREADS=1 ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/95_final_all_metrics.py

WHAT GOES IN
------------
  1. part 2a   results/cgcnn/95_recomputed_accuracy.csv    accuracy metrics, recomputed from
                                                            the saved predictions
  2. part 2b   results/cgcnn/95_recomputed_screening.csv   screening metrics, recomputed from
                                                            their inputs
  3. the stored metric files 2a/2b could NOT recompute, because the run never saved its
     per-crystal predictions: per-seed training JSONs, the joint sweeps, rounds 7-9, the
     step 51-58 summary tables. Their numbers are copied in as they are: "stored only".
  4. step 46, rebuilt here from step 45's scored GNoME table (its input is still on disk).
  5. CROSS-FILE CHECKS, new in this part. Wherever two files hold the same number, or one
     was built from the other (a seed mean, a hand-typed copy, precision = tp/(tp+fp)),
     check that they agree. For a "stored only" number this is the only check there is.

WHAT COMES OUT
--------------
  results/cgcnn/95_final_all_metrics.csv   one row per (model, metric, comparison)
  results/cgcnn/95_final_all_metrics.png   the status of every row, and every cross-file check

THE COLUMNS
-----------
  source       where the row comes from: 2a / 2b / part 3 recomputed / stored only /
               part 3 cross-file check
  row_type     metric            a number that describes a model
               check             a data check (is the per-crystal table identical? is the
                                 ensemble the mean of its members? are the row counts right?)
               cross-file check  two files that must agree
  family, model, run, trained_on, test_set, split, target
               which model, trained on what, tested on which crystals, predicting what
  metric, n    the metric's name, and how many crystals it is computed on
  value        THE NUMBER TO QUOTE: the recomputed one when there is one, else the stored one
  recomputed, stored, stored_in
               the two sides of the check, and the file the stored side was read from
  abs_diff, rel_diff, status
               how far apart the two sides are, and the verdict:
                 match         equal, within the stored rounding / float32 noise
                 close         within 0.1%; the cause is explained in 'note'
                 MISMATCH      anything else (there should be none)
                 new           recomputed, but never stored before: nothing to compare with
                 stored only   no predictions saved, so it cannot be recomputed
  primary      True on the FIRST row of each (model, metric). One number is often compared
               against several stored files (e.g. a per-run CSV AND a summary table), which
               gives several rows; keep primary == True to see each metric once.
  note         anything needed to read the row

LABEL CORRECTIONS against the part-1 inventory (checked in the code that made the runs):
  * summary_gamma3.json and summary_r8_*.json were trained and tested on AFLOW's 1,460-crystal
    gamma set (1,022 train; scripts/cgcnn/37_train_gamma.py, 38_train_gamma_transfer.py),
    NOT the 5,563-crystal set with the 835-crystal test split.
  * step 46 compares two models on 33,118 GNoME candidates, which have no ground truth;
    it is not an AFLOW-test-set result.
"""

import os
import re
import ast            # reads step 47's ROSTER list out of its source file without running it
import json
import glob
import importlib      # imports part 2a, whose file name starts with a digit
import itertools      # every way to pick k items from a list (for ties at a cut)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # draw straight to a file, never open a window
import matplotlib.pyplot as plt

# Part 2a's file supplies the folder names, the two split labels and the
# decimals_of() helper, so all three parts describe things in the same words.
# Importing it only DEFINES its functions; nothing in it runs.
acc = importlib.import_module("95_recompute_accuracy")

ROOT, RES = acc.ROOT, acc.RES
MB, AF = acc.MB, acc.AF
GAMMA_SET = "AFLOW gamma set, 1,460 crystals (train 1,022; val and test 15% each)"
GNOME = "GNoME, 33,118 candidates (no truth: the two models compared with each other)"
N_AFLOW_TEST = 835

IN_ACC = os.path.join(RES, "95_recomputed_accuracy.csv")
IN_SCR = os.path.join(RES, "95_recomputed_screening.csv")
IN_INV = os.path.join(RES, "95_metrics_inventory.csv")
OUT_CSV = os.path.join(RES, "95_final_all_metrics.csv")
OUT_PNG = os.path.join(RES, "95_final_all_metrics.png")

STORED_ONLY = "stored only (read from file; no saved predictions to recompute from)"
RECOMP3 = "part 3: recomputed from inputs"
CROSS = "part 3: cross-file check"

ROWS = []      # every row this part makes; one dict per row, turned into a table at the end


# =============================================================================
#  1. Recording rows
# =============================================================================
def add(source, row_type, where, metric, stored, stored_in, recomputed=np.nan, n=np.nan,
        split="test", decimals=None, note=""):
    """Append one row. where = dict(family, model, run, trained_on, test_set, target).

    decimals = how many decimals the stored number was ROUNDED to, when that is
    known (a hand-typed 0.1154 -> 4). None = read it off the number itself.
    """
    # **where unpacks the dict: its six name -> value pairs become six columns
    ROWS.append(dict(source=source, row_type=row_type, **where, split=split, metric=metric,
                     n=n, recomputed=float(recomputed), stored=float(stored),
                     stored_in=stored_in, decimals=decimals, note=note))


def cross(group, item, metric, derived, stored, stored_in, derived_from, decimals=None, note=""):
    """One cross-file check: 'derived' is worked out from one file, 'stored' read from another."""
    where = dict(family="cross-file check", model=group, run=item, trained_on="", test_set="",
                 target="")
    text = "derived from " + derived_from + (". " + note if note else "")
    add(CROSS, "cross-file check", where, metric, stored, stored_in, recomputed=derived,
        split="", decimals=decimals, note=text)


def judge(df):
    """The verdict rule of parts 2a/2b (95_recompute_accuracy.finish), unchanged.

    allowed gap = half a unit in the stored number's last decimal (its rounding),
                  or one part in a million (float32 noise), whichever is larger
    match = within that, close = within 0.1%, MISMATCH = otherwise,
    new = nothing stored; undefined on BOTH sides (0/0) with a stored file named = match.
    """
    has = df.stored.notna()
    abs_diff = (df.recomputed - df.stored).abs()
    rel_diff = abs_diff / df.stored.abs().where(df.stored != 0)
    # .where(cond, other) keeps a value where cond is True and uses 'other' elsewhere
    dec = df.decimals.where(df.decimals.notna(), df.stored.map(
        lambda x: acc.decimals_of(x) if pd.notna(x) else 17)).astype(float)
    allowed = np.maximum(0.5 * 10.0 ** (-dec.clip(upper=15)), 1e-6 * df.stored.abs()) + 1e-12
    both_undefined = df.recomputed.isna() & df.stored.isna() & df.stored_in.fillna("").ne("")
    status = np.select([both_undefined, ~has, abs_diff <= allowed, rel_diff <= 1e-3],
                       ["match", "new", "match", "close"], default="MISMATCH")
    return abs_diff, rel_diff, status


def path_of(p):
    return os.path.join(ROOT, p)


# =============================================================================
#  2. Stored-only training JSONs
# =============================================================================
# Bookkeeping keys: they describe the run (its settings, size, timing), not how good it is.
NOT_METRICS = {"tag", "target", "data_dir", "epochs", "epochs_run", "best_epoch",
               "stage1_best_epoch", "stage2_best_epoch", "n_params", "n_total", "n_train",
               "n_val", "n_test", "seconds", "members", "n_members", "config"}
# Metrics stored at the top level: name in the file -> (split, name in the final table)
TOP_LEVEL = {"test_mae": ("test", "MAE_log10"),
             "best_val_mae": ("val", "MAE_log10"),       # the val MAE of the kept checkpoint
             "stage1_best_val_ratio": ("val", "stage-1 mae_log_ratio")}


def quintile_values(q):
    """The 'test_by_kappa_quintile' block -> (split, metric, value, n), named as part 2a names them."""
    flat = acc.flatten_quintile_json(q)          # {'Q1': {'n': ..}} -> {'Q1_n': ..}
    n = flat.pop("n_scored", None)               # .pop = take it out of the dict and return it
    return [("test", metric, v, n) for metric, v in flat.items()]


def json_values(path):
    """Every metric in one training-summary JSON, as (split, metric, value, n) tuples.

    Three layouts occur:
      top level   test_mae / best_val_mae                 single-target CGCNN runs
      'results'   results -> train / val / test -> ...    joint runs, rounds 7 and 9
      'test'      test -> ...                             round 8 (it kept only the test block)
    plus an optional 'test_by_kappa_quintile' block, at the top level or inside 'results'.
    A key that is in none of these lists stops the script: nothing is dropped unnoticed.
    """
    j = json.load(open(path))
    out = []
    for key, value in j.items():
        if key in NOT_METRICS:
            continue
        if key in TOP_LEVEL:
            split, metric = TOP_LEVEL[key]
            out.append((split, metric, value, j.get("n_" + split)))
        elif key in ("results", "test"):
            blocks = value if key == "results" else {"test": value}
            for split, metrics in blocks.items():
                if split == "test_by_kappa_quintile":
                    out += quintile_values(metrics)
                    continue
                for metric, v in metrics.items():
                    if metric != "n":
                        out.append((split, metric, v, metrics.get("n")))
        elif key == "test_by_kappa_quintile":
            out += quintile_values(value)
        else:
            raise SystemExit(f"{os.path.relpath(path, ROOT)}: unknown key '{key}' - "
                             "add it to NOT_METRICS or TOP_LEVEL")
    return out


def json_labels(path):
    """Which model a summary JSON belongs to, from its file name."""
    name = os.path.basename(path)[len("summary_"):-len(".json")]     # "summary_x.json" -> "x"
    seed = name.rsplit("_s", 1)[-1]                                    # "..._s42" -> "42"
    gkappa = "K, G, gamma -> kappa"
    # re.fullmatch = does the WHOLE name fit this pattern? [KG] = K or G, \d+ = digits
    if re.fullmatch(r"[KG]_VRH_s\d+", name):
        return dict(family="CGCNN moduli", model=f"CGCNN single (seed {seed})", run=name,
                    trained_on="matbench", test_set=MB, target=name[:5])
    if name.startswith("joint_"):
        tag = name[len("joint_"):]
        model = ("joint CGCNN sweep (one setting changed)" if tag.startswith("sweep")
                 else "joint CGCNN single")
        return dict(family="joint CGCNN", model=model, run=tag, trained_on="matbench",
                    test_set=MB, target="K+G -> kappa")
    if name.startswith("37_r9_full"):
        return dict(family="round 9 gamma head", model="round 9 (CGCNN, K + G + gamma heads)",
                    run=name[len("37_"):], trained_on="AFLOW (5,563; AGL gamma)", test_set=AF,
                    target=gkappa)
    if name.startswith("37_control"):
        return dict(family="direct kappa", model="round-9 control (rerun in direct_kappa_no_slack)",
                    run=name[len("37_"):], trained_on="AFLOW (5,563; AGL gamma)", test_set=AF,
                    target=gkappa)
    if name.startswith("r8_"):
        kind = "adapt" if "adapt" in name else "frozen"
        return dict(family="round 8 transfer", model=f"round 8 ({kind}: matbench trunk, then gamma)",
                    run=name, trained_on="matbench moduli, then the AFLOW gamma set",
                    test_set=GAMMA_SET, target=gkappa)
    if name == "gamma3":
        return dict(family="round 7 gamma head", model="gamma3 (CGCNN, K + G + gamma heads)",
                    run=name, trained_on=GAMMA_SET, test_set=GAMMA_SET, target=gkappa)
    if name.startswith("mbsubset_"):
        _, t1, t2, size, _ = name.split("_")      # mbsubset_K_VRH_sub_s1 -> its five parts
        n_train = "3,894" if size == "sub" else "7,691"
        return dict(family="CGCNN moduli, matbench subset (step 55)",
                    model=f"CGCNN single, {n_train} training crystals", run=name,
                    trained_on=f"matbench, {n_train} crystals", test_set=MB, target=f"{t1}_{t2}")
    if name.startswith("aflow_mbrecipe_"):
        return dict(family="CGCNN moduli on AFLOW (step 53)", model="CGCNN single, matbench recipe",
                    run=name, trained_on="AFLOW (train 3,894)", test_set=AF, target=name[15:20])
    raise SystemExit(f"no labels for {path}")


def stored_json(path):
    where = json_labels(path)
    for split, metric, value, n in json_values(path):
        w = dict(where)
        if metric.startswith("Q") or metric == "recall_at_10pct":
            w["target"] = "kappa quintiles"           # the same name part 2a gives these
        add(STORED_ONLY, "metric", w, metric, value, os.path.relpath(path, ROOT),
            n=np.nan if n is None else n, split=split)


# =============================================================================
#  3. Stored-only summary tables
# =============================================================================
def stored_csv(path, ids, where, metrics=None, keep=None, split="test", n_col=None, note_col=None):
    """Turn every number in a stored table into one row ("melting" a wide table into a long one).

    ids      columns that NAME a row (they become the 'run' text)
    where    the labels, or a function row -> labels when they differ per row
    metrics  which columns are metrics (None = every number column that is not an id)
    keep     a function table -> True/False per row, to take only some rows
    """
    d = pd.read_csv(path_of(path))
    if keep is not None:
        d = d[keep(d)]
    if metrics is None:
        skip = set(ids) | {n_col, note_col}
        metrics = [c for c in d.columns if c not in skip and pd.api.types.is_numeric_dtype(d[c])]
    for _, r in d.iterrows():                       # one table row at a time
        w = where(r) if callable(where) else dict(where)
        w.setdefault("run", " | ".join(str(r[c]) for c in ids))   # only if 'where' gave none
        for metric in metrics:
            add(STORED_ONLY, "metric", w, metric, float(r[metric]), path,
                n=r[n_col] if n_col else np.nan, split=split,
                note=str(r[note_col]) if note_col else "")


def stored_tables():
    af = lambda family, model, target: dict(family=family, model=model, trained_on="AFLOW",
                                            test_set=AF, target=target)

    def gamma_comparison_labels(r):
        if r["run"].startswith("r9_full"):
            return json_labels(path_of(f"results/cgcnn/summary_37_{r['run']}.json"))
        return json_labels(path_of("results/cgcnn/summary_gamma3.json"))   # "BASELINE 1022-xtal"
    stored_csv("results/gamma_comparison.csv", ["run"], gamma_comparison_labels)

    p51 = "results/cgcnn/adjudication/csv/"
    stored_csv(p51 + "51_adjudication_scores.csv", ["model", "gamma_source"],
               lambda r: dict(af("51 adjudication", r["model"], "kappa_L <= 1 screen vs AFLOW AGL kappa"),
                              run=f"gamma: {r['gamma_source']}"))
    stored_csv(p51 + "51_head_to_head.csv", ["winner"],
               dict(af("51 adjudication", "round 9 vs tree (paired bootstrap of F1)",
                       "kappa_L <= 1 screen vs AFLOW AGL kappa")))

    p5x = "results/cgcnn/underfitting/csv/"
    stored_csv(p5x + "53_underfitting_summary.csv", ["target", "arm"],
               lambda r: dict(af("53 underfitting (AFLOW)", r["arm"], r["target"])), split="train / test")
    mb = lambda family, model, target: dict(family=family, model=model, trained_on="matbench subset",
                                            test_set=MB, target=target)
    stored_csv(p5x + "54_matbench_subset_trees.csv", ["target", "size", "n_train", "model"],
               lambda r: mb("54 trees on matbench subsets", r["model"], r["target"]),
               metrics=["train_mae", "test_mae", "test_over_train"], split="train / test")
    stored_csv(p5x + "55_size_effect.csv", ["target"],
               lambda r: mb("55 size effect (matbench)", "CGCNN and best tree", r["target"]))
    stored_csv(p5x + "55_verdict_at_equal_size.csv", ["dataset", "target", "n_train", "winner"],
               lambda r: dict(family="55 verdict at 3,894 training crystals", model="CGCNN vs best tree",
                              trained_on=f"{r['dataset']}, 3,894 crystals",
                              test_set=MB if r["dataset"] == "matbench" else AF, target=r["target"]),
               metrics=["cgcnn", "tree", "floor", "cgcnn_over_floor"])

    stored_csv("results/alignn/58_rho_decomposition.csv", ["scope", "comparison"],
               lambda r: dict(family="58 model agreement", model=r["comparison"], trained_on="",
                              test_set=r["scope"] + " (no truth: model vs model)",
                              target="kappa ranking"),
               metrics=["spearman_rho"], n_col="n", note_col="note")

    # direct_kappa 02: part 2b recomputed the direct and tree rows; the control and
    # round-9 rows have no saved per-crystal predictions in that folder
    p02 = "direct_kappa_no_slack/results/csv/"
    arm_model = {"control": "round-9 control (rerun in direct_kappa_no_slack)",
                 "round 9": "round 9 (CGCNN, K + G + gamma heads)"}
    stored_csv(p02 + "02_direct_vs_slack_scores.csv", ["arm", "tag", "member"],
               lambda r: dict(af("direct-kappa 02 screen", arm_model[r["arm"]],
                                 "kappa_L <= 1 screen vs AFLOW AGL kappa"), run=r["tag"]),
               keep=lambda d: d.arm.isin(["control", "round 9"]))
    stored_csv(p02 + "02_direct_vs_slack_bootstrap.csv", ["vs"],
               lambda r: dict(af("direct-kappa 02 screen", f"direct CGCNN vs {r['vs']}",
                                 "F1 difference (paired bootstrap)"), run=f"vs {r['vs']}"),
               keep=lambda d: d.vs.isin(["control", "round 9"]))


# Stored files that need no rows of their own, and why. Every uncovered file must be
# either read above or listed here - the script stops on anything else.
NOT_MELTED = {
    "kaggle/output_matbench_subset/mbsubset_summary.csv":
        "same numbers as the per-seed JSONs (rows from those); checked against them",
    "results/cgcnn/baseline_tree/csv/46_baseline_vs_r9_gap_decomposition.csv":
        "rebuilt from step 45's table in section 4",
    "results/cgcnn/baseline_tree/csv/46_baseline_vs_r9_screen_confusion.csv":
        "rebuilt from step 45's table in section 4",
}
CSV_READ = {"results/gamma_comparison.csv",
            "results/cgcnn/adjudication/csv/51_adjudication_scores.csv",
            "results/cgcnn/adjudication/csv/51_head_to_head.csv",
            "results/cgcnn/underfitting/csv/53_underfitting_summary.csv",
            "results/cgcnn/underfitting/csv/54_matbench_subset_trees.csv",
            "results/cgcnn/underfitting/csv/55_size_effect.csv",
            "results/cgcnn/underfitting/csv/55_verdict_at_equal_size.csv",
            "results/alignn/58_rho_decomposition.csv"}


def uncovered_files(done):
    """Stored metric files (part 1 inventory) that no 2a/2b row compared against."""
    inv = pd.read_csv(IN_INV)
    # re.sub strips a trailing " (...)" - 2a writes e.g. "summary_K_VRH_full.json (test_mae)"
    covered = set(done.stored_in.dropna().map(lambda s: re.sub(r" \(.*\)$", "", s)))
    return [p for p in inv[inv.kind == "stored"].path if p not in covered]


# =============================================================================
#  4. Step 46, rebuilt from step 45's scored GNoME table
# =============================================================================
def step46():
    """Where the tree baseline and round 9 disagree on GNoME.

    Conventions copied on purpose (they DEFINE the stored numbers):
      * crystals kept: both kappas present and > 0
      * the gap is split into a gamma part and the rest: kappa carries exp(-gamma), so a
        gamma difference d moves log10(kappa) by -d * log10(e); the moduli part = total - gamma
      * the K, G and gamma gaps use the random-forest columns (as step 46 did)
      * sd = sample sd (divide by n-1); p05/p95 = linear interpolation
      * low = kappa <= 1 W/m/K
    """
    src = "results/cgcnn/45_gnome_screen_all_baseline.csv"
    cols = ["Kappa_r9_gamma", "Kappa_baseline", "K_r9_pred", "G_r9_pred", "gamma_r9_pred",
            "K_baseline_rf", "G_baseline_rf", "gamma_baseline_rf"]
    d = pd.read_csv(path_of(src), usecols=cols)
    d = d.dropna(subset=["Kappa_r9_gamma", "Kappa_baseline"])
    d = d[(d.Kappa_r9_gamma > 0) & (d.Kappa_baseline > 0)]
    gap = np.log10(d.Kappa_baseline) - np.log10(d.Kappa_r9_gamma)
    from_gamma = -(d.gamma_baseline_rf - d.gamma_r9_pred) * np.log10(np.e)
    series = {"log10 kappa gap (tree - r9)": gap,
              "...from the moduli (K, G)": gap - from_gamma,
              "...from gamma": from_gamma,
              "log10 K gap (tree - r9)": np.log10(d.K_baseline_rf) - np.log10(d.K_r9_pred),
              "log10 G gap (tree - r9)": np.log10(d.G_baseline_rf) - np.log10(d.G_r9_pred),
              "gamma gap (tree - r9)": d.gamma_baseline_rf - d.gamma_r9_pred}
    p46 = "results/cgcnn/baseline_tree/csv/"
    stored = pd.read_csv(path_of(p46 + "46_baseline_vs_r9_gap_decomposition.csv")).set_index("quantity")
    where = dict(family="46 tree vs round 9 on GNoME", model="tree baseline vs round 9",
                 trained_on="AFLOW (both)", test_set=GNOME)
    for name, s in series.items():
        x = s.values
        vals = {"median": np.median(x), "mean": x.mean(), "sd": x.std(ddof=1),
                "p05": np.quantile(x, 0.05), "p95": np.quantile(x, 0.95),
                "median_as_factor": 10 ** np.median(x) if "gamma gap" not in name else np.nan}
        for metric, v in vals.items():
            add(RECOMP3, "metric", dict(where, run=name, target="model-vs-model gap"), metric,
                stored.loc[name, metric], p46 + "46_baseline_vs_r9_gap_decomposition.csv",
                recomputed=v, n=len(x), split="all candidates")
    conf = pd.read_csv(path_of(p46 + "46_baseline_vs_r9_screen_confusion.csv"), index_col=0)
    r9_low, tree_low = d.Kappa_r9_gamma <= 1.0, d.Kappa_baseline <= 1.0
    # & = "and", ~ = "not", applied crystal by crystal
    counts = {("r9: low", "tree: low"): r9_low & tree_low, ("r9: low", "tree: high"): r9_low & ~tree_low,
              ("r9: high", "tree: low"): ~r9_low & tree_low, ("r9: high", "tree: high"): ~r9_low & ~tree_low}
    for (r, c), mask in counts.items():
        add(RECOMP3, "metric", dict(where, run="screen decision, kappa <= 1", target="agreement count"),
            f"{r}, {c}", conf.loc[r, c], p46 + "46_baseline_vs_r9_screen_confusion.csv",
            recomputed=mask.sum(), n=len(d), split="all candidates")


# =============================================================================
#  5. Cross-file checks
# =============================================================================
def flat_json(path, block="results"):
    """{'train.mae_log_K': value, ...} for every number inside one JSON block."""
    out = {}
    for split, metrics in json.load(open(path))[block].items():
        for k, v in metrics.items():
            out[f"{split}.{k}"] = v
    return out


def test_split(files):
    """The test crystals of one step-47 roster entry: true and predicted log10 K and G."""
    if len(files) == 1:                                   # a joint run keeps K and G in one file
        d = pd.read_csv(path_of("results/cgcnn/" + files[0]))
    else:                                                 # a separate run: one file for K, one for G
        K, G = (pd.read_csv(path_of("results/cgcnn/" + f)) for f in files)
        # merge lines the two up by crystal; the suffixes turn K's 'true_log10' into
        # 'true_log10_K' and G's into 'true_log10_G' - the joint files' own column names
        d = K.merge(G, on=["material_id", "split"], suffixes=("_K", "_G"))
    return d[d.split == "test"]


def lowest_n(values, n):
    """The n lowest values, allowing for a TIE at the n-th place.

    Returns (positions certainly in, positions tied at the cut, how many of the tied fit).
    If two crystals share the 164th-lowest value, either one may fill the last place.
    """
    cut = np.sort(values)[n - 1]
    sure = np.flatnonzero(values < cut)       # flatnonzero = the positions where it is True
    tied = np.flatnonzero(values == cut)
    return set(sure), tied, n - len(sure)


def hit_range(d, n10):
    """Fewest and most hits (true bottom 10% that are also predicted bottom 10%) over every tie choice."""
    kappa = {}
    for side in ["true", "pred"]:
        pre, anh, _ = acc.slack_parts(d[f"{side}_log10_K"].values, d[f"{side}_log10_G"].values)
        kappa[side] = pre * anh                    # the same kappa part 2a ranks by
    t_sure, t_tied, t_fit = lowest_n(kappa["true"], n10)
    p_sure, p_tied, p_fit = lowest_n(kappa["pred"], n10)
    if len(t_tied) > 12 or len(p_tied) > 12:
        raise SystemExit("too many ties at the cut to list every choice")
    # itertools.combinations(tied, k) = every way to pick k of the tied crystals
    hits = [len((t_sure | set(a)) & (p_sure | set(b)))
            for a in itertools.combinations(t_tied, t_fit)
            for b in itertools.combinations(p_tied, p_fit)]
    ties = [d.material_id.values[i] for i in t_tied] if len(t_tied) > t_fit else []
    return min(hits), max(hits), ties


def check_47_recall(done):
    """Step 47's hit counts (from its own scoring) vs part 2a's recall_at_10pct (independent code).

    Both score the same prediction files on the same 1,648 test crystals; 47 stores recall
    rounded to 1 decimal, but also the integer n_hit, which can be compared exactly:
    2a's recall x 164 must equal 47's n_hit.

    One wrinkle: when the 164th-lowest TRUE kappa is shared by two crystals (same labels),
    either one may take the last place, and the hit count can differ by one depending on
    which. Both are correct. So the 'derived' side is the valid count nearest to 47's, from
    the full range every tie choice allows (the range is in the note): inside the range ->
    match, outside -> the distance to the nearest valid count shows as a MISMATCH.
    """
    src = open(os.path.join(ROOT, "scripts", "cgcnn", "47_missed_decile_stability.py")).read()
    roster = None
    for node in ast.parse(src).body:               # walk the file's top-level statements
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ROSTER":
            roster = {label: files for label, tier, files in ast.literal_eval(node.value)}
    p47 = "results/cgcnn/decile_stability/csv/47_missed_decile_run_summary.csv"
    s47 = pd.read_csv(path_of(p47))
    n10 = int(0.10 * 1648)
    cross("47 recall vs part 2a", "all 26 runs", "runs whose bottom-10% size is not int(0.1 x 1,648)",
          int((s47.n_decile != n10).sum()), 0, "(should be 0)", p47, decimals=12)
    for _, r in s47.iterrows():
        f = roster[r["run"]][0]
        if f.startswith("predictions_joint_"):
            fam, run = "joint CGCNN", f[len("predictions_joint_"):-len(".csv")]
        else:
            fam, run = "CGCNN separate -> kappa", "{K,G}_VRH_" + ("ens" if "_ens" in f else "full")
        a = done[(done.family == fam) & (done.run == run) & (done.split == "test")
                 & (done.metric == "recall_at_10pct")]
        hits_2a = round(a.recomputed.iloc[0] * n10)
        lo, hi, ties = hit_range(test_split(roster[r["run"]]), n10)
        if not lo <= hits_2a <= hi:
            raise SystemExit(f"{r['run']}: part 2a's {hits_2a} hits is outside the valid {lo}-{hi}")
        if lo == hi:
            note = f"part 2a: {hits_2a} hits (no tie at the cut can change it)"
        else:
            note = (f"the last place in the true bottom {n10} is a tie between {' and '.join(ties)} "
                    f"(identical true K and G), so any count from {lo} to {hi} is correct; "
                    f"part 2a and the run's own JSON count {hits_2a}")
        cross("47 recall vs part 2a", r["run"], "n_hit (true bottom 10% also predicted bottom 10%)",
              min(max(r["n_hit"], lo), hi), r["n_hit"], p47,
              f"part 2a's recall_at_10pct of {f} x {n10}, and the hit range over tie choices",
              decimals=6, note=note)


def check_82_moduli(done):
    rows = done[done.family.str.startswith("82") &
                done.metric.str.fullmatch(r"per-crystal '[KG]_(ALIGNN|CGCNN-ens|newbase)'.*")]
    cross("82 moduli = model files", "6 columns x 1,648 crystals",
          "largest relative difference, any K or G column", rows.recomputed.max(), 0, "(should be 0)",
          "part 2b's rebuild of 82's K/G columns from results/alignn/18_alignn_ensemble_predictions.csv, "
          "predictions_{K,G}_VRH_ens.csv and the newbase OOF files",
          decimals=12, note=f"{len(rows)} part-2b check rows, all '{'/'.join(sorted(rows.status.unique()))}'")


def check_control_equals_round9():
    """direct_kappa_no_slack re-ran round 9 as its in-session control: identical numbers expected."""
    for s in ["42", "1", "2"]:
        a = flat_json(path_of(f"direct_kappa_no_slack/models/summary_37_control_s{s}.json"))
        b = flat_json(path_of(f"results/cgcnn/summary_37_r9_full_s{s}.json"))
        common = sorted(set(a) & set(b))
        only = len(set(a) ^ set(b))                # ^ on sets = in one but not both
        cross("direct-kappa control = round 9", f"seed {s}",
              "largest |difference|, every train/val/test metric", max(abs(a[k] - b[k]) for k in common),
              0, "(should be 0)",
              f"summary_37_control_s{s}.json vs results/cgcnn/summary_37_r9_full_s{s}.json",
              decimals=12, note=f"{len(common)} metrics compared, {only} in one file only")
    p02 = "direct_kappa_no_slack/results/csv/"
    sc = pd.read_csv(path_of(p02 + "02_direct_vs_slack_scores.csv")).set_index("tag")
    num = [c for c in sc.columns if pd.api.types.is_numeric_dtype(sc[c])]
    for c_tag, r_tag in [("control_s1", "r9_full_s1"), ("control_s2", "r9_full_s2"),
                         ("control_s42", "r9_full_s42"), ("control ensemble", "round 9 ensemble")]:
        cross("direct-kappa control = round 9", f"02 scores: {c_tag} vs {r_tag}",
              f"largest |difference| over {len(num)} columns",
              (sc.loc[c_tag, num] - sc.loc[r_tag, num]).abs().max(), 0, "(should be 0)",
              p02 + "02_direct_vs_slack_scores.csv", decimals=12)
    bs = pd.read_csv(path_of(p02 + "02_direct_vs_slack_bootstrap.csv")).set_index("vs")
    num = [c for c in bs.columns if pd.api.types.is_numeric_dtype(bs[c])]
    cross("direct-kappa control = round 9", "02 bootstrap: vs control vs vs round 9",
          f"largest |difference| over {len(num)} columns",
          (bs.loc["control", num].astype(float) - bs.loc["round 9", num].astype(float)).abs().max(),
          0, "(should be 0)", p02 + "02_direct_vs_slack_bootstrap.csv", decimals=12)


def check_gamma_comparison():
    p = "results/gamma_comparison.csv"
    g = pd.read_csv(path_of(p)).set_index("run")
    for run in g.index:
        if run.startswith("r9_full"):
            src, dec = f"results/cgcnn/summary_37_{run}.json", None
        else:
            # typed into kaggle/build_joint_kernel.py by hand: 4 decimals, recall 3
            src, dec = "results/cgcnn/summary_gamma3.json", 4
        test = json.load(open(path_of(src)))["results"]["test"]
        for col in g.columns:
            cross("gamma_comparison.csv = run JSONs", run, col, test[col], g.loc[run, col], p, src,
                  decimals=None if dec is None else (3 if col.startswith("recall") else dec))


def check_summary_vs_json(summary, json_glob, group):
    """A Kaggle kernel's collected summary table vs the per-run JSONs it was collected from."""
    s = pd.read_csv(path_of(summary)).set_index("tag")
    for path in sorted(glob.glob(path_of(json_glob))):
        j = json.load(open(path))
        for theirs, mine in [("top_test_mae", "test_mae"), ("top_best_val_mae", "best_val_mae")]:
            cross(group, j["tag"], mine, j[mine], s.loc[j["tag"], theirs], summary,
                  os.path.relpath(path, ROOT))


def check_53():
    """Step 53's table from its sources: round-9 JSONs, the recipe runs, step 43's tree scores."""
    p = "results/cgcnn/underfitting/csv/53_underfitting_summary.csv"
    s53 = pd.read_csv(path_of(p)).set_index(["target", "arm"])
    trees = pd.read_csv(path_of("results/cgcnn/baseline_tree/csv/43_baseline_tree_scores.csv"))
    for target, key in [("K_VRH", "mae_log_K"), ("G_VRH", "mae_log_G")]:
        arms = {}
        r9 = [json.load(open(f))["results"] for f in sorted(glob.glob(path_of("results/cgcnn/summary_37_r9_full_s*.json")))]
        arms["r9"] = (pd.Series([r["train"][key] for r in r9]), pd.Series([r["test"][key] for r in r9]),
                      "the 3 round-9 JSONs")
        # recipe train = LAST epoch of the training history (its JSON has no train MAE)
        hist = sorted(glob.glob(path_of(f"kaggle/output_aflow_recipe/predictions/history_aflow_mbrecipe_{target}_s*.csv")))
        tests = [json.load(open(path_of(f"kaggle/output_aflow_recipe/models/summary_aflow_mbrecipe_{target}_s{h.rsplit('_s', 1)[1][:-4]}.json")))["test_mae"]
                 for h in hist]
        arms["recipe"] = (pd.Series([pd.read_csv(h).train_mae.iloc[-1] for h in hist]), pd.Series(tests),
                          "the 3 recipe history files (last epoch) and JSONs")
        for model in ["random_forest", "xgboost"]:
            t = trees[(trees.source == "aflow") & (trees.target == target) & (trees.model == model)].set_index("split")
            arms[f"tree_{model}"] = (pd.Series([t.loc["train", "mae_log10"]]), pd.Series([t.loc["test", "mae_log10"]]),
                                     "43_baseline_tree_scores.csv")
        for arm, (tr, te, src) in arms.items():
            row = s53.loc[(target, arm)]
            vals = {"train": tr.mean(), "test": te.mean(), "test_over_train": te.mean() / tr.mean()}
            if len(tr) > 1:
                vals.update(train_sd=tr.std(), test_sd=te.std())      # pandas .std() = sample sd (n-1)
            for col, v in vals.items():
                cross("53 table = its sources", f"{target} {arm}", col, v, row[col], p, src)


def check_54_55():
    p54 = "results/cgcnn/underfitting/csv/54_matbench_subset_trees.csv"
    t54 = pd.read_csv(path_of(p54))
    for _, r in t54.iterrows():
        cross("54/55 tables = their sources", f"54 {r['target']} {r['size']} {r['model']}", "test_over_train",
              r["test_mae"] / r["train_mae"], r["test_over_train"], p54, "the same row's test_mae / train_mae")
    # step 55: CGCNN = mean over 3 seeds of the subset runs' test MAE; sd divides by n (np.std);
    # tree = the better of the two trees in step 54
    cg = {}
    for f in glob.glob(path_of("kaggle/output_matbench_subset/models/summary_mbsubset_*.json")):
        j = json.load(open(f))
        _, t1, t2, size, _ = j["tag"].split("_")
        cg.setdefault((f"{t1}_{t2}", size), []).append(j["test_mae"])   # setdefault = start an empty list once
    p55 = "results/cgcnn/underfitting/csv/55_size_effect.csv"
    s55 = pd.read_csv(path_of(p55)).set_index("target")
    for target in s55.index:
        t = t54[t54.target == target]
        vals = {"cgcnn_full": np.mean(cg[(target, "full")]), "cgcnn_sub": np.mean(cg[(target, "sub")]),
                "tree_full": t[t["size"] == "full"].test_mae.min(), "tree_sub": t[t["size"] == "sub"].test_mae.min(),
                "cgcnn_seed_sd_sub": np.std(cg[(target, "sub")])}
        for col, v in vals.items():
            cross("54/55 tables = their sources", f"55 size effect {target}", col, v, s55.loc[target, col], p55,
                  "the mbsubset JSONs and 54_matbench_subset_trees.csv")
    p55v = "results/cgcnn/underfitting/csv/55_verdict_at_equal_size.csv"
    v = pd.read_csv(path_of(p55v))
    s53 = pd.read_csv(path_of("results/cgcnn/underfitting/csv/53_underfitting_summary.csv")).set_index(["target", "arm"])
    floors = pd.read_csv(path_of("results/cgcnn/underfitting/csv/56_dataset_comparability.csv"))
    wrong_winner = 0
    for _, r in v.iterrows():
        item, tgt = f"55 verdict {r['dataset']} {r['target']}", r["target"]
        if r["dataset"] == "matbench":
            c, t, dec, src = s55.loc[tgt, "cgcnn_sub"], s55.loc[tgt, "tree_sub"], None, "55_size_effect.csv"
        else:
            # typed into 55's CONFIG by hand from steps 53 and 43, at 4 decimals
            c = s53.loc[(tgt, "recipe"), "test"]
            t = min(s53.loc[(tgt, "tree_random_forest"), "test"], s53.loc[(tgt, "tree_xgboost"), "test"])
            dec, src = 4, "53_underfitting_summary.csv (hand-typed into step 55 at 4 decimals)"
        fl = floors[(floors.dataset == r["dataset"]) & (floors.target == tgt)].within_formula_sd_mean.iloc[0]
        cross("54/55 tables = their sources", item, "cgcnn", c, r["cgcnn"], p55v, src, decimals=dec)
        cross("54/55 tables = their sources", item, "tree", t, r["tree"], p55v, src, decimals=dec)
        cross("54/55 tables = their sources", item, "floor", fl, r["floor"], p55v, "56_dataset_comparability.csv")
        cross("54/55 tables = their sources", item, "cgcnn_over_floor", r["cgcnn"] / r["floor"],
              r["cgcnn_over_floor"], p55v, "the same row's cgcnn / floor")
        wrong_winner += ("CGCNN" if r["cgcnn"] < r["tree"] else "TREE") != r["winner"]
    cross("54/55 tables = their sources", "55 verdict, all 4 rows", "rows whose 'winner' is not the lower error",
          wrong_winner, 0, "(should be 0)", p55v, decimals=12)


def check_confusion_tables():
    """Counts and rates in the screening tables must agree with each other.

    precision = tp / (tp + fp)   recall = tp / (tp + fn)   F1 = 2 tp / (2 tp + fp + fn)
    and every crystal is counted once: tp + fp + fn + tn = 835.
    """
    tables = [("results/cgcnn/adjudication/csv/51_adjudication_scores.csv", None),
              ("direct_kappa_no_slack/results/csv/02_direct_vs_slack_scores.csv",
               lambda d: d.arm.isin(["control", "round 9"]))]
    for path, keep in tables:
        d = pd.read_csv(path_of(path))
        if keep is not None:
            d = d[keep(d)]
        tp, fp, fn, tn = d.tp, d.fp, d.fn, d.tn
        rules = {"n_called = tp + fp": (tp + fp, d.n_called),
                 "n_true_low = tp + fn": (tp + fn, d.n_true_low),
                 f"tp + fp + fn + tn = {N_AFLOW_TEST}": (tp + fp + fn + tn, pd.Series(N_AFLOW_TEST, index=d.index)),
                 "precision = tp / (tp + fp)": (tp / (tp + fp), d.precision),
                 "recall = tp / (tp + fn)": (tp / (tp + fn), d.recall),
                 "F1 = 2 tp / (2 tp + fp + fn)": (2 * tp / (2 * tp + fp + fn), d.f1)}
        for rule, (derived, stored) in rules.items():
            cross("confusion counts = rates", f"{os.path.basename(path)} ({len(d)} rows)",
                  rule + ": largest |difference|", (derived - stored).abs().max(), 0, "(should be 0)",
                  path, decimals=12)
    s = pd.read_csv(path_of("results/cgcnn/adjudication/csv/51_adjudication_scores.csv"))
    h = pd.read_csv(path_of("results/cgcnn/adjudication/csv/51_head_to_head.csv")).iloc[0]
    p = "results/cgcnn/adjudication/csv/51_head_to_head.csv"
    pick = lambda model: s[(s.model == model) & (s.gamma_source == "predicted")].iloc[0]
    for col, model in [("f1_round9", "round 9 (CGCNN)"), ("f1_tree", "tree (random_forest)")]:
        cross("confusion counts = rates", "51 head to head", col, pick(model).f1, h[col], p,
              f"51_adjudication_scores.csv, {model}, predicted gamma")
    r = pick("round 9 (CGCNN)")
    cross("confusion counts = rates", "51 head to head", "n_true_low", r.n_true_low, h.n_true_low, p,
          "51_adjudication_scores.csv")
    cross("confusion counts = rates", "51 head to head", "n_crystals", r.tp + r.fp + r.fn + r.tn,
          h.n_crystals, p, "51_adjudication_scores.csv (tp + fp + fn + tn)")


# =============================================================================
#  6. Assemble, judge, save, print, plot
# =============================================================================
def main():
    # ---- parts 2a and 2b, already judged --------------------------------------
    parts = []
    for path, source in [(IN_ACC, "2a: recomputed from saved predictions"),
                         (IN_SCR, "2b: recomputed from inputs")]:
        d = pd.read_csv(path)
        is_check = (d.family.str.contains("check") | d.metric.str.startswith("per-crystal")
                    | d.metric.str.startswith("rows whose labels") | (d.metric == "n_rows"))
        d.insert(0, "source", source)
        d.insert(1, "row_type", np.where(is_check, "check", "metric"))
        parts.append(d)
    done = pd.concat(parts, ignore_index=True)

    # ---- the stored files 2a/2b did not reach ---------------------------------
    todo = uncovered_files(done)
    for p in todo:
        if p.endswith(".json"):
            stored_json(path_of(p))
        elif p not in CSV_READ and p not in NOT_MELTED:
            raise SystemExit(f"stored file with no reader: {p}")
    stored_tables()
    step46()

    # ---- cross-file checks ------------------------------------------------------
    check_47_recall(done)
    check_82_moduli(done)
    check_control_equals_round9()
    check_gamma_comparison()
    check_summary_vs_json("kaggle/output_matbench_subset/mbsubset_summary.csv",
                          "kaggle/output_matbench_subset/models/summary_mbsubset_*.json",
                          "Kaggle summary tables = run JSONs")
    check_summary_vs_json("kaggle/output_aflow_recipe/aflow_mbrecipe_summary.csv",
                          "kaggle/output_aflow_recipe/models/summary_aflow_mbrecipe_*.json",
                          "Kaggle summary tables = run JSONs")
    check_53()
    check_54_55()
    check_confusion_tables()

    # ---- judge this part's rows ------------------------------------------------
    new = pd.DataFrame(ROWS)
    judged = new.source != STORED_ONLY
    new["abs_diff"], new["rel_diff"], new["status"] = np.nan, np.nan, "stored only"
    a, r, s = judge(new[judged])
    new.loc[judged, "abs_diff"], new.loc[judged, "rel_diff"], new.loc[judged, "status"] = a, r, s
    new = new.drop(columns="decimals")

    # ---- one table ---------------------------------------------------------------
    out = pd.concat([done, new], ignore_index=True)
    out["value"] = out.recomputed.where(out.recomputed.notna(), out.stored)
    # primary: the first row of each (model, metric); later rows compare the same number
    # against another stored file. fillna("") so that blank labels still compare equal.
    key = ["family", "model", "run", "trained_on", "test_set", "split", "target", "metric"]
    out["primary"] = ~out[key].fillna("").astype(str).duplicated()
    cols = (["source", "row_type"] + key[:-1] + ["metric", "n", "value", "recomputed", "stored",
            "stored_in", "abs_diff", "rel_diff", "status", "primary", "note"])
    out = out[cols]
    out.to_csv(OUT_CSV, index=False)

    # ---- print -------------------------------------------------------------------
    w = 100
    print("=" * w)
    print("STEP 95 PART 3 - one final table of every metric, every model")
    print("=" * w)
    print(f"rows written: {len(out)}   (to {os.path.relpath(OUT_CSV, ROOT)})")
    print(f"stored files 2a/2b had not reached: {len(todo)} - every one read here or explained\n")
    print(pd.crosstab([out.source, out.row_type], out.status, margins=True, margins_name="total").to_string())
    m = out[(out.row_type == "metric") & out.primary]
    print(f"\ndistinct metric values (primary == True): {len(m)}, "
          f"of which recomputed and confirmed: {(m.status == 'match').sum()}, "
          f"close: {(m.status == 'close').sum()}, new: {(m.status == 'new').sum()}, "
          f"stored only: {(m.status == 'stored only').sum()}, MISMATCH: {(m.status == 'MISMATCH').sum()}")
    c = out[out.row_type == "cross-file check"]
    print("\ncross-file checks by group:")
    print(pd.crosstab(c.model, c.status, margins=True, margins_name="total").to_string())
    bad = out[out.status.isin(["close", "MISMATCH"]) & out.source.isin([RECOMP3, CROSS])]
    if len(bad):
        print(f"\n{len(bad)} part-3 rows that are not an exact match:")
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(bad[["model", "run", "metric", "recomputed", "stored", "rel_diff", "status",
                       "stored_in"]].to_string(index=False))
    old_bad = out[out.status.isin(["close", "MISMATCH"]) & ~out.source.isin([RECOMP3, CROSS])]
    print(f"\nfrom parts 2a/2b: {(old_bad.status == 'close').sum()} close (each explained in 'note'), "
          f"{(old_bad.status == 'MISMATCH').sum()} MISMATCH")

    # ---- figure ----------------------------------------------------------------
    colours = {"match": "#3b6ea8", "close": "#e69f00", "new": "#59a14f", "stored only": "#a0a0a0",
               "MISMATCH": "#c0392b"}
    fig, ax = plt.subplots(1, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [1.25, 1]})
    # Left: each family's rows as SHARES (every bar is 100%), with the row count written
    # beside it. Families range from 10 to 4,000 rows, and stacking counts on a log axis
    # would draw the segments at misleading lengths.
    rows = out[out.row_type != "cross-file check"]
    tab = pd.crosstab(rows.family, rows.status).reindex(columns=list(colours), fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]           # biggest family at the top
    share = 100 * tab.div(tab.sum(axis=1), axis=0)                 # each row divided by its own total
    share.plot.barh(stacked=True, ax=ax[0], color=[colours[k] for k in share.columns], width=0.8)
    for i, total in enumerate(tab.sum(axis=1)):
        ax[0].text(101, i, f"{total:,}", va="center", fontsize=7)  # {:,} = thousands separator
    ax[0].set(xlim=(0, 112), xlabel="share of the family's rows (%); row count on the right",
              ylabel="", title="Every metric and data-check row, by model family")
    ax[0].tick_params(axis="y", labelsize=7)
    ax[0].legend(title="status", loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=8)
    tab = pd.crosstab(c.model, c.status).reindex(columns=list(colours), fill_value=0)
    tab = tab.loc[:, tab.sum() > 0]
    tab.plot.barh(stacked=True, ax=ax[1], color=[colours[k] for k in tab.columns], width=0.7)
    for i, total in enumerate(tab.sum(axis=1)):
        ax[1].text(total + 0.3, i, str(total), va="center", fontsize=8)
    ax[1].set(xlabel="checks", ylabel="", title="Cross-file checks (two files that must agree)")
    ax[1].tick_params(axis="y", labelsize=8)
    fig.suptitle("Step 95 part 3 - the final metrics table: how every number was checked")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"figure: {os.path.relpath(OUT_PNG, ROOT)}")


if __name__ == "__main__":
    main()
