"""
Step 95, part 1 of the final metrics table: the INVENTORY.

WHAT THIS STEP ANSWERS
----------------------
The goal is ONE csv with every metric the project ever computed, for every model, with
every number re-checked from scratch. Before recomputing anything, we must know every
place where a model was scored. There are two kinds of place:

  predictions    one row per crystal, holding the TRUE value and the MODEL's value.
                 Part 2 recomputes every metric from these files (MAE, MSE, RMSE, R^2,
                 Spearman, and precision / recall / accuracy / F1 for the "is kappa low?"
                 screens). Nothing is trusted; it is all recomputed.
  stored         tables or JSON files holding numbers that were computed earlier. Part 2
                 compares its own recomputed numbers against these and flags any mismatch.

Some files hold numbers that are NOT a model's score (for example, two databases compared
with each other). They are listed as "data_check" so it is clear they were looked at and
left out on purpose. Files replaced by later work are listed as "superseded".

THE SWEEP (the part that makes this trustworthy)
------------------------------------------------
A hand-written list can forget a file. So after the list, the script opens EVERY csv and
json in the result folders and asks: "does this look like it holds metrics or predictions?"
Any file that does, and is not on the list, is printed as UNLISTED. Byte-identical copies
(Kaggle downloads of files already listed) are recognised and marked as copies.

Run (ml_env; pandas only, no torch):
    OMP_NUM_THREADS=1 ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/95_metrics_inventory.py

Writes: results/cgcnn/95_metrics_inventory.csv
"""

# "import X" loads a library. "from X import Y" loads one piece of it. "as pd" is a nickname.
import glob          # finds files by pattern, e.g. "results/*.csv"
import hashlib       # makes a fingerprint (hash) of a file, to spot identical copies
import json          # reads .json files
import re            # "regular expressions": text-pattern matching
from pathlib import Path   # a Path is a file location that knows how to join and test itself

import pandas as pd  # tables (DataFrames): read csv, count rows, look at columns

# Path(__file__) is this script's own location. .resolve() makes it absolute.
# .parents[2] climbs up 2 folders: scripts/cgcnn/ -> scripts/ -> the project root.
ROOT = Path(__file__).resolve().parents[2]
OTHER = Path.home() / "Desktop" / "a new project"     # the coherent-heat project
OUT = ROOT / "results" / "cgcnn" / "95_metrics_inventory.csv"

# ---------------------------------------------------------------------------------------
# Test-set names, written once so every row spells them the same way.
# A metric means nothing without the crystals it was measured on, and models trained on
# DIFFERENT data must never sit in one comparison unlabelled (matbench vs AFLOW).
# ---------------------------------------------------------------------------------------
MB_TEST = "matbench test split (split_seed 42, 1,648 crystals)"
MB_ALL = "matbench train / val / test splits (7,691 / 1,648 / 1,648)"
AF_TEST = "AFLOW test split (835 crystals)"
PHONIX = "PhoNIX phonon-DFT kappa_L (external; 2,520 crystals with all 3 models)"
PHONIX_MLIP = "PhoNIX sample scored with MLIP gamma (617 rows, weighted)"
TABLE1 = "PINK Table 1 measured kappa (45 materials)"
AGL_MATCHED = "matbench test crystals with an AFLOW AGL kappa (441)"

# ---------------------------------------------------------------------------------------
# THE LIST. Each call to add(...) records one source. A "def" makes a function: a named,
# reusable block of code. Arguments with "=None" are optional.
# ---------------------------------------------------------------------------------------
rows = []   # an empty list; add() appends one dictionary per source


def add(kind, pattern, step, model, trained_on, test_set, target, truth_is,
        truth_cols=None, pred_cols=None, note="", base=ROOT):
    """Record one source. `pattern` may contain * so one call can cover several files."""
    # sorted(glob.glob(...)) returns every matching file path, in alphabetical order.
    paths = sorted(glob.glob(str(base / pattern)))
    if not paths:                       # an empty list counts as False in Python
        paths = [str(base / pattern)]   # keep a row anyway, so the missing file is visible
    for p in paths:
        # A dictionary maps names (keys) to values: {"key": value, ...}
        rows.append(dict(kind=kind, step=step, path=p, model=model,
                         trained_on=trained_on, test_set=test_set, target=target,
                         truth_is=truth_is,
                         truth_cols=";".join(truth_cols or []),   # list -> "a;b"
                         pred_cols=";".join(pred_cols or []),
                         note=note))


# ---- 1. Moduli models (bulk K and shear G), trained and tested on matbench ------------
add("predictions", "results/cgcnn/predictions_K_VRH_full.csv", "02-06",
    "CGCNN single (seed 42)", "matbench", MB_ALL, "log10 K_VRH",
    "DFT bulk modulus (matbench)", ["true_log10"], ["pred_log10"])
add("predictions", "results/cgcnn/predictions_G_VRH_full.csv", "02-06",
    "CGCNN single (seed 42)", "matbench", MB_ALL, "log10 G_VRH",
    "DFT shear modulus (matbench)", ["true_log10"], ["pred_log10"])
add("predictions", "results/cgcnn/predictions_K_VRH_ens.csv", "02-06",
    "CGCNN 3-ensemble (seeds 42, 1, 2)", "matbench", MB_ALL, "log10 K_VRH",
    "DFT bulk modulus (matbench)", ["true_log10"], ["pred_log10"])
add("predictions", "results/cgcnn/predictions_G_VRH_ens.csv", "02-06",
    "CGCNN 3-ensemble (seeds 42, 1, 2)", "matbench", MB_ALL, "log10 G_VRH",
    "DFT shear modulus (matbench)", ["true_log10"], ["pred_log10"])
add("predictions", "results/alignn/alignn_bulk_modulus_kv*/prediction_results_test_set.csv",
    "12-18", "ALIGNN single (folder name gives the seed; no suffix = first run)",
    "matbench", MB_TEST, "K_VRH in GPa", "DFT bulk modulus (matbench)",
    ["target"], ["prediction"], "values in GPa, not log10")
add("predictions", "results/alignn/alignn_shear_modulus_gv*/prediction_results_test_set.csv",
    "12-18", "ALIGNN single (folder name gives the seed; no suffix = first run)",
    "matbench", MB_TEST, "G_VRH in GPa", "DFT shear modulus (matbench)",
    ["target"], ["prediction"], "values in GPa, not log10")
add("predictions", "results/alignn/alignn_*/prediction_results_train_set.csv", "12-18",
    "ALIGNN single, TRAIN-split predictions (folder name gives target and seed)", "matbench",
    "matbench train split (7,680 of 7,691: ALIGNN drops a last partial batch)",
    "K_VRH or G_VRH in GPa", "DFT moduli (matbench)", ["target"], ["prediction"],
    "for the train-vs-test gap only; never a test score")
add("predictions", "results/alignn/18_alignn_ensemble_predictions.csv", "18",
    "ALIGNN 3-ensemble", "matbench", MB_TEST, "log10 K and G (column 'target')",
    "DFT moduli (matbench)", ["true_log10"], ["pred_log10"])

# ---- 2. Joint CGCNN (one network predicts K and G together), rounds 4, 5, 7 -----------
# r7 "af" runs were trained on matbench PLUS soft AFLOW crystals; all are TESTED on matbench.
add("predictions", "results/cgcnn/predictions_joint_r[45]_*.csv", "28-33",
    "joint CGCNN, rounds 4-5 (file name = run)", "matbench", MB_ALL, "log10 K and G",
    "DFT moduli (matbench)", ["true_log10_K", "true_log10_G"],
    ["pred_log10_K", "pred_log10_G"], "kappa error is derived from K and G in part 2")
add("predictions", "results/cgcnn/predictions_joint_r7_mb_*.csv", "34",
    "joint CGCNN, round 7, matbench only (a00 = no weighting, a04 = soft weighting)",
    "matbench", MB_ALL, "log10 K and G", "DFT moduli (matbench)",
    ["true_log10_K", "true_log10_G"], ["pred_log10_K", "pred_log10_G"])
add("predictions", "results/cgcnn/predictions_joint_r7_af_*.csv", "34",
    "joint CGCNN, round 7, + AFLOW augmentation", "matbench + 737 soft AFLOW crystals",
    MB_ALL, "log10 K and G", "DFT moduli (matbench)",
    ["true_log10_K", "true_log10_G"], ["pred_log10_K", "pred_log10_G"])

# ---- 3. Composition-only tree baselines (decision tree, random forest, xgboost) --------
add("predictions", "results/cgcnn/baseline_tree/csv/43_baseline_tree_predictions.csv", "43",
    "tree baselines: decision tree, random forest, xgboost",
    "matbench rows: matbench; aflow rows: AFLOW (column 'source')",
    MB_TEST + " / " + AF_TEST, "log10 K, G (both sources); log10 gamma (AFLOW only)",
    "DFT moduli; AFLOW AGL gamma",
    ["K_VRH_true_log10", "G_VRH_true_log10", "gamma_true_log10"],
    ["K_VRH_pred_log10_decision_tree", "K_VRH_pred_log10_random_forest",
     "K_VRH_pred_log10_xgboost", "G_VRH_pred_log10_decision_tree",
     "G_VRH_pred_log10_random_forest", "G_VRH_pred_log10_xgboost",
     "gamma_pred_log10_decision_tree", "gamma_pred_log10_random_forest",
     "gamma_pred_log10_xgboost"])

# ---- 4. Direct kappa CGCNN (structure -> kappa, no Slack model), AFLOW -----------------
add("predictions", "direct_kappa_no_slack/results/csv/01_direct_kappa_test_s*.csv",
    "direct_kappa 01", "direct-kappa CGCNN (file name = seed)", "AFLOW", AF_TEST,
    "log10 kappa", "AFLOW AGL kappa (Debye model, NOT phonon DFT)",
    ["true_log10_kappa"], ["pred_log10_kappa"])
add("predictions", "kaggle/output_aflow_recipe/predictions/01_direct_kappa_test_mbrecipe_s*.csv",
    "direct_kappa 01 (matbench recipe)", "direct-kappa CGCNN, matbench training recipe",
    "AFLOW", AF_TEST, "log10 kappa", "AFLOW AGL kappa (Debye model, NOT phonon DFT)",
    ["true_log10_kappa"], ["pred_log10_kappa"])

# ---- 5. The kappa screens ------------------------------------------------------------
add("predictions", "results/cgcnn/82_test_predictions.csv", "82",
    "ALIGNN, CGCNN-ens, newbase, max3: moduli and kappa chains", "matbench", MB_TEST,
    "K, G in GPa; kappa_L (Slack)",
    "K_VRH, G_VRH = DFT moduli; kappa_DFT = Slack kappa from the DFT moduli (same gamma rule), "
    "NOT a phonon calculation",
    ["K_VRH", "G_VRH", "kappa_DFT"],
    ["K_ALIGNN", "G_ALIGNN", "K_CGCNN-ens", "G_CGCNN-ens", "K_newbase", "G_newbase",
     "ALIGNN_moduli_only", "CGCNN-ens_moduli_only", "newbase_moduli_only",
     "max3_moduli_only", "ALIGNN_full_chain", "CGCNN-ens_full_chain",
     "newbase_full_chain", "max3_full_chain"])
add("predictions", "results/cgcnn/87_phonix_poisson_predictions.csv", "87",
    "Poisson-gamma Slack chain: ALIGNN, CGCNN-ens, newbase, max3",
    "moduli models trained on matbench", PHONIX, "kappa_L, W/m/K",
    "PhoNIX phonon-DFT kappa_L ('klat')", ["klat"],
    ["ALIGNN", "CGCNN-ens", "newbase", "kappa_max3"],
    "some PhoNIX crystals are also matbench training crystals (step 85)")
add("predictions", "results/cgcnn/89_mlip_vs_poisson_crystals.csv", "89",
    "MLIP-gamma chain vs Poisson-gamma chain (max3)", "moduli: matbench; gamma: MLIP phonons",
    PHONIX_MLIP, "kappa_L, W/m/K", "PhoNIX phonon-DFT kappa_L ('klat')", ["klat"],
    ["poisson_max3", "mlip_max3", "mlip1000_max3", "mlip_typical", "mlip_stress"],
    "stratified sample: metrics need the 'weight' column and 'usable' == True")
add("predictions", "results/cgcnn/63_experimental_vs_predicted.csv", "63",
    "CGCNN-ens and ALIGNN chains (derived gamma / paper gamma) and PINK's own values",
    "matbench (moduli)", TABLE1, "kappa_L, W/m/K", "measured kappa ('kappa_exp')",
    ["kappa_exp"],
    ["kappa_cgcnn_derived_gamma", "kappa_alignn_derived_gamma",
     "kappa_cgcnn_equal_footing", "kappa_alignn_equal_footing", "kappa_pink"],
    "column 'provenance' says which of the 45 were matbench training crystals")
add("predictions", "results/cgcnn/agl_target/csv/48_agl_matched_test_crystals.csv", "48, 50",
    "truth file only: run predictions are joined from the files above",
    "-", AGL_MATCHED, "kappa (AGL)", "AFLOW AGL kappa ('agl_kappa')", ["agl_kappa"], [],
    "used to re-score every matbench run against an outside kappa")

# ---- 6. Outside this repo: the direct ALIGNN used as the shortlist's third opinion ------
add("predictions", "results/25_oof_klat_predictions.csv", "CoherentHeat 25",
    "direct ALIGNN structure -> kappa_L (5-fold, out-of-fold)", "PhoNIX (other 4 folds)",
    "PhoNIX, every crystal held out once (5,316)", "log10 kappa_L",
    "PhoNIX phonon-DFT kappa_L", ["true_log_klat"], ["oof_pred"],
    "other repo (~/Desktop/a new project); the 6,641-crystal final model has no held-out set",
    base=OTHER)
add("stored", "results/25_oof_summary.csv", "CoherentHeat 25",
    "direct ALIGNN, per-fold held-out MAE", "PhoNIX", "PhoNIX 5-fold", "log10 kappa_L",
    "PhoNIX phonon-DFT kappa_L", note="other repo", base=OTHER)

# ---- 7. Stored metric tables and JSON summaries (part 2 checks these) -----------------
STORED = [
    # (pattern, step, model, trained_on, test_set, target)
    ("results/cgcnn/metrics_*_VRH_*.csv", "06", "CGCNN single / 3-ens", "matbench", MB_ALL, "K, G"),
    ("results/cgcnn/metrics_summary.csv", "06", "CGCNN single / 3-ens", "matbench", MB_ALL, "K, G"),
    ("results/cgcnn/summary_[KG]_VRH_*.json", "02", "CGCNN single seeds", "matbench", MB_TEST, "K, G"),
    ("results/alignn/18_alignn_ensemble_scores.csv", "18", "ALIGNN and CGCNN, single and 3-ens",
     "matbench", MB_TEST, "K, G"),
    ("results/cgcnn/summary_joint_*.json", "22-34", "every joint CGCNN run (incl. sweeps)",
     "matbench (r7 af: + AFLOW)", MB_ALL, "K, G, K/G, kappa terms, gamma"),
    ("results/joint_comparison.csv", "22-33", "joint CGCNN runs", "matbench", MB_TEST, "K, G, kappa"),
    ("results/gamma_comparison.csv", "37", "round 9 gamma-head runs", "AFLOW", AF_TEST,
     "K, G, gamma, kappa"),
    ("results/cgcnn/summary_37_r9_full_*.json", "37", "round 9 (CGCNN + gamma head)", "AFLOW",
     AF_TEST, "K, G, gamma, kappa"),
    ("results/cgcnn/summary_r8_*.json", "36", "round 8 (adapt / frozen transfer)", "AFLOW",
     AF_TEST, "K, G, gamma, kappa"),
    ("results/cgcnn/summary_gamma3.json", "35", "gamma3 model", "AFLOW", AF_TEST, "gamma"),
    ("results/cgcnn/baseline_tree/csv/43_baseline_tree_scores.csv", "43", "tree baselines",
     "matbench / AFLOW", MB_TEST + " / " + AF_TEST, "K, G, gamma"),
    ("results/cgcnn/baseline_tree/csv/44_confusion_matrix_aflow_low_kappa_metrics.csv", "44",
     "tree baseline low-kappa screen", "AFLOW", AF_TEST, "kappa <= threshold"),
    ("results/cgcnn/baseline_tree/csv/44_confusion_matrix_aflow_low_kappa.csv", "44",
     "tree baseline low-kappa screen", "AFLOW", AF_TEST, "kappa <= threshold"),
    ("results/cgcnn/baseline_tree/csv/46_baseline_vs_r9_*.csv", "46", "tree vs round 9",
     "AFLOW", AF_TEST, "kappa screen"),
    ("results/cgcnn/decile_stability/csv/47_missed_decile_run_summary.csv", "47",
     "every matbench run", "matbench", MB_TEST, "recall@10% (lowest-kappa decile)"),
    ("results/cgcnn/agl_target/csv/48_agl_vs_inhouse_scores.csv", "48", "every matbench run",
     "matbench", AGL_MATCHED, "recall / Spearman vs AGL kappa"),
    ("results/cgcnn/rebaseline/csv/50_rebaseline_scores.csv", "50", "every matbench run",
     "matbench", AGL_MATCHED, "recall / Spearman / MAE vs AGL kappa"),
    ("results/cgcnn/rebaseline/csv/50_rebaseline_bootstrap_ci.csv", "50", "every matbench run",
     "matbench", AGL_MATCHED, "bootstrap CIs of differences"),
    ("results/cgcnn/adjudication/csv/51_adjudication_scores.csv", "51", "round 9 vs tree",
     "AFLOW", AF_TEST, "kappa screen"),
    ("results/cgcnn/adjudication/csv/51_head_to_head.csv", "51", "round 9 vs tree", "AFLOW",
     AF_TEST, "F1 difference"),
    ("results/cgcnn/underfitting/csv/53_underfitting_summary.csv", "53", "CGCNN / trees",
     "matbench / AFLOW", "own test splits", "train vs test error"),
    ("results/cgcnn/underfitting/csv/54_matbench_subset_trees.csv", "54", "trees on matbench subsets",
     "matbench subset", MB_TEST, "K, G"),
    ("results/cgcnn/underfitting/csv/55_*.csv", "55", "CGCNN on matbench subsets",
     "matbench subset", MB_TEST, "K, G"),
    ("kaggle/output_matbench_subset/mbsubset_summary.csv", "55", "CGCNN on matbench subsets",
     "matbench full / subset", MB_TEST, "K, G"),
    ("kaggle/output_matbench_subset/models/summary_mbsubset_*.json", "55",
     "CGCNN on matbench subsets, per seed", "matbench full / subset", MB_TEST, "K, G"),
    ("kaggle/output_aflow_recipe/aflow_mbrecipe_summary.csv", "53", "CGCNN, matbench recipe on AFLOW",
     "AFLOW", AF_TEST, "K, G, direct kappa"),
    ("kaggle/output_aflow_recipe/models/summary_*.json", "53",
     "CGCNN, matbench recipe on AFLOW, per seed", "AFLOW", AF_TEST, "K, G, direct kappa"),
    ("results/alignn/58_rho_decomposition.csv", "58", "ALIGNN vs round 9 on GNoME",
     "matbench / AFLOW", "GNoME (no truth: model-vs-model agreement)", "Spearman between models"),
    ("results/cgcnn/60_residual_corr_table.csv", "60", "CGCNN / ALIGNN / joint", "matbench",
     MB_TEST, "K, G residual correlation"),
    ("results/cgcnn/65_leakage_audit.csv", "65", "ALIGNN ens, CGCNN ens", "matbench", MB_TEST,
     "K, G split by formula-seen-in-train"),
    ("results/cgcnn/82_confusion_matrices.csv", "82", "ALIGNN, CGCNN-ens, newbase, max3",
     "matbench", MB_TEST, "kappa <= 1 and <= 0.5 screen"),
    ("results/cgcnn/82_accuracy_in_range.csv", "82", "ALIGNN, CGCNN-ens, newbase", "matbench",
     MB_TEST, "K, G within 25% / 2x"),
    ("results/cgcnn/82_trust_table.csv", "82", "max3 screen", "matbench", MB_TEST,
     "confirmed-low fraction"),
    ("results/cgcnn/87_phonix_poisson_confusion.csv", "87", "Poisson chain", "matbench",
     PHONIX, "kappa <= cutoff screen"),
    ("results/cgcnn/87_precision_vs_cutoff.csv", "87", "Poisson chain", "matbench", PHONIX,
     "precision vs cutoff"),
    ("results/cgcnn/89_mlip_vs_poisson_summary.csv", "89", "MLIP vs Poisson chain", "matbench",
     PHONIX_MLIP, "precision / recall"),
    ("results/cgcnn/89_mlip_vs_poisson_secondary.csv", "89", "MLIP vs Poisson chain", "matbench",
     PHONIX_MLIP, "median ratio, Spearman"),
    ("direct_kappa_no_slack/results/csv/02_direct_vs_slack_scores.csv", "direct_kappa 02",
     "direct kappa vs Slack chain", "AFLOW", AF_TEST, "kappa screen"),
    ("direct_kappa_no_slack/results/csv/02_direct_vs_slack_bootstrap.csv", "direct_kappa 02",
     "direct kappa vs Slack chain", "AFLOW", AF_TEST, "F1 difference"),
    ("direct_kappa_no_slack/results/csv/direct_vs_control_summary.csv", "direct_kappa 01",
     "direct kappa vs control", "AFLOW", AF_TEST, "kappa, K, G, gamma"),
    ("direct_kappa_no_slack/models/summary_*.json", "direct_kappa 01", "direct kappa / control",
     "AFLOW", AF_TEST, "kappa"),
    ("results/cgcnn/calibration_check.csv", "08", "CGCNN-ens uncertainty intervals", "matbench",
     MB_TEST, "interval coverage"),
]
# A "for" loop runs its indented block once per item. Each item here is a tuple (a fixed
# list in round brackets); "pattern, step, ... = item" unpacks it into named variables.
for pattern, step, model, trained_on, test_set, target in STORED:
    add("stored", pattern, step, model, trained_on, test_set, target, "see file")

# ---- 8. Looked at and left out on purpose --------------------------------------------
NOT_MODEL = [
    ("results/cgcnn/59_cross_source_label_noise.csv", "59", "AFLOW vs matbench labels, no model"),
    ("results/cgcnn/agl_target/csv/49_*.csv", "49", "AGL gamma vs measured gamma, no model"),
    ("results/cgcnn/agl_target/csv/48_match_diagnostics.csv", "48", "matching bookkeeping"),
    ("results/cgcnn/rebaseline/csv/50_match_diagnostics.csv", "50", "matching bookkeeping"),
    ("results/cgcnn/baseline_tree/csv/44_correlation*.csv", "44", "feature / target correlations"),
    ("results/cgcnn/baseline_tree/csv/44_feature_correlation_184dim.csv", "44", "feature correlations"),
    ("results/cgcnn/underfitting/csv/56_*.csv", "56", "dataset shape comparison, no model"),
    ("results/cgcnn/80_gamma_reproducibility.csv", "80", "MLIP gamma re-run spread, no truth"),
    ("results/cgcnn/kappa_comparison.csv", "07", "ours vs PINK's PREDICTED kappa: agreement, not accuracy"),
    ("results/cgcnn/pink_*.csv", "07", "PINK's own predictions, no truth"),
    ("results/cgcnn/decile_stability/csv/47_missed_decile_per_crystal.csv", "47",
     "per-crystal detail behind step 47's recall (which crystals were missed)"),
    ("results/cgcnn/decile_stability/csv/47_missed_by_every_ensemble.csv", "47",
     "per-crystal detail behind step 47's recall (missed by every ensemble)"),
    ("results/cgcnn/decile_stability/csv/47_missed_decile_pairwise_*.csv", "47",
     "overlap between runs' missed crystals, not a score"),
    ("results/families/*/*.csv", "40", "per-family slices of the listed predictions"),
    ("results/alignn/alignn_*/Train_results.json", "12-18", "ALIGNN's own dump: ONE batch only (120 rows)"),
    ("results/alignn/alignn_*/Val_results.json", "12-18", "ALIGNN's own dump: ONE batch only (25 rows)"),
    ("results/alignn/alignn_*/Test_results.json", "12-18",
     "ALIGNN's own dump of the test predictions (same values as the csv: checked below)"),
]
for pattern, step, why in NOT_MODEL:
    add("data_check", pattern, step, "-", "-", "-", "-", "-", note=why)

SUPERSEDED = [
    ("results/cgcnn/archive/cpu-150epoch/*.csv", "early CPU run, replaced by the 200-epoch run"),
    ("results/cgcnn/archive/cpu-150epoch/*.json", "early CPU run, replaced by the 200-epoch run"),
    ("colab/alignn_output/extracted/*/*.csv", "40-crystal ALIGNN smoke test on Colab"),
    ("colab/alignn_output/extracted/*/*.json", "40-crystal ALIGNN smoke test on Colab"),
    ("results/cgcnn/table1_comparison.csv", "replaced by step 63"),
    ("results/cgcnn/table1_kappa_predictions.csv", "replaced by step 63"),
]
for pattern, why in SUPERSEDED:
    add("superseded", pattern, "-", "-", "-", "-", "-", "-", note=why)

inv = pd.DataFrame(rows)   # list of dictionaries -> table, one row per source

# Guard: two patterns must never pick up the same file, or it would be counted twice.
dup = inv.path[inv.path.duplicated()]
if len(dup):
    raise SystemExit("listed twice:\n  " + "\n  ".join(dup))

# ---------------------------------------------------------------------------------------
# CHECK every listed file: does it exist, how many rows, are the named columns really there?
# ---------------------------------------------------------------------------------------


def n_rows_and_columns(path):
    """Return (row count, list of column/key names) for a csv or json file."""
    if path.endswith(".csv"):
        d = pd.read_csv(path, low_memory=False)
        return len(d), list(d.columns)
    with open(path) as f:          # "with" opens the file and closes it again afterwards
        d = json.load(f)
    if isinstance(d, list):        # a list of records, e.g. ALIGNN's Test_results.json
        # (some ALIGNN dumps are lists of plain lists, which have no column names)
        return len(d), list(d[0].keys()) if d and isinstance(d[0], dict) else []
    return 1, flat_keys(d)


def flat_keys(d, prefix=""):
    """All key names in a nested dictionary, e.g. {"test": {"mae": 1}} -> ["test_mae"].
    Needed because the summary JSONs keep their metrics one or two levels down."""
    keys = []
    for k, v in d.items():
        name = f"{prefix}_{k}" if prefix else str(k)
        keys.append(name)
        if isinstance(v, dict):
            keys += flat_keys(v, name)   # a function calling itself: "recursion"
    return keys


exists, nrows, missing_cols = [], [], []
for _, r in inv.iterrows():        # iterrows() walks the table one row at a time
    if not Path(r.path).exists():
        exists.append(False); nrows.append(None); missing_cols.append("")
        continue                   # skip the rest of this loop pass
    n, cols = n_rows_and_columns(r.path)
    wanted = [c for c in (r.truth_cols + ";" + r.pred_cols).split(";") if c]
    # A "list comprehension": [x for x in things if test] builds a list in one line.
    exists.append(True); nrows.append(n)
    missing_cols.append(";".join(c for c in wanted if c not in cols))
inv["exists"], inv["n_rows"], inv["missing_columns"] = exists, nrows, missing_cols

# ALIGNN wrote each test prediction twice: a csv and its own Test_results.json. Check the
# two really hold the same numbers before treating the json as a duplicate.
json_gap = []
for jp in sorted(glob.glob(str(ROOT / "results/alignn/alignn_*/Test_results.json"))):
    j = pd.DataFrame(json.load(open(jp)))
    j["pred_json"] = j.pred_out.str[0]          # each value is stored as a 1-item list
    c = pd.read_csv(Path(jp).with_name("prediction_results_test_set.csv"),
                    skipinitialspace=True)       # the csv has a space after each comma
    m = c.merge(j, on="id")                      # line rows up by crystal id
    gap = (m.prediction - m.pred_json).abs().max()
    json_gap.append(gap)
    if len(m) != len(c) or len(m) != len(j) or gap > 1e-3:
        raise SystemExit(f"ALIGNN json and csv disagree: {jp} (max gap {gap})")

# ---------------------------------------------------------------------------------------
# THE SWEEP: open every csv/json in the result folders and flag unlisted ones.
# ---------------------------------------------------------------------------------------
METRIC = re.compile(r"(^|_)(mae|mse|rmse|r2|spearman|pearson|rho|recall\w*|precision|"
                    r"accuracy|f1|tp|fp|fn|tn)($|_)", re.I)
TRUTH = re.compile(r"true|target|klat|kappa_exp|kappa_dft|agl_kappa", re.I)
PRED = re.compile(r"pred|prediction", re.I)


def fingerprint(path):
    """md5 of the file's bytes: two files with the same fingerprint are identical copies."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


listed = set(inv.path)                            # a set: fast "is it in here?" lookups
seen_hash = {fingerprint(p): p for p in inv.path[inv.exists]}  # dict: hash -> first path
SWEEP_DIRS = ["results", "direct_kappa_no_slack", "kaggle/output_*", "colab/alignn_output"]
swept = []
for d in SWEEP_DIRS:
    for ext in ("csv", "json"):
        swept += glob.glob(str(ROOT / d / "**" / f"*.{ext}"), recursive=True)

unlisted = []
# Sort so that results/ comes before kaggle/: the copy is then always the Kaggle download.
for p in sorted(set(swept), key=lambda x: ("/kaggle/" in x, x)):
    name = Path(p).name
    if p in listed or p == str(OUT) or name.startswith(("history", "config", "ids_train", "mad")):
        continue        # listed already, or a training curve / run config (not a final score)
    if re.search(r"gnome|jarvis|_dft_queue|shortlist|prior_|candidates|census|plan|cost", name):
        continue        # screening outputs: predictions for crystals with NO truth value
    try:
        n, cols = n_rows_and_columns(p)
    except Exception:   # unreadable file: skip it rather than crash the whole sweep
        continue
    has_metric = any(METRIC.search(str(c)) for c in cols)
    has_pairs = any(TRUTH.search(str(c)) for c in cols) and any(PRED.search(str(c)) for c in cols)
    if not (has_metric or has_pairs):
        continue
    h = fingerprint(p)
    copy_of = seen_hash.get(h, "")       # .get returns "" if this content was not seen yet
    seen_hash.setdefault(h, p)           # remember it, so a later copy points back here
    unlisted.append(dict(path=p, n_rows=n, copy_of=copy_of,
                         looks_like="metrics" if has_metric else "predictions"))
unl = pd.DataFrame(unlisted, columns=["path", "n_rows", "copy_of", "looks_like"])

# Save: the listed sources, then the sweep's finds marked kind = "UNLISTED" / "copy".
for u in unl.itertuples():
    rows_extra = dict(kind="copy" if u.copy_of else "UNLISTED", step="", path=u.path,
                      model="", trained_on="", test_set="", target="", truth_is="",
                      truth_cols="", pred_cols="", note=("copy of " + u.copy_of) if u.copy_of
                      else "looks like " + u.looks_like, exists=True, n_rows=u.n_rows,
                      missing_columns="")
    inv = pd.concat([inv, pd.DataFrame([rows_extra])], ignore_index=True)

# Shorter paths in the csv: drop everything up to the project folder name.
for col in ("path", "note"):
    inv[col] = (inv[col].str.replace(str(ROOT) + "/", "", regex=False)
                        .str.replace(str(OTHER) + "/", "[coherent-heat repo] ", regex=False))
inv.to_csv(OUT, index=False)

# ---------------------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------------------
print(f"wrote {OUT.relative_to(ROOT)}  ({len(inv)} rows)\n")
print("sources by kind:")
print(inv.kind.value_counts().to_string(), "\n")
bad = inv[(inv.kind != "UNLISTED") & (inv.kind != "copy") & ~inv.exists.astype(bool)]
print(f"listed files that do NOT exist: {len(bad)}")
for p in bad.path:
    print("   ", p)
bad = inv[inv.missing_columns.fillna("") != ""]
print(f"prediction files missing a named column: {len(bad)}")
for r in bad.itertuples():
    print(f"    {r.path}: {r.missing_columns}")
print(f"ALIGNN Test_results.json vs csv: {len(json_gap)} pairs, largest gap "
      f"{max(json_gap):.1e} GPa (the csv is rounded to 6 decimals)")
print(f"\nUNLISTED files the sweep found (not copies): {(inv.kind == 'UNLISTED').sum()}")
for r in inv[inv.kind == "UNLISTED"].itertuples():
    print(f"    {r.path}  [{r.n_rows} rows, {r.note}]")

print("\nprediction files (part 2 recomputes metrics from these):")
pr = inv[inv.kind == "predictions"]
for r in pr.itertuples():
    print(f"  {r.n_rows:>6}  {r.path}")
