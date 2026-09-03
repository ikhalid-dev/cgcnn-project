#!/usr/bin/env python3
"""
STEP 26 - Re-score the DFT shortlist with a calibrated uncertainty interval.
================================================================================

    python scripts/cgcnn/26_recalibrated_dft_shortlist.py

WHAT THIS IS
-------------
25_calibration_check.py found the raw ensemble-spread interval is badly
overconfident (claimed 90% coverage, actual ~48-51% on 1,648 held-out
matbench test crystals) and derived a fix - widen the interval by a factor
fit on the VAL split (K: 4.27x, G: 4.54x, on top of the untouched
normal-quantile 1.645) - and checked it out-of-sample on TEST (~90% actual
coverage, target 90%). This script applies that SAME, already-validated
factor to the 267-candidate small-cell DFT shortlist
(gnome_small_cell_dft_ranking.csv) and its 24-row oxide subset, and re-runs
the ranking/bucketing that was built on the old, overconfident numbers.

WHAT DOES NOT CHANGE
------------------------
Nothing about the model. K_VRH_pred, G_VRH_pred, and the point-estimate
Kappa_cal are byte-for-byte unchanged - this only widens the ensemble-spread
INPUT to the Monte Carlo uncertainty step (07_predict_kappa.py's
monte_carlo_kappa(), reused unchanged, not reimplemented), which changes
Kappa_cal_p05/p95 and therefore the confidence bucket a candidate falls
into. No retraining, no architecture change - purely a correction to how
wide the error bars are reported.

WHY THESE EXACT FACTORS, NOT RE-DERIVED HERE
-------------------------------------------------
The factors (K: 4.27x, G: 4.54x) are read directly from
results/cgcnn/calibration_check.csv's 90%-nominal row - the same file
25_calibration_check.py already wrote, having derived them from the VAL
split and validated them on the disjoint TEST split. Re-deriving them again
here would risk a silent drift between the two scripts; reading the
already-checked number is the same "reuse, not reimplementation" principle
used throughout this project.
"""

import os                             # path joining/checking
import sys                            # sys.path manipulation for the importlib workaround below
from importlib import import_module   # loads a module whose filename starts with a digit

# torch first - see cgcnn_scratch/data.py for why (duplicate libiomp5 OpenMP
# init crash otherwise, since 07_predict_kappa.py imports torch internally).
import torch  # noqa: F401           # imported only for its side effect (OpenMP init order), not used directly here

import pandas as pd                   # CSV I/O and DataFrame column arithmetic

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 22_best_dft_candidate.py's Kappa_cal_p95-ranked small-cell shortlist
DFT_RANKING_CSV = os.path.join(RESULTS, "22_gnome_small_cell_dft_ranking.csv")
# input: 24_small_cell_oxides_for_dft.py's oxide-only subset of the above
OXIDES_CSV = os.path.join(RESULTS, "24_gnome_small_cell_oxides_for_dft.csv")
# input: 25_calibration_check.py's validated interval-widening factors
CALIBRATION_CSV = os.path.join(RESULTS, "25_calibration_check.csv")
# outputs: this script's own recalibrated versions of the two shortlists above, "26_" prefix
OUT_DFT_CSV = os.path.join(RESULTS, "26_gnome_small_cell_dft_ranking_recalibrated.csv")
OUT_OXIDES_CSV = os.path.join(RESULTS, "26_gnome_small_cell_oxides_for_dft_recalibrated.csv")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # put this file's own directory first on the import path
_kappa = import_module("07_predict_kappa")  # import by string name since "07_predict_kappa" isn't a valid `import` identifier


def confidence_bucket(ratio):
    if ratio < 2:
        return "tight"
    if ratio <= 5:
        return "moderate"
    return "wide"


def recalibrate(df, k_factor, g_factor):
    """Widen the ensemble spread by the validated factor, re-run the
    project's own unchanged Monte Carlo, recompute the ratio/bucket. Returns
    a new dataframe re-sorted by the recalibrated Kappa_cal_p95."""
    df = df.copy()  # don't mutate the caller's dataframe in place
    df["K_VRH_spread_log10"] = df["K_VRH_spread_log10"] * k_factor  # widen the bulk-modulus ensemble spread by the validated factor
    df["G_VRH_spread_log10"] = df["G_VRH_spread_log10"] * g_factor  # same, for the shear-modulus spread

    mc = _kappa.monte_carlo_kappa(df, n_samples=2000, seed=0)  # reuse 07's own Monte Carlo sampler, now fed the widened spreads
    df["Kappa_cal_p05"] = mc["Kappa_cal_p05"]   # 5th-percentile kappa across the Monte Carlo draws
    df["Kappa_cal_p50"] = mc["Kappa_cal_p50"]   # median kappa across the draws
    df["Kappa_cal_p95"] = mc["Kappa_cal_p95"]   # 95th-percentile (pessimistic) kappa across the draws
    df["p95_p05_ratio"] = df["Kappa_cal_p95"] / df["Kappa_cal_p05"]  # recompute the interval-width ratio with the new p05/p95
    df["confidence"] = df["p95_p05_ratio"].apply(confidence_bucket)  # re-bucket using the recalibrated ratio
    return df.sort_values("Kappa_cal_p95").reset_index(drop=True)  # re-rank ascending by the recalibrated pessimistic estimate


def main():
    print("=== Re-scoring the DFT shortlist with a calibrated uncertainty interval ===\n")
    cal = pd.read_csv(CALIBRATION_CSV)
    row90 = cal[cal["nominal"] == 0.90].iloc[0]  # the row for the 90%-nominal coverage target
    k_factor = row90["K_VRH_scale_factor"] / 1.645  # factor on top of the untouched normal-quantile 1.645 (90% two-sided z-score)
    g_factor = row90["G_VRH_scale_factor"] / 1.645
    print(f"Recalibration factors (from calibration_check.csv, validated on held-out test): "
         f"K x{k_factor:.2f}, G x{g_factor:.2f}")

    for label, path, out_path in [("small-cell shortlist", DFT_RANKING_CSV, OUT_DFT_CSV),
                                  ("oxide subset", OXIDES_CSV, OUT_OXIDES_CSV)]:  # apply the identical recalibration to both shortlists
        old = pd.read_csv(path)
        old["confidence_old"] = old["p95_p05_ratio"].apply(confidence_bucket)  # bucket using the pre-recalibration ratio, for the before/after comparison

        new = recalibrate(old, k_factor, g_factor)
        new.to_csv(out_path, index=False)  # index=False: don't write the pandas row-number column

        print(f"\n{'=' * 70}")
        print(f"{label.upper()} ({len(new)} candidates) - before vs. after recalibration")
        print(f"{'=' * 70}")
        old_buckets = old["confidence_old"].value_counts()  # counts per bucket, before recalibration
        new_buckets = new["confidence"].value_counts()      # counts per bucket, after recalibration
        for bucket in ["tight", "moderate", "wide"]:
            print(f"  {bucket:10s}  before: {old_buckets.get(bucket, 0):4d}   "
                 f"after: {new_buckets.get(bucket, 0):4d}")

        old_top = old.sort_values("Kappa_cal_p95").iloc[0]  # best candidate under the OLD (overconfident) interval
        new_top = new.iloc[0]                               # best candidate under the NEW (recalibrated) interval; new is already sorted
        print(f"\n  Old #1 (overconfident interval): {old_top.formula}  "
             f"p95={old_top.Kappa_cal_p95:.4f}  ratio={old_top.p95_p05_ratio:.2f}x "
             f"({old_top.confidence_old})")
        print(f"  New #1 (calibrated interval)   : {new_top.formula}  "
             f"p95={new_top.Kappa_cal_p95:.4f}  ratio={new_top.p95_p05_ratio:.2f}x "
             f"({new_top.confidence})")

        # Did the previous #1 survive being re-scored, and where does it rank now?
        old_top_new_row = new[new["material_id"] == old_top.material_id]  # find the old winner's row in the recalibrated table
        if len(old_top_new_row):
            new_rank = new.index[new["material_id"] == old_top.material_id][0] + 1  # 0-indexed position -> 1-indexed rank
            r = old_top_new_row.iloc[0]
            print(f"  Old #1 under the calibrated interval: rank #{new_rank} of {len(new)}, "
                 f"now p95={r.Kappa_cal_p95:.4f}, ratio={r.p95_p05_ratio:.2f}x ({r.confidence})")

        print(f"\n  Wrote {out_path}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
