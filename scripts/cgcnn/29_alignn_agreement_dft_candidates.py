#!/usr/bin/env python3
"""
STEP 29 - Cross-model-agreement DFT candidates, two kappa_L thresholds.
================================================================================

    python scripts/cgcnn/29_alignn_agreement_dft_candidates.py

WHAT THIS IS
-------------
28_alignn_small_cell_filter.py scored all 1,215 small-cell (<10 atom)
candidates with both CGCNN and ALIGNN independently. This script pulls out
the candidates BOTH models agree are low-kappa - the strongest evidence
available in this project short of an actual DFT run, since two
architectures sharing nothing but training data landing on the same verdict
is not something either model's own (currently miscalibrated - see
25_calibration_check.py) uncertainty interval can fake.

Two output files, two thresholds, same agreement logic:
    kappa_L < 1  W/m/K  - this project's own standard low-kappa cutoff
    kappa_L < 5  W/m/K  - a looser net, still clearly low by any normal
                          material's standard (typical crystalline solids
                          run 10-100+ W/m/K), for a larger DFT candidate
                          pool if the strict cutoff turns out too narrow

"AGREE" MEANS BOTH MODELS INDIVIDUALLY CLEAR THE THRESHOLD, NOT THE AVERAGE
--------------------------------------------------------------------------------
A candidate only qualifies if CGCNN's OWN Kappa_cal and ALIGNN's OWN
Kappa_cal both independently fall under the threshold - not if their average
does. A candidate where one model says 0.1 and the other says 8 would pass
an average-based filter at the 5x threshold despite one model flatly
disagreeing; requiring both individually is the stricter, more defensible
reading of "both models agree this is good."

SORTED BY THE WORSE OF THE TWO MODELS, NOT EITHER ONE ALONE
------------------------------------------------------------------
Ranked by max(CGCNN Kappa_cal, ALIGNN Kappa_cal) ascending - the candidate's
own worst-case across the two independent opinions. This rewards agreement
itself, not just whichever model happened to be most optimistic; a candidate
both models rate very low ranks above one where only the friendlier model's
number was low.

ATOM COUNT
------------
Every row is already <10 atoms by construction - both inputs
(gnome_small_cells_alignn.csv) come from 19_small_cells.py's own <10-atom
filter. Asserted explicitly below rather than assumed, so a future change
to the upstream file's scope cannot silently violate the DFT-cost constraint
this list exists for.
"""

import os               # path joining/checking

import pandas as pd     # CSV I/O, DataFrame filtering/merging/sorting

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 28_alignn_small_cell_filter.py's per-candidate CGCNN+ALIGNN scores
ALIGNN_ALL_CSV = os.path.join(RESULTS, "28_gnome_small_cells_alignn.csv")
# input: 17_spacegroups_general.py's per-material space-group/crystal-system table
SPACEGROUP_CSV = os.path.join(RESULTS, "17_gnome_screen_spacegroups.csv")
# outputs: this script's own two agreement shortlists, "29_" prefix so they
# are distinguishable from any other script's output in the same directory
# (read downstream by 30_compare_with_paper_screens.py)
OUT_LT1_CSV = os.path.join(RESULTS, "29_gnome_dft_candidates_agreement_lt1.csv")
OUT_LT5_CSV = os.path.join(RESULTS, "29_gnome_dft_candidates_agreement_lt5.csv")


def agreement_subset(df, threshold):
    """Rows where BOTH CGCNN's and ALIGNN's own Kappa_cal individually clear
    the threshold - see module docstring for why individual, not average.
    Also requires alignn_prediction_reliable (28_alignn_small_cell_filter.py):
    a handful of candidates have ALIGNN's raw regression output hit its
    1e-3 GPa floor - a degenerate extrapolation failure, not a genuine
    low-kappa confirmation - and including those would let exactly the
    kind of untrustworthy "agreement" this list exists to rule out back in."""
    both = ((df["Kappa_cal (W m-1 K-1)"] <= threshold) &   # CGCNN's own prediction clears the bar
           (df["ALIGNN_Kappa_cal"] <= threshold) &          # ALIGNN's own prediction also clears it
           df["alignn_prediction_reliable"])                # and ALIGNN's output isn't a degenerate floor-hit
    sub = df[both].copy()                                   # boolean-mask filter, then an explicit copy to silence pandas' SettingWithCopyWarning
    sub["worse_model_Kappa_cal"] = sub[["Kappa_cal (W m-1 K-1)", "ALIGNN_Kappa_cal"]].max(axis=1)  # row-wise max across the two model columns
    return sub.sort_values("worse_model_Kappa_cal").reset_index(drop=True)  # ascending by worst-case; reindex 0..N-1


def main():
    print("=== Cross-model-agreement DFT candidates (CGCNN + ALIGNN, both < threshold) ===\n")
    df = pd.read_csv(ALIGNN_ALL_CSV).dropna(subset=["ALIGNN_Kappa_cal"])  # drop rows ALIGNN never scored
    print(f"Loaded {len(df)} small-cell candidates scored by both models from {ALIGNN_ALL_CSV}")

    assert (df["Number of Atoms"] < 10).all(), \
        "Found a candidate with >=10 atoms - upstream small-cell filter was violated"
    print(f"  Atom count range: {df['Number of Atoms'].min()}-{df['Number of Atoms'].max()} "
         f"(all < 10, confirmed)")

    # pull in only the three columns needed for the join
    sg = pd.read_csv(SPACEGROUP_CSV, usecols=["material_id", "space_group", "crystal_system"])
    df = df.merge(sg, on="material_id", how="left")  # left join: keep every candidate even if the space-group lookup misses

    for threshold, out_path in [(1.0, OUT_LT1_CSV), (5.0, OUT_LT5_CSV)]:  # run the same logic at both thresholds
        sub = agreement_subset(df, threshold)
        sub.to_csv(out_path, index=False)  # index=False: don't write the pandas row-number column
        print(f"\n{'=' * 60}")
        print(f"kappa_L < {threshold} W/m/K, BOTH models agree: {len(sub)} candidates")
        print(f"{'=' * 60}")
        print(f"Wrote {out_path}")

        if len(sub):  # only print highlights if the threshold actually kept any rows
            top = sub.iloc[0]  # first row after sorting = strongest agreement candidate
            print(f"  Best: {top.formula} ({int(top['Number of Atoms'])} atoms, "
                 f"{top.crystal_system}, {top.space_group}) - "
                 f"CGCNN={top['Kappa_cal (W m-1 K-1)']:.4f}, ALIGNN={top.ALIGNN_Kappa_cal:.4f}")
            print(f"\n  Top 5:")
            for _, row in sub.head(5).iterrows():  # iterate the 5 best rows for a quick console preview
                print(f"    {row.formula:16s} CGCNN={row['Kappa_cal (W m-1 K-1)']:.4f}  "
                     f"ALIGNN={row.ALIGNN_Kappa_cal:.4f}  worse={row.worse_model_Kappa_cal:.4f}  "
                     f"{int(row['Number of Atoms'])} atoms, {row.crystal_system}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
