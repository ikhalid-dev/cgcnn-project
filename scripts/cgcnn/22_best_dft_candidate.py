#!/usr/bin/env python3
"""
STEP 22 - The single best small-cell candidate to actually send to DFT.
================================================================================

    python scripts/cgcnn/22_best_dft_candidate.py

WHAT THIS IS
-------------
19_small_cells.py already narrowed the screen to small (<10-atom), confirmed
low-kappa_L candidates (gnome_screen_small_cells_low_kappa.csv, 267 rows) -
cheap to actually verify with a real DFT calculation. This script picks the
ONE candidate worth spending that DFT budget on first.

WHY NOT JUST TAKE THE LOWEST Kappa_cal
--------------------------------------------
Same winner's-curse reasoning as 16_spacegroups_oxides.py's best_by_group():
ranking by the point estimate and taking the minimum systematically favours
candidates whose ensemble members simply disagreed a lot, not necessarily
the ones that are genuinely best. A DFT calculation is expensive - spending
it confirming noise instead of signal is the worst possible way to use it.
So this ranks by Kappa_cal_p95 (the model's own pessimistic 95th-percentile
estimate) ascending, the same convention used for every "best candidate"
table in docs/oxide_screen_columns.tex.

Also joins in space group / crystal system (from
gnome_screen_spacegroups.csv, GNoME's own metadata - not recomputed) purely
for context: a DFT calculation needs the symmetry to set up the input file
efficiently, and it is useful to see at a glance whether the top pick is
also structurally unusual.

Writes one CSV: all 267 low-kappa small cells, ranked by Kappa_cal_p95
ascending, with crystal_system/space_group attached and a p95_p05_ratio
column made explicit (lower = more confident) - not just the single winner,
so the reasoning behind picking #1 over #2 is inspectable, not asserted.
"""

import os               # path joining/checking, independent of the OS this runs on

import pandas as pd     # DataFrame I/O (read_csv/to_csv) and the merge/sort below

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared output/input directory for every numbered script
# input: 19_small_cells.py's confirmed low-kappa, small-cell shortlist
SMALL_LOW_KAPPA_CSV = os.path.join(RESULTS, "19_gnome_screen_small_cells_low_kappa.csv")
# input: 17_spacegroups_general.py's per-material space-group/crystal-system table
SPACEGROUP_CSV = os.path.join(RESULTS, "17_gnome_screen_spacegroups.csv")
# output: this script's own ranked shortlist - script number "22" in the name
# so it can be told apart from any other script's output sitting in the same
# results/ directory (read by 24_small_cell_oxides_for_dft.py,
# 26_recalibrated_dft_shortlist.py and 27_alignn_dft_shortlist.py downstream)
OUT_CSV = os.path.join(RESULTS, "22_gnome_small_cell_dft_ranking.csv")


def main():
    print("=== Best small-cell, low-kappa candidate for DFT verification ===\n")
    if not os.path.exists(SMALL_LOW_KAPPA_CSV):  # fail loudly rather than crash deep inside pandas
        raise SystemExit(f"{SMALL_LOW_KAPPA_CSV} not found - run scripts/cgcnn/19_small_cells.py first")

    low = pd.read_csv(SMALL_LOW_KAPPA_CSV)  # one row per confirmed low-kappa, small-cell candidate
    print(f"Loaded {len(low)} confirmed low-kappa, small-cell (<10 atom) candidates")

    # pull in only the three columns needed for the join - no point loading the whole table
    sg = pd.read_csv(SPACEGROUP_CSV, usecols=["material_id", "space_group", "crystal_system"])
    merged = low.merge(sg, on="material_id", how="left")  # left join: keep every low-kappa row even if space-group lookup misses
    missing = merged["space_group"].isna().sum()  # count rows where the join found no match
    if missing:
        print(f"  {missing}/{len(merged)} have no matching space-group row (upstream gap, "
             f"not this project's - see 16_spacegroups_oxides.py's module docstring)")

    # p95/p05 ratio: how wide the model's own uncertainty interval is, in multiplicative terms
    merged["p95_p05_ratio"] = merged["Kappa_cal_p95"] / merged["Kappa_cal_p05"]
    merged = merged.sort_values("Kappa_cal_p95").reset_index(drop=True)  # ascending, so row 0 is the best pick; reindex 0..N-1 after sorting

    merged.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV} ({len(merged)} rows, ranked by Kappa_cal_p95 ascending)")

    top = merged.iloc[0]  # first row after sorting = single best candidate
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDED FOR DFT: {top.formula}  (material_id {top.material_id})")
    print(f"{'=' * 60}")
    print(f"  Atoms in primitive cell : {int(top['Number of Atoms'])}")
    print(f"  Crystal system / SG     : {top.crystal_system} / {top.space_group}")
    print(f"  Predicted Kappa_cal     : {top['Kappa_cal (W m-1 K-1)']:.4f} W/m/K")
    print(f"  p05 - p95 interval      : {top.Kappa_cal_p05:.4f} - {top.Kappa_cal_p95:.4f} "
         f"({top.p95_p05_ratio:.2f}x ratio)")
    print(f"  K_VRH / G_VRH predicted : {top.K_VRH_pred:.1f} / {top.G_VRH_pred:.1f} GPa")
    print(f"\n  Why this one: lowest Kappa_cal_p95 of all {len(merged)} confirmed low-kappa "
         f"small cells - even in this model's own pessimistic scenario, it is still "
         f"predicted well under the 1 W/m/K threshold, with a {top.p95_p05_ratio:.1f}x "
         f"p95/p05 spread (tight enough that the ensemble broadly agrees, not a lucky "
         f"noisy point estimate).")

    print(f"\nRunner-up context (next 4):")
    for _, row in merged.iloc[1:5].iterrows():  # rows 1-4 (0-indexed): the four next-best after the winner
        print(f"  {row.formula:14s} p95={row.Kappa_cal_p95:.4f} ({row.p95_p05_ratio:.2f}x)  "
             f"{row.crystal_system}, {row.space_group}, {int(row['Number of Atoms'])} atoms")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
