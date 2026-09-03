#!/usr/bin/env python3
"""
STEP 17 - Space groups of the FULL GNoME screen (not just oxides).
================================================================================

    python scripts/cgcnn/17_spacegroups_general.py

WHAT THIS IS
-------------
16_spacegroups_oxides.py joined the GNoME summary CSV's own space-group
columns onto the oxide-filtered subset only. This script repeats the exact
same join on the FULL screen - all 33,323 scored candidates from
13_screen_gnome.py, no chemistry filter - so it shows what kappa_L <= 1 W/m/K
candidates look like crystallographically across the whole screen, not just
the oxide slice of it.

REUSE, NOT REIMPLEMENTATION
------------------------------
load_space_groups() and plot_histogram() are imported from
16_spacegroups_oxides.py (via importlib, same numeric-prefixed-filename
workaround 05_ensemble.py/13_screen_gnome.py already use) rather than
copy-pasted - same join, same source file, same plot style, just a
different input population.

Writes one CSV (every scored candidate + the three space-group columns +
an is_low_kappa_candidate flag - not present in gnome_screen_all.csv itself,
computed here the same way 13_screen_gnome.py's own threshold does, SORTED by
crystal system -> space group number -> Kappa_cal ascending, same as
16_spacegroups_oxides.py) and one histogram, identical style to
16_spacegroups_oxides.py's.

Also writes the same two "best candidate" CSVs (lowest predicted Kappa_cal
per crystal system, and per space group), scoped to the FULL screen this
time rather than just the oxide subset - see 16_spacegroups_oxides.py's
docstring for what "best" means here and why unpromising groups still get a
row.
"""

import os                          # path joining/checking
import sys                         # sys.path manipulation for the importlib workaround below
from importlib import import_module  # loads a module whose filename starts with a digit (not a valid identifier for a normal `import`)

import pandas as pd                # CSV I/O, DataFrame filtering/merging/sorting

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 13_screen_gnome.py's full, unfiltered scored-candidate table
SCREEN_CSV = os.path.join(RESULTS, "13_gnome_screen_all.csv")
# outputs: this script's own space-group join + derived tables, "17_" prefix
# so they're identifiable (OUT_CSV is read downstream by 20_high_symmetry.py,
# 22_best_dft_candidate.py and 29_alignn_agreement_dft_candidates.py)
OUT_CSV = os.path.join(RESULTS, "17_gnome_screen_spacegroups.csv")
OUT_PNG = os.path.join(RESULTS, "17_gnome_screen_spacegroups.png")
BEST_BY_CS_CSV = os.path.join(RESULTS, "17_gnome_screen_best_by_crystal_system.csv")
BEST_BY_SG_CSV = os.path.join(RESULTS, "17_gnome_screen_best_by_spacegroup.csv")
KAPPA_THRESHOLD = 1.0  # W/m/K cutoff used throughout this project to flag a "low-kappa candidate"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # put this file's own directory first on the import path
_sg = import_module("16_spacegroups_oxides")  # import by string name since "16_spacegroups_oxides" isn't a valid `import` identifier


def main():
    print("=== Space groups of the FULL GNoME screen ===\n")
    df = pd.read_csv(SCREEN_CSV)
    print(f"Loaded {len(df)} scored candidates from {SCREEN_CSV}")

    df["is_low_kappa_candidate"] = df["Kappa_cal (W m-1 K-1)"] <= KAPPA_THRESHOLD  # per-row boolean: does the point estimate clear the threshold
    print(f"  {int(df.is_low_kappa_candidate.sum())} clear the kappa_L <= "
         f"{KAPPA_THRESHOLD} W/m/K threshold")

    sg = _sg.load_space_groups()  # reuse 16's loader: reads GNoME's own summary CSV and returns the space-group columns
    merged = df.merge(sg, left_on="material_id", right_on="MaterialId", how="left") \
              .drop(columns=["MaterialId"])  # left join keeps every screened row; drop the now-duplicate join key column
    missing = merged["space_group"].isna().sum()  # rows where the join found no matching GNoME summary entry
    print(f"Joined against {_sg.GNOME_SUMMARY} on MaterialId")
    if missing:
        print(f"  {missing}/{len(merged)} rows have no matching GNoME summary row "
             f"(upstream data gap - see 16_spacegroups_oxides.py's module docstring) "
             f"- space_group columns left blank for those")

    print("\nBy crystal system (full screen):")
    for system, count in merged["crystal_system"].value_counts().items():  # iterate (crystal system, row count) pairs
        n_low = int(merged[(merged.crystal_system == system) &
                           merged.is_low_kappa_candidate].shape[0])  # count low-kappa rows within this crystal system
        print(f"  {system:14s} {count:6d} total, {n_low:6d} low-kappa candidates")

    sorted_df = _sg.sort_by_crystal_structure(merged)  # reuse 16's sort: crystal system -> space group number -> Kappa_cal ascending
    sorted_df.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV} ({len(sorted_df)} rows, sorted by crystal system "
         f"-> space group number -> Kappa_cal ascending)")

    _sg.plot_histogram(merged, OUT_PNG,
                       title="Space groups of the full GNoME screen",
                       all_label="All screened candidates",
                       low_label="Low-κ candidates")
    print(f"Wrote {OUT_PNG}")

    best_cs = _sg.best_by_group(merged, "crystal_system")  # reuse 16's group-best picker, grouped by crystal system
    best_cs.to_csv(BEST_BY_CS_CSV, index=False)
    print(f"\nWrote {BEST_BY_CS_CSV} ({len(best_cs)} rows - best candidate per crystal system, "
         f"ranked by lowest Kappa_cal_p95, not the point estimate - see best_by_group() docstring "
         f"in 16_spacegroups_oxides.py)")
    print("Best candidate per crystal system (lowest Kappa_cal_p95):")
    for _, row in best_cs.iterrows():  # one console line per crystal system's best pick
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05  # width of the model's own uncertainty interval, multiplicatively
        print(f"  {row.crystal_system:14s} {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} "
             f"p95={row.Kappa_cal_p95:.4f} (p95/p05={ratio:.1f}x) (sg {row.space_group}, {flag})")

    best_sg = _sg.best_by_group(merged, "space_group")  # same picker, grouped by space group instead
    best_sg.to_csv(BEST_BY_SG_CSV, index=False)
    print(f"\nWrote {BEST_BY_SG_CSV} ({len(best_sg)} rows - best candidate per space group, "
         f"ranked by lowest Kappa_cal_p95)")
    print("Top 10 overall (best candidate per space group, ranked by Kappa_cal_p95):")
    for _, row in best_sg.head(10).iterrows():  # ten best rows overall for a quick console preview
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05
        print(f"  {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} p95={row.Kappa_cal_p95:.4f} "
             f"(p95/p05={ratio:.1f}x) (sg {row.space_group}, {row.crystal_system}, {flag})")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
