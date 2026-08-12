#!/usr/bin/env python3
"""
STEP 15 - Pull out the oxide candidates from the GNoME screen.
================================================================================

    python scripts/cgcnn/15_filter_oxides.py

WHAT THIS IS
-------------
13_screen_gnome.py scored all 33,323 filtered GNoME candidates for K, G and
kappa_L. This script does not re-run any of that - it just reads the already-
scored CSV and pulls out the rows that are oxides (contain oxygen at all,
e.g. ABO3 perovskites, spinels, simple binary oxides, anything else with an
O in the formula), so a materials-scientist audience can look at a
manageable, chemically-meaningful slice instead of all 33,323 rows.

Writes two files: every scored oxide (gnome_oxide_candidates.csv), and a
second, smaller one containing only the ones that also cleared the screen's
own kappa_L <= 1 threshold (gnome_oxide_low_kappa_candidates.csv) - the
actual low-kappa oxide finds, not just "oxides that got scored at all."

WHY "CONTAINS O" IS CHECKED WITH PYMATGEN, NOT A STRING SEARCH
-----------------------------------------------------------------
The formula column is a reduced formula like "Ho7Er(OsBr4)2" - a naive
substring search for "O" would wrongly flag Os (osmium), Og (oganesson) and
anything else whose symbol merely contains the letter O. pymatgen's
Composition parser (used everywhere else in this project) expands nested
parentheses correctly and gives back real element symbols to check against,
so "OsBr2" is correctly excluded while "Fe3O4" is correctly included.

WHAT'S KEPT FROM THE SCREEN, AND WHAT'S ADDED
-----------------------------------------------
Every predicted-property column from 13_screen_gnome.py's output rides along
unchanged (K_VRH_pred, G_VRH_pred, both ensemble spreads, Poisson ratio,
Gruneisen parameter, Kappa_cal and its Monte Carlo p05/p50/p95 interval) -
this script only filters rows and adds two new columns:

    is_low_kappa_candidate   True if this oxide also cleared the screen's own
                              kappa_L <= 1 W/m/K threshold (i.e. it is one of
                              the 16,299 in gnome_screen_candidates.csv, not
                              just one of the 33,323 that got scored at all).
                              Kept explicit rather than silently restricting
                              to only the sub-threshold rows, so this file
                              answers "which oxides did we screen" AND "which
                              of those are the actual low-kappa finds" at
                              once.
    stoichiometry_pattern     A light structural tag from the REDUCED cation
                              ratios (not a DFT-verified structure type - see
                              classify_oxide docstring) - ABO3-type
                              (perovskite-like), AB2O4-type (spinel-like),
                              binary oxide, or a generic ternary/complex-oxide
                              label otherwise.
"""

import argparse
import os
import sys
import warnings

import pandas as pd
from pymatgen.core import Composition

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")


def has_oxygen(formula):
    """True if O appears anywhere in the parsed formula - see module
    docstring for why this has to go through pymatgen, not str.find("O")."""
    try:
        return "O" in {str(el) for el in Composition(formula).elements}
    except Exception:
        return False


def classify_oxide(formula):
    """A rough stoichiometry tag from the REDUCED cation:oxygen ratio.

    This is pattern-matching on formula ratios only, not a structure
    prediction - two materials with an ABO3-style 1:1:3 ratio are not
    guaranteed to share the perovskite structure (that needs the actual
    atomic positions, which this doesn't look at), but the ratio is exactly
    the giveaway the user asked for ("eg ABO3, other likes this"), so it's a
    useful first-pass label to sort/skim by.
    """
    amounts = Composition(formula).reduced_composition.get_el_amt_dict()
    o_amt = amounts.pop("O", 0)
    cations = sorted(amounts.values())

    if len(cations) == 1:
        return "binary oxide (A-O)"
    if len(cations) == 2 and cations == [1, 1] and o_amt == 3:
        return "ABO3-type (perovskite-like)"
    if len(cations) == 2 and cations == [1, 2] and o_amt == 4:
        return "AB2O4-type (spinel-like)"
    if len(cations) == 2 and cations == [2, 2] and o_amt == 7:
        return "A2B2O7-type (pyrochlore-like)"
    if len(cations) == 2:
        return "ternary oxide (A-B-O, other ratio)"
    return f"complex oxide ({len(cations) + 1} elements)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-csv", default=os.path.join(RESULTS, "gnome_screen_all.csv"),
                        help="the full scored screen (all 33,323, pre-threshold) - "
                             "default, so oxides that scored above the kappa "
                             "threshold are still visible with is_low_kappa_candidate=False")
    parser.add_argument("--kappa-threshold", type=float, default=1.0,
                        help="must match 13_screen_gnome.py's own threshold to "
                             "correctly flag is_low_kappa_candidate")
    parser.add_argument("--out", default=os.path.join(RESULTS, "gnome_oxide_candidates.csv"))
    parser.add_argument("--out-low-kappa",
                        default=os.path.join(RESULTS, "gnome_oxide_low_kappa_candidates.csv"),
                        help="second file, the is_low_kappa_candidate subset only - "
                             "the actual low-kappa oxide finds, not every oxide scored")
    args = parser.parse_args()

    print("=== Filtering the GNoME screen for oxide candidates ===\n")
    if not os.path.exists(args.screen_csv):
        sys.exit(f"{args.screen_csv} not found - run scripts/cgcnn/13_screen_gnome.py first")

    df = pd.read_csv(args.screen_csv)
    print(f"Loaded {len(df)} scored candidates from {args.screen_csv}")

    is_oxide = df["formula"].apply(has_oxygen)
    oxides = df[is_oxide].copy()
    print(f"{len(oxides)} of {len(df)} contain oxygen ({len(oxides) / len(df) * 100:.1f}%)")

    oxides["stoichiometry_pattern"] = oxides["formula"].apply(classify_oxide)
    oxides["is_low_kappa_candidate"] = oxides["Kappa_cal (W m-1 K-1)"] <= args.kappa_threshold
    oxides = oxides.sort_values("Kappa_cal (W m-1 K-1)").reset_index(drop=True)

    n_candidates = int(oxides["is_low_kappa_candidate"].sum())
    print(f"  {n_candidates} of those also clear the kappa_L <= {args.kappa_threshold} "
         f"W/m/K threshold (real low-kappa oxide candidates, not just scored oxides)")

    print("\nBy stoichiometry pattern:")
    for pattern, count in oxides["stoichiometry_pattern"].value_counts().items():
        n_cand = int(oxides[oxides.stoichiometry_pattern == pattern]["is_low_kappa_candidate"].sum())
        print(f"  {pattern:38s} {count:6d} total, {n_cand:6d} low-kappa candidates")

    oxides.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(oxides)} rows, sorted by predicted kappa_L ascending)")

    low_kappa = oxides[oxides["is_low_kappa_candidate"]].drop(columns=["is_low_kappa_candidate"])
    low_kappa.to_csv(args.out_low_kappa, index=False)
    print(f"Wrote {args.out_low_kappa} ({len(low_kappa)} rows - just the oxides that "
         f"cleared kappa_L <= {args.kappa_threshold})")


if __name__ == "__main__":
    main()
