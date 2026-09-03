#!/usr/bin/env python3
"""
STEP 24 - The oxide-containing candidates in the small-cell DFT ranking.
================================================================================

    python scripts/cgcnn/24_small_cell_oxides_for_dft.py

WHAT THIS IS
-------------
22_best_dft_candidate.py ranked all 267 confirmed low-kappa, small-cell
(<10 atom) candidates by Kappa_cal_p95 (the winner's-curse-safe ranking used
throughout this project - see that script's docstring). This one pulls out
just the oxide-containing rows from that same ranked list - oxides are
chemically the most familiar/easiest class to actually synthesize and
characterize of anything in the screen (halides dominate the low-kappa
population generally - see 18_element_frequency.py's enrichment finding -
so oxides are a deliberately narrower, more approachable slice for a first
real DFT/synthesis attempt).

WHY PYMATGEN, NOT A STRING SEARCH
-------------------------------------
Same reasoning as 15_filter_oxides.py's has_oxygen(): a substring search for
"O" would wrongly flag Os (osmium) and Og (oganesson). None of those happen
to appear in this particular 267-row slice, but the check is done the
correct way regardless rather than relying on it not mattering this time.

Writes one CSV: the oxide-containing subset of
gnome_small_cell_dft_ranking.csv, same columns, same Kappa_cal_p95-ascending
order, plus one new column - "confidence" (tight/moderate/wide, same <2x /
2-5x / >5x p95-p05-ratio cutoffs docs/method.tex already uses for the GNoME
overlap confidence breakdown). The printed recommendation excludes "wide"
candidates before picking a winner: Kappa_cal_p95 already discounts for
uncertainty once, but a >5x ratio is wide enough that even the p95 number
itself is not trustworthy at face value - so this is the same
winner's-curse-safe filtering applied a second time.
"""

import os                            # path joining/checking
import warnings                      # suppress a noisy, harmless pymatgen warning below

import pandas as pd                  # CSV I/O and DataFrame filtering
from pymatgen.core import Composition  # parses a chemical formula into its constituent elements

warnings.filterwarnings("ignore")    # silence pymatgen's formula-parsing warnings for the whole script

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 22_best_dft_candidate.py's Kappa_cal_p95-ranked small-cell shortlist
DFT_RANKING_CSV = os.path.join(RESULTS, "22_gnome_small_cell_dft_ranking.csv")
# output: this script's own oxide-only subset, "24_" prefix (read downstream
# by 26_recalibrated_dft_shortlist.py)
OUT_CSV = os.path.join(RESULTS, "24_gnome_small_cell_oxides_for_dft.csv")


def has_oxygen(formula):
    """True if O appears anywhere in the parsed formula - see module
    docstring for why this has to go through pymatgen, not str.find("O")."""
    try:
        return "O" in {str(el) for el in Composition(formula).elements}  # parse into element symbols, then a plain set-membership test
    except Exception:
        return False  # unparseable formula: treat as "no oxygen" rather than crashing the whole run


def main():
    print("=== Oxide-containing candidates in the small-cell DFT ranking ===\n")
    if not os.path.exists(DFT_RANKING_CSV):  # fail loudly with a fix hint rather than a raw pandas error
        raise SystemExit(f"{DFT_RANKING_CSV} not found - run scripts/cgcnn/22_best_dft_candidate.py first")

    df = pd.read_csv(DFT_RANKING_CSV)
    print(f"Loaded {len(df)} ranked small-cell, low-kappa candidates from {DFT_RANKING_CSV}")

    oxides = df[df["formula"].apply(has_oxygen)].copy()  # boolean-mask filter to oxide-containing rows, then copy to avoid a pandas view/copy warning
    print(f"\n{len(oxides)} of {len(df)} contain oxygen "
         f"({len(oxides) / len(df) * 100:.1f}%) - already sorted by Kappa_cal_p95 "
         f"ascending, order preserved from the source ranking")

    # Confidence bucket, same cutoffs docs/method.tex already uses for the
    # GNoME-overlap confidence breakdown (<2x tight, 2-5x moderate, >5x wide)
    # - kept explicit as a column rather than left implicit in p95_p05_ratio,
    # so "which of these do we actually trust" doesn't require re-deriving
    # the cutoff each time.
    def confidence_bucket(ratio):
        if ratio < 2:
            return "tight"
        if ratio <= 5:
            return "moderate"
        return "wide"

    oxides["confidence"] = oxides["p95_p05_ratio"].apply(confidence_bucket)  # one bucket label per row
    print("\nConfidence breakdown:")
    print(oxides["confidence"].value_counts().to_string())  # counts per bucket, formatted as plain text

    oxides.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV} ({len(oxides)} rows)")

    # The #1 row by Kappa_cal_p95 alone can still be a "wide" candidate if
    # nothing tighter happens to also have a low p95 - so the recommendation
    # explicitly excludes "wide" (>5x) candidates first, then takes the best
    # p95 among what remains. This is the same winner's-curse-safe logic as
    # best_by_group(), applied a second time on top of it: p95 already
    # discounts for uncertainty once, but a >5x ratio is wide enough that
    # even the p95 number itself shouldn't be taken at face value.
    trustworthy = oxides[oxides["confidence"] != "wide"]  # excludes "wide" rows from the recommendation pool
    excluded = oxides[oxides["confidence"] == "wide"]     # the excluded rows, kept only to report what was dropped
    if len(excluded):
        print(f"\nExcluding {len(excluded)} 'wide' (>5x p95/p05) candidates from the "
             f"recommendation - untrustworthy even after the p95 discount: "
             f"{', '.join(excluded.formula)}")

    top = trustworthy.iloc[0]  # first row of the trustworthy subset = best oxide pick
    print(f"\n{'=' * 60}")
    print(f"BEST OXIDE FOR DFT: {top.formula}  (material_id {top.material_id})")
    print(f"{'=' * 60}")
    print(f"  Atoms in primitive cell : {int(top['Number of Atoms'])}")
    print(f"  Crystal system / SG     : {top.crystal_system} / {top.space_group}")
    print(f"  Predicted Kappa_cal     : {top['Kappa_cal (W m-1 K-1)']:.4f} W/m/K")
    print(f"  p05 - p95 interval      : {top.Kappa_cal_p05:.4f} - {top.Kappa_cal_p95:.4f} "
         f"({top.p95_p05_ratio:.2f}x ratio, {top.confidence})")
    print(f"  K_VRH / G_VRH predicted : {top.K_VRH_pred:.1f} / {top.G_VRH_pred:.1f} GPa")

    print(f"\nNext 4 (trustworthy only):")
    for _, row in trustworthy.iloc[1:5].iterrows():  # rows 1-4 (0-indexed): the four next-best trustworthy picks
        print(f"  {row.formula:14s} p95={row.Kappa_cal_p95:.4f} ({row.p95_p05_ratio:.2f}x, "
             f"{row.confidence})  {row.crystal_system}, {row.space_group}, "
             f"{int(row['Number of Atoms'])} atoms")

    tightest = oxides.sort_values("p95_p05_ratio").iloc[0]  # separately, the single tightest uncertainty interval regardless of confidence bucket
    if tightest.formula != top.formula:  # only worth a note if it differs from the p95-based pick above
        print(f"\nMost statistically confident of all 24 (tightest ratio, not lowest p95): "
             f"{tightest.formula}, ratio={tightest.p95_p05_ratio:.2f}x, but Kappa_cal="
             f"{tightest['Kappa_cal (W m-1 K-1)']:.3f} is close to the threshold - "
             f"a less dramatic value than {top.formula}'s.")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
