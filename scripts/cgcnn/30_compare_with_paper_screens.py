#!/usr/bin/env python3
"""
STEP 30 - Compare our DFT candidate list against the paper's own screens.
================================================================================

    python scripts/cgcnn/30_compare_with_paper_screens.py

WHAT THIS IS
-------------
14_compare_gnome_screen.py already checked our FULL 16,299-candidate GNoME
screen against the paper's own published JMI candidate list (70.8%
overlap). This script asks a narrower, more specific question: of the 218
candidates in gnome_dft_candidates_agreement_lt1.csv - this project's
strongest, most cross-validated pick (CGCNN AND ALIGNN independently agree,
kappa_L < 1, < 10 atoms, cheap to actually DFT-verify) - how many did the
paper's own screens already flag too?

THREE COMPARISON TARGETS, ALL FROM external/AI4Kappa/ (the paper's own
released supporting information, not re-derived)
------------------------------------------------------------------------------
  JMI_Supporting_Information/Nature-filtered-low-kappa.csv      (11,869 rows)
      The paper's headline GNoME screen result - same one 14_compare_gnome_screen.py
      already checked the full screen against.
  KappaP_Supporting_Information/Nature-filtered-low-kappa.csv   (10,476 rows)
      A DIFFERENT GNoME candidate list from a different journal's supporting
      info for the same paper - never previously compared against in this
      project.
  KappaP_Supporting_Information/MP-semiconductor-low-Kappa.csv  (18,710 rows)
      A genuinely different population: REAL Materials Project entries
      (mp-xxxx ids), not hypothetical GNoME structures - never previously
      compared against either.

WHY OVERLAP IS CHECKED BY FORMULA, NOT ID
-----------------------------------------------
Same reasoning as 14_compare_gnome_screen.py: none of the paper's published
CSVs carry a GNoME material ID, only a formula. JMI's and KappaP's own
"Reduced Formula" columns already match pymatgen's reduced-formula
convention (spot-checked directly). MP-semiconductor-low-Kappa.csv's
"formula" column does NOT - it carries un-reduced, full-cell compositions
(e.g. "Mn6Ag4P8O28", "Ni1Sn1Cl6O6" with an explicit "1") - checked directly,
not assumed, after an initial run against it came back with a suspicious
flat 0/218 overlap. Every formula on both sides is therefore normalised
through pymatgen's Composition(...).reduced_formula before comparing, not
just for MP-semiconductor - applying the same normalisation everywhere is
safer than trusting two of three files' formatting by inspection alone and
only fixing the one that happened to look wrong.

Writes one CSV per comparison target (the 218 rows, tagged with whether
each paper's screen also flagged that formula) and prints the overlap
counts plus the specific overlapping materials.
"""

import os                            # path joining/checking
import warnings                      # suppress a noisy, harmless pymatgen warning below

import pandas as pd                  # CSV I/O and DataFrame filtering
from pymatgen.core import Composition  # parses a chemical formula string and reduces it to lowest integer ratios

warnings.filterwarnings("ignore")    # silence pymatgen's formula-parsing warnings for the whole script

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 29_alignn_agreement_dft_candidates.py's strict (<1 W/m/K) agreement shortlist
OURS_CSV = os.path.join(RESULTS, "29_gnome_dft_candidates_agreement_lt1.csv")
AI4KAPPA = os.path.join(PROJECT_ROOT, "external", "AI4Kappa")  # the paper's own released supporting-information files

TARGETS = {
    "JMI (paper's headline GNoME screen)":
        (os.path.join(AI4KAPPA, "JMI_Supporting_Information", "Nature-filtered-low-kappa.csv"),
        "Reduced Formula"),
    "KappaP (a different GNoME screen)":
        (os.path.join(AI4KAPPA, "KappaP_Supporting_Information", "Nature-filtered-low-kappa.csv"),
        "Reduced Formula"),
    # This file's own header uses lowercase "formula", not "Reduced Formula"
    # like the other two - checked directly (head -1), not assumed to match.
    "MP-semiconductor (real Materials Project entries)":
        (os.path.join(AI4KAPPA, "KappaP_Supporting_Information", "MP-semiconductor-low-Kappa.csv"),
        "formula"),
}


def reduced(formula):
    """pymatgen's own reduced formula - the normalisation applied to every
    formula on both sides before comparing, see module docstring for why
    this can't be skipped even for columns that already look reduced."""
    try:
        return Composition(formula).reduced_formula  # parse the string and reduce to lowest integer ratios
    except Exception:
        return formula  # unparseable formula: fall back to the raw string rather than crashing the whole run


def main():
    print("=== Comparing our DFT candidates against the paper's own screens ===\n")
    ours = pd.read_csv(OURS_CSV)
    print(f"Loaded {len(ours)} candidates from {OURS_CSV}")
    ours["formula_reduced"] = ours["formula"].apply(reduced)  # normalise our own formulas the same way as every comparison target
    our_formulas = set(ours["formula_reduced"])  # set for O(1) membership tests below

    for label, (path, formula_col) in TARGETS.items():  # one pass per comparison target
        if not os.path.exists(path):
            print(f"\n{label}: MISSING at {path}, skipped")
            continue
        paper = pd.read_csv(path, usecols=[formula_col])  # only the one formula column is needed from the paper's file
        paper_formulas = set(paper[formula_col].apply(reduced))  # normalised set of the paper's own formulas

        overlap = our_formulas & paper_formulas  # set intersection: formulas appearing in both lists
        out = ours.copy()  # explicit copy so this loop's edits don't leak into the next target's comparison
        out["also_in_this_screen"] = out["formula_reduced"].isin(paper_formulas)  # per-row flag for this target

        safe_name = label.split(" (")[0].lower().replace("-", "_")  # e.g. "JMI (paper's...)" -> "jmi", filesystem/column-safe
        # "30_" prefix identifies this script as the source; nothing else in
        # the pipeline reads this dynamically-named per-target file (checked
        # by grepping every other script for this literal pattern)
        out_path = os.path.join(RESULTS, f"30_gnome_dft_candidates_vs_{safe_name}.csv")
        out.to_csv(out_path, index=False)  # index=False: don't write the pandas row-number column

        print(f"\n{'=' * 70}")
        print(f"{label}  ({len(paper)} rows)")
        print(f"{'=' * 70}")
        print(f"  Overlap: {len(overlap)}/{len(ours)} of our candidates also appear here")
        if overlap:
            print(f"  Materials: {sorted(overlap)}")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
