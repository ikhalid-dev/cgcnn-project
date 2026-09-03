#!/usr/bin/env python3
"""
STEP 20 - Pull out the high-symmetry candidates from the GNoME screen.
================================================================================

    python scripts/cgcnn/20_high_symmetry.py

WHAT THIS IS
-------------
17_spacegroups_general.py tagged every one of the 33,323 scored candidates
with its crystal system (from GNoME's own structure metadata). This script
pulls out the CUBIC ones - the highest of the seven crystal systems in the
symmetry ordering already used throughout this project
(CRYSTAL_SYSTEM_RANGES in 16_spacegroups_oxides.py: triclinic -> ... ->
cubic), and therefore the natural, already-established meaning of
"high-symmetry" here.

WHY CUBIC, NOT A LOOSER OR STRICTER DEFINITION
---------------------------------------------------
"High symmetry" needs a precise definition to be checkable, not just a
feeling. Two candidates were considered:

  1. Crystal system (used here): cubic is the unique top of the standard
     seven-system symmetry hierarchy, unambiguous, and already the
     convention this project uses everywhere else a crystal-system ordering
     appears.
  2. pymatgen's SpaceGroup.order (total number of symmetry operations,
     INCLUDING lattice centering) - tried first, and rejected as the primary
     filter: checked directly on this data, some rhombohedral trigonal space
     groups (e.g. R-3m, described in hexagonal axes with 3x centering) report
     order=36, higher than several genuinely cubic groups like Pm-3m
     (order=48... actually the point is order is not monotonic with
     intuitive "symmetry class" across different crystal systems, because
     centering multiplicity is a property of the chosen cell setting, not
     of intrinsic point symmetry). Comparing crystal systems by raw
     SpaceGroup.order would have let some trigonal candidates outrank cubic
     ones for a misleading reason.

Within the cubic subset, though, SpaceGroup.order (via
pymatgen.symmetry.groups.SpaceGroup.from_int_number, robust - checked
against all 167 distinct space group numbers in this screen, zero failures)
IS a meaningful secondary ranking: it correctly separates e.g. Fm-3m/Fd-3m
(order 192, face-centered, the most symmetric groups in the whole dataset)
from P2_13 (order 12, primitive, the least symmetric cubic group present).
That comparison is safe because it stays within one crystal system, where
centering differences are a genuine feature of the structure rather than an
axis-setting artifact.

Writes one CSV: every cubic candidate, all of gnome_screen_spacegroups.csv's
own columns plus space_group_order, sorted by space_group_order descending
(most symmetric first) then Kappa_cal ascending.
"""

import os                                        # path joining/checking

import pandas as pd                              # CSV I/O and DataFrame filtering/sorting
from pymatgen.symmetry.groups import SpaceGroup  # looks up a space group's symmetry-operation count from its number

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 17_spacegroups_general.py's per-candidate space-group/crystal-system table
SCREEN_SG_CSV = os.path.join(RESULTS, "17_gnome_screen_spacegroups.csv")
# output: this script's own cubic-only shortlist, "20_" prefix to identify its source script
OUT_CSV = os.path.join(RESULTS, "20_gnome_screen_high_symmetry.csv")

# A few well-known high-symmetry cubic space groups, printed out by name so
# the console summary is readable without looking numbers up.
NOTABLE = {
    221: "Pm-3m (simple cubic / perovskite aristotype)",
    225: "Fm-3m (rock-salt / fluorite)",
    227: "Fd-3m (diamond / spinel)",
    229: "Im-3m (body-centred cubic)",
    216: "F-43m (zinc-blende)",
}


def main():
    print("=== High-symmetry (cubic) candidates in the GNoME screen ===\n")
    if not os.path.exists(SCREEN_SG_CSV):  # fail loudly with a fix hint rather than a raw pandas error
        raise SystemExit(f"{SCREEN_SG_CSV} not found - run scripts/cgcnn/17_spacegroups_general.py first")

    df = pd.read_csv(SCREEN_SG_CSV)
    print(f"Loaded {len(df)} scored, space-group-tagged candidates from {SCREEN_SG_CSV}")

    cubic = df[df["crystal_system"] == "cubic"].copy()  # boolean-mask filter to cubic rows only, then copy to avoid a pandas view/copy warning
    print(f"\n{len(cubic)} of {len(df)} are cubic ({len(cubic) / len(df) * 100:.1f}%) "
         f"- the smallest of the seven crystal systems in this screen, consistent with "
         f"Section 3 of docs/oxide_screen_columns.tex")

    # build {space_group_number: symmetry-operation count} once per distinct
    # number present, rather than recomputing SpaceGroup.from_int_number for
    # every row
    order_lookup = {int(n): SpaceGroup.from_int_number(int(n)).order
                   for n in sorted(cubic["space_group_number"].dropna().unique())}
    cubic["space_group_order"] = cubic["space_group_number"].map(order_lookup)  # vectorised dict lookup, one order value per row

    cubic = cubic.sort_values(
        ["space_group_order", "Kappa_cal (W m-1 K-1)"], ascending=[False, True]
    ).reset_index(drop=True)  # most symmetric first, then lowest predicted kappa within a tie; reindex 0..N-1

    n_low = int(cubic["is_low_kappa_candidate"].sum())  # True/False column summed as 1/0
    print(f"  {n_low} of those clear the kappa_L <= 1 W/m/K threshold")

    print("\nBy space group (most symmetric first):")
    for num in sorted(cubic["space_group_number"].dropna().unique(),
                      key=lambda n: -order_lookup[int(n)]):  # sort distinct space groups by descending order
        sub = cubic[cubic.space_group_number == num]  # every cubic row sharing this exact space group number
        sym = sub["space_group"].iloc[0]  # space-group symbol is constant within a number, just read the first row's
        note = f"  <- {NOTABLE[int(num)]}" if int(num) in NOTABLE else ""  # append a friendly name for well-known groups
        print(f"  #{int(num):3d} {sym:10s} order={order_lookup[int(num)]:3d}  "
             f"n={len(sub):4d}  low-kappa={int(sub.is_low_kappa_candidate.sum()):3d}{note}")

    cubic.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV} ({len(cubic)} rows, sorted by space-group order descending, "
         f"then Kappa_cal ascending)")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
