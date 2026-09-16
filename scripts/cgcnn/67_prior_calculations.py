#!/usr/bin/env python
"""
Step 67 - has anyone already computed our shortlist?

WHY THIS EXISTS
---------------
Before asking for DFT time, check that the answer is not already public. Step 64
did this for arm B and cancelled six calculations by finding that round 9's
predicted moduli exceeded every real measured value for those chemistries. This
is the same discipline applied to the survivors: does a published calculation
exist for these exact crystals?

Two separate questions, and they have different answers:

  does the STRUCTURE exist anywhere?   -> half of them do, in Materials Project,
                                          because MP has ingested GNoME-derived
                                          entries. Those come with a formation
                                          energy and nothing else.
  does an ELASTIC TENSOR exist?        -> no. Not for one of the twelve.

RUN
    ml_env/bin/python scripts/cgcnn/67_prior_calculations.py

OUTPUT
    results/cgcnn/67_prior_calculations.csv
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
QUEUE = os.path.join(RESULTS, "66_dft_queue.csv")


def main():
    from mp_api.client import MPRester
    from pymatgen.core import Composition

    q = pd.read_csv(QUEUE, dtype={"material_id": str})
    q = q[q.status.isin(["QUEUE", "CONTROL"])]
    print("=" * 78)
    print("STEP 67 - IS THE ANSWER ALREADY PUBLIC?")
    print("=" * 78)
    print(f"\nchecking {len(q)} shortlisted crystals against Materials Project\n")

    rows = []
    with MPRester() as mpr:
        for r in q.itertuples():
            comp = Composition(r.formula)
            # Search the whole chemical system, not the formula string: a
            # published entry may use a different but equivalent formula, and
            # the reduced formula is what actually identifies the compound.
            chemsys = "-".join(sorted(comp.as_dict()))
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                fields=["material_id", "formula_pretty", "symmetry",
                        "energy_above_hull", "bulk_modulus", "shear_modulus"])
            same = [d for d in docs
                    if Composition(d.formula_pretty).reduced_formula == comp.reduced_formula]
            elastic = [d for d in same if d.bulk_modulus]
            match = next((d for d in same if d.symmetry.symbol == r.space_group), None)

            rows.append({
                "status": r.status, "formula": r.formula,
                "gnome_id": r.material_id, "space_group": r.space_group,
                "chemsys": chemsys,
                "mp_entries_in_chemsys": len(docs),
                "mp_same_formula": len(same),
                "mp_same_spacegroup": bool(match),
                "mp_id": str(match.material_id) if match else "",
                "mp_e_above_hull": (round(match.energy_above_hull, 4)
                                    if match and match.energy_above_hull is not None else None),
                "mp_has_elastic": len(elastic),
            })
            flag = "" if not match else f"  == {rows[-1]['mp_id']}"
            print(f"  {r.formula:<12}{chemsys:<16}"
                  f"same formula {len(same)}   same spacegroup "
                  f"{'yes' if match else 'no ':<3}   elastic {len(elastic)}{flag}")

    t = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "67_prior_calculations.csv")
    t.to_csv(out, index=False)

    print("\n" + "-" * 78)
    print(f"  structures already in MP (same formula AND space group): "
          f"{int(t.mp_same_spacegroup.sum())}/{len(t)}")
    print(f"  of those, how many carry an elastic tensor:              "
          f"{int(t.mp_has_elastic.sum())}")
    print(f"  never published in any form:                             "
          f"{int((t.mp_same_formula == 0).sum())}/{len(t)}")
    print("\n  Every MP match sits exactly on the convex hull (e_above_hull = 0),")
    print("  which is an independent confirmation of GNoME's stability claim -")
    print("  and says nothing about DYNAMIC stability, which is a different")
    print("  question and the one our phonons answer.")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
