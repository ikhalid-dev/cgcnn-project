#!/usr/bin/env python
"""
Step 75 - a DFT queue from the surviving oxides.
================================================================================

    SUPERSEDED BY STEP 81 (2026-10-05). This queue is oxides only, because when
    it was written the halides had no real gamma. Step 81 merges both lists and
    adds the amorphous-limit check. Request from 81_combined_dft_queue.csv, not
    from this file. Kept because 81 reads its negative controls from here.

    Also rebuilt on 2026-10-05: before then its input (step 74) still carried the
    gamma from the buggy 0.001 THz frequency cutoff - 93 survivors, now 92.

    ml_env/bin/python scripts/cgcnn/75_oxide_dft_queue.py
    ml_env/bin/python scripts/cgcnn/75_oxide_dft_queue.py --skip-mp

WHAT THESE 93 ACTUALLY HAVE, AND WHAT THEY DO NOT
--------------------------------------------------
Stated precisely, because it is easy to overclaim and this queue is a request
for somebody's compute time:

    K and G            PREDICTED   ALIGNN, CGCNN-ens, new matminer baseline
    gamma              COMPUTED    real phonons at three volumes, mode
                                   Grueneisen averaged over the mesh
    dynamic stability  COMPUTED    same phonon calculation, 0 imaginary modes
    kappa              DERIVED     predicted K,G + computed gamma

The moduli are still machine-learning predictions - that never changed, and
they dominate kappa. What changed is that gamma stopped being the Poisson
formula 3(1+nu)/(2(2-3nu)) and became a phonon calculation, and every crystal
here survived a stability test that killed 48% of its cohort.

And "computed" is not "measured": gamma comes from MACE-OMAT-0, a machine-
learned interatomic potential. ML forces driving a real phonon calculation, not
DFT, not experiment. That is exactly why DFT is still the thing being asked for.

WHY THIS QUEUE IS CHEAPER THAN IT LOOKS
----------------------------------------
No GPU time is outstanding. The ~28 GPU-hours that produced these gammas are
already spent; steps 74 and 75 are arithmetic on columns that exist. Compare
the halide top-300, which needs 28.6 GPU-hours before it can even be triaged.

TIERS, BECAUSE 93 DFT CALCULATIONS IS NOT A REQUEST ANYONE GRANTS
------------------------------------------------------------------
    A1  the 20 lowest kappa_max3 - the actual ask
    A2  the next 30 - queued behind A1 if A1 looks good
    A3  the remaining 43 - documented, not requested
    C   the four existing negative controls, unchanged. Their job is to catch a
        pipeline that calls everything soft, and step 69 showed the real gamma
        moves them DOWN while moving candidates up - opposite directions, which
        is what makes the correction credible.

Ranking is kappa_max3 = max(ALIGNN, CGCNN-ens, newbase): three matbench-trained
models. round 9 and the composition tree gate nothing here or anywhere else any
more - see step 73.

OUTPUTS
    results/cgcnn/75_oxide_dft_queue.csv
    results/cgcnn/75_oxide_prior_calculations.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
MB = ["ALIGNN", "CGCNN-ens", "newbase"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mp", action="store_true")
    args = ap.parse_args()

    ox = pd.read_csv(os.path.join(RESULTS, "74_oxide_shortlist_real_gamma.csv"),
                     dtype={"material_id": str})
    ox = ox[ox.survives].sort_values("kappa_max3").reset_index(drop=True)
    ox["rank"] = np.arange(1, len(ox) + 1).astype(object)  # object: keeps ints unformatted beside floats
    # np.select picks the first condition that is True, row by row.
    ox["tier"] = np.select([ox["rank"] <= 20, ox["rank"] <= 50],
                           ["A1: request", "A2: queued"], "A3: documented")
    ox["status"] = "QUEUE"
    ox["origin"] = "oxide shortlist, real gamma"

    ctl = pd.read_csv(os.path.join(RESULTS, "73_matbench3_dft_queue.csv"),
                      dtype={"material_id": str})
    ctl = ctl[ctl.status == "CONTROL"].copy()
    ctl["tier"] = "C: negative control"
    ctl["origin"] = "step 61 control"
    ctl["rank"] = np.nan

    keep = ["status", "tier", "rank", "origin", "formula", "material_id",
            "Number of Atoms", "gamma_mlip"] + MB + ["kappa_max3", "binding3"]
    q = pd.concat([ox.reindex(columns=keep), ctl.reindex(columns=keep)],
                  ignore_index=True)
    q.to_csv(os.path.join(RESULTS, "75_oxide_dft_queue.csv"), index=False)

    print("=" * 98)
    print("STEP 75 - DFT QUEUE FROM THE SURVIVING OXIDES")
    print("=" * 98)
    for t in ["A1: request", "A2: queued", "C: negative control"]:
        sub = q[q.tier == t]
        if sub.empty:
            continue
        print(f"\n{t}  ({len(sub)})")
        print(sub[["rank", "formula", "material_id", "Number of Atoms",
                   "gamma_mlip"] + MB + ["kappa_max3", "binding3"]]
              .to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))
    a3 = q[q.tier == "A3: documented"]
    print(f"\nA3: documented  ({len(a3)})  kappa_max3 "
          f"{a3.kappa_max3.min():.3f} - {a3.kappa_max3.max():.3f}  (in the CSV, not requested)")
    print(f"\nrequest = {len(q[q.tier=='A1: request'])} candidates + "
          f"{len(ctl)} controls = {len(q[q.tier=='A1: request']) + len(ctl)} calculations")
    print(f"outstanding GPU time: none - these gammas are already computed")

    if args.skip_mp:
        return

    # ---- has anyone already done this? ------------------------------------
    from mp_api.client import MPRester
    from pymatgen.core import Composition
    # one tqdm bar per query x 54 queries buries the answer
    import warnings; warnings.filterwarnings('ignore')
    live = q[q.tier.isin(["A1: request", "A2: queued", "C: negative control"])]
    rows = []
    print("\n" + "=" * 98)
    print("PRIOR CALCULATIONS - Materials Project")
    print("=" * 98)
    with MPRester() as mpr:
        for r in live.itertuples():
            comp = Composition(r.formula)
            chemsys = "-".join(sorted(e.symbol for e in comp.elements))
            docs = mpr.materials.summary.search(
                chemsys=chemsys, fields=["material_id", "formula_pretty"],
                num_chunks=None, chunk_size=1000)
            same = [x for x in docs if Composition(x.formula_pretty).reduced_formula
                    == comp.reduced_formula]
            n_el = 0
            if same:
                n_el = len(mpr.materials.elasticity.search(
                    material_ids=[x.material_id for x in same],
                    fields=["material_id"]))
            rows.append({"tier": r.tier, "formula": r.formula, "gnome_id": r.material_id,
                         "chemsys": chemsys, "mp_entries_in_chemsys": len(docs),
                         "mp_same_formula": len(same),
                         "mp_id": same[0].material_id if same else "",
                         "mp_has_elastic": n_el})
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(RESULTS, "75_oxide_prior_calculations.csv"), index=False)
    print(f"\nqueried {len(pc)} crystals")
    print(f"   structure already in MP : {(pc.mp_same_formula > 0).sum()}")
    print(f"   chemical system empty   : {(pc.mp_entries_in_chemsys == 0).sum()}")
    print(f"   ELASTIC TENSORS in MP   : {int(pc.mp_has_elastic.sum())}"
          f"   <- decides whether this duplicates existing work")
    print(f"\nwrote {RESULTS}/75_oxide_prior_calculations.csv")


if __name__ == "__main__":
    main()
