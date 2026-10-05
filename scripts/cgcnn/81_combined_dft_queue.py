#!/usr/bin/env python
"""
Step 81 - ONE DFT queue from both shortlists, oxides and halides together.
================================================================================

    ml_env/bin/python scripts/cgcnn/81_combined_dft_queue.py
    ml_env/bin/python scripts/cgcnn/81_combined_dft_queue.py --skip-mp

WHY THIS REPLACES STEP 75
-------------------------
Step 75 built the DFT request from the oxides alone - not by choice, but
because the halides had no real gamma yet. Step 79 gave them one, and they are
the stronger candidates (median kappa_max3 0.30 against the oxides' 0.75). A
request built from one list when two exist asks for the best 20 OXIDES, not the
best 20 CANDIDATES.

Both inputs are on the FIXED 0.3 THz frequency cutoff (step 74 was rebuilt on
2026-10-05 to read the recut; before that it still carried the buggy gamma).

THREE CRYSTALS ARE IN BOTH LISTS
---------------------------------
Cs3RuO4, CsIrOF4, Cs3NiO3 contain oxygen AND sit in the halide top-300, so each
was computed twice, in different shards. They are listed once. Their two gammas
are compared below as a free check on the pipeline.

A COLUMN THE OLD QUEUE DID NOT HAVE: model spread
--------------------------------------------------
kappa_max3 is the most pessimistic of three models - a crystal survives only if
all three agree it is below 1.0. But "all three below 1.0" can mean they agree
closely or that they span a factor of five. `spread3` = kappa_max3 / kappa_min3
says which. A large spread means the moduli are uncertain, so that crystal's
DFT result is the more informative one - but also the likelier to disappoint.

A SECOND NEW COLUMN: the amorphous limit
-----------------------------------------
The Slack formula has no floor - push gamma up or the sound speed down and its
kappa goes toward zero. Real solids do have a floor: the Cahill-Watson-Pohl
minimum, the conductivity of the same atoms with no crystalline order at all.

    kappa_min = 1/2 * (pi/6)^(1/3) * kB * n^(2/3) * (v_l + 2 v_t)      (high T)

n is atoms per volume; v_l, v_t are the sound speeds. Validated before use: on
PINK Table 1 not one of 44 measured crystals sits below its computed minimum
(the closest, AgCl, is 2.6x above it).

`kappa_cahill` is that floor, from CGCNN-ens's K and G (available for every
GNoME crystal; it depends on the moduli only through sqrt, so a 2x model
disagreement in G moves it ~1.4x). `below_floor` marks a Slack prediction under
it. Such a prediction is not "even better" - it is outside where the Slack
formula means anything, and the crystal's real kappa is more likely near the
floor. Ranking by lowest kappa_max3 PREFERENTIALLY selects these: on
2026-10-05, 18 of the 20 in A1 were below the floor.

The ranking is NOT changed here - that is a policy choice for the user - but
`kappa_floored` = max(kappa_max3, kappa_cahill) and `rank_floored` are written
alongside, so the alternative ranking is one sort away.

WHAT IS STILL NOT HERE
----------------------
Halide ranks 1-60 were mostly skipped (step 76): the pilot found 1 of 10 usable
there. Nine of those 60 were computed in the pilot; the other ~50 never were.
So "the best 20 candidates" means the best 20 among what has been computed.

TIERS (same as step 75)
    A1  the 20 lowest kappa_max3 - the actual request
    A2  the next 30 - queued behind A1
    A3  the rest - documented, not requested
    C   the four negative controls from step 61, unchanged

OUTPUTS
    results/cgcnn/81_combined_dft_queue.csv
    results/cgcnn/81_prior_calculations.csv     (unless --skip-mp)
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
MB = ["ALIGNN", "CGCNN-ens", "newbase"]
KB = 1.380649e-23   # Boltzmann constant, J/K


def cahill_min(K, G, rho, V, n):
    """Cahill-Watson-Pohl minimum thermal conductivity, high-temperature form.

    K, G in GPa; rho in g/cm3; V in A^3 (primitive cell); n atoms in that cell.
    sqrt(GPa / (g/cm3)) is km/s, hence the *1000 to get m/s.
    The leading 1/2 is the high-T limit of the Cahill integral - dropping it
    doubles the floor (a mistake made, and caught on silicon, while writing this).
    """
    vl = np.sqrt((K + 4 * G / 3) / rho) * 1000
    vt = np.sqrt(G / rho) * 1000
    n_per_m3 = n / (V * 1e-30)
    return 0.5 * (np.pi / 6) ** (1 / 3) * KB * n_per_m3 ** (2 / 3) * (vl + 2 * vt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mp", action="store_true",
                    help="skip the Materials Project prior-calculation lookup")
    args = ap.parse_args()

    # ---- 1. the two shortlists, survivors only ----------------------------
    ox = pd.read_csv(os.path.join(RESULTS, "74_oxide_shortlist_real_gamma.csv"),
                     dtype={"material_id": str})
    ha = pd.read_csv(os.path.join(RESULTS, "79_halide_shortlist_real_gamma.csv"),
                     dtype={"material_id": str})

    # A guard, not a formality: if step 74 is ever reverted to the banked gamma,
    # a survivor would carry "banked 0.001 THz" and this stops the queue.
    bad = ox[ox.survives & (ox.gamma_source != "recut 0.3 THz")]
    if len(bad):
        raise SystemExit(f"{len(bad)} oxide survivors are on the OLD gamma - rerun step 74")

    ox = ox[ox.survives].assign(origin="oxide")
    # step 79 calls the gamma it used `gamma_used`; give both the same name
    ha = ha[ha.survives].rename(columns={"gamma_used": "gamma_mlip"}).assign(origin="halide")

    # ---- 2. the overlap: a free consistency check --------------------------
    both = ox.merge(ha, on="material_id", suffixes=("_ox", "_ha"))
    print("=" * 96)
    print("STEP 81 - COMBINED DFT QUEUE, OXIDES + HALIDES, FIXED CUTOFF")
    print("=" * 96)
    print(f"\noxide survivors {len(ox)}   halide survivors {len(ha)}   in both {len(both)}")
    if len(both):
        print("\nthe crystals computed twice, in separate shards:")
        print(both[["formula_ox", "material_id", "gamma_mlip_ox", "gamma_mlip_ha",
                    "kappa_max3_ox", "kappa_max3_ha"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 3. one table ------------------------------------------------------
    keep = ["formula", "material_id", "Number of Atoms", "gamma_mlip"] + MB + \
           ["kappa_max3", "kappa_min3", "binding3", "origin"]
    q = pd.concat([ox.reindex(columns=keep), ha.reindex(columns=keep)],
                  ignore_index=True)
    # Mark the overlap before dropping its second copy, so it is visible.
    q.loc[q.material_id.isin(both.material_id), "origin"] = "oxide+halide"
    q = q.drop_duplicates("material_id")

    q["spread3"] = q.kappa_max3 / q.kappa_min3

    # The amorphous-limit floor. Volume, density and atom count come from the
    # same screen table as the moduli, so every input describes one cell.
    scr = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"),
                      dtype={"material_id": str},
                      usecols=["material_id", "K_VRH_pred", "G_VRH_pred",
                               "Volume (A3)", "Density (g cm-3)", "Number of Atoms"])
    scr["kappa_cahill"] = cahill_min(scr.K_VRH_pred, scr.G_VRH_pred,
                                     scr["Density (g cm-3)"], scr["Volume (A3)"],
                                     scr["Number of Atoms"])
    q = q.merge(scr[["material_id", "kappa_cahill"]], on="material_id", how="left")
    q["below_floor"] = q.kappa_max3 < q.kappa_cahill
    # np.maximum compares two columns element by element and keeps the larger.
    q["kappa_floored"] = np.maximum(q.kappa_max3, q.kappa_cahill)
    # rank(method="first") numbers rows 1..N by kappa_floored; ties broken by order.
    q["rank_floored"] = q.kappa_floored.rank(method="first").astype(int)
    q = q.sort_values("kappa_max3").reset_index(drop=True)
    q["rank"] = np.arange(1, len(q) + 1).astype(object)  # object: no "1.0" printing
    q["tier"] = np.select([q["rank"] <= 20, q["rank"] <= 50],
                          ["A1: request", "A2: queued"], "A3: documented")
    q["status"] = "QUEUE"

    # ---- 4. the negative controls, unchanged from step 75 ------------------
    ctl = pd.read_csv(os.path.join(RESULTS, "75_oxide_dft_queue.csv"),
                      dtype={"material_id": str})
    ctl = ctl[ctl.status == "CONTROL"].copy()
    ctl["origin"] = "step 61 control"
    ctl["rank"] = np.nan

    cols = ["status", "tier", "rank", "origin", "formula", "material_id",
            "Number of Atoms", "gamma_mlip"] + MB + \
           ["kappa_max3", "kappa_min3", "spread3", "binding3",
            "kappa_cahill", "below_floor", "kappa_floored", "rank_floored"]
    out = pd.concat([q.reindex(columns=cols), ctl.reindex(columns=cols)],
                    ignore_index=True)
    out.to_csv(os.path.join(RESULTS, "81_combined_dft_queue.csv"), index=False)

    # ---- 5. report ---------------------------------------------------------
    fmt = lambda v: f"{v:.3f}"
    a1 = q[q.tier == "A1: request"]
    print(f"\nA1: request  ({len(a1)})")
    print(a1[["rank", "origin", "formula", "material_id", "Number of Atoms",
              "gamma_mlip"] + MB + ["kappa_max3", "spread3", "binding3",
                                    "kappa_cahill", "below_floor", "rank_floored"]]
          .to_string(index=False, float_format=fmt))

    print("\nA1 by origin: " + ", ".join(f"{k} {v}" for k, v in
                                          a1.origin.value_counts().items()))
    a2 = q[q.tier == "A2: queued"]
    print(f"A2: queued   ({len(a2)})  kappa_max3 {a2.kappa_max3.min():.3f} - "
          f"{a2.kappa_max3.max():.3f}   by origin: "
          + ", ".join(f"{k} {v}" for k, v in a2.origin.value_counts().items()))
    a3 = q[q.tier == "A3: documented"]
    print(f"A3: documented ({len(a3)})  in the CSV, not requested")

    # What the switch from step 75 actually changes.
    old = pd.read_csv(os.path.join(RESULTS, "75_oxide_dft_queue.csv"),
                      dtype={"material_id": str})
    old_a1 = set(old[old.tier == "A1: request"].material_id)
    new_a1 = set(a1.material_id)
    print(f"\nagainst step 75's oxide-only A1: kept {len(old_a1 & new_a1)}, "
          f"dropped {len(old_a1 - new_a1)}, new {len(new_a1 - old_a1)}")

    print(f"\nmodel spread in A1: median {a1.spread3.median():.2f}x, "
          f"max {a1.spread3.max():.2f}x "
          f"({a1.loc[a1.spread3.idxmax(), 'formula']})")
    print(f"\nAMORPHOUS-LIMIT CHECK: below the Cahill floor - A1 {int(a1.below_floor.sum())}/{len(a1)}, "
          f"A2 {int(a2.below_floor.sum())}/{len(a2)}, all {int(q.below_floor.sum())}/{len(q)}")
    top_f = set(q[q.rank_floored <= 20].material_id)
    print(f"if ranked on kappa_floored instead, A1 would keep {len(top_f & set(a1.material_id))} "
          f"of these 20 and take {len(top_f - set(a1.material_id))} others")
    print(f"\nrequest = {len(a1)} candidates + {len(ctl)} controls "
          f"= {len(a1) + len(ctl)} calculations")
    print(f"wrote {RESULTS}/81_combined_dft_queue.csv")

    if args.skip_mp:
        return

    # ---- 6. has anyone already done this? (same lookup as step 75) --------
    from mp_api.client import MPRester
    from pymatgen.core import Composition
    import warnings; warnings.filterwarnings("ignore")   # one tqdm bar per query
    live = out[out.tier.isin(["A1: request", "C: negative control"])
               | (out.status == "CONTROL")]
    rows = []
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
                    material_ids=[x.material_id for x in same], fields=["material_id"]))
            rows.append({"tier": r.tier if isinstance(r.tier, str) else "C: negative control",
                         "formula": r.formula, "gnome_id": r.material_id,
                         "chemsys": chemsys, "mp_entries_in_chemsys": len(docs),
                         "mp_same_formula": len(same),
                         "mp_id": same[0].material_id if same else "",
                         "mp_has_elastic": n_el})
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(RESULTS, "81_prior_calculations.csv"), index=False)
    print(f"\nPRIOR CALCULATIONS - Materials Project, {len(pc)} crystals queried")
    print(f"   structure already in MP : {(pc.mp_same_formula > 0).sum()}")
    print(f"   ELASTIC TENSORS in MP   : {int((pc.mp_has_elastic > 0).sum())}"
          f"   <- a hit here means DFT moduli already exist; check before requesting")
    if (pc.mp_has_elastic > 0).any():
        print(pc[pc.mp_has_elastic > 0][["tier", "formula", "gnome_id", "mp_id"]]
              .to_string(index=False))
    print(f"wrote {RESULTS}/81_prior_calculations.csv")


if __name__ == "__main__":
    main()
