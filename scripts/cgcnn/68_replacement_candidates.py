#!/usr/bin/env python
"""
Step 68 - replacements for the two flagged candidates.

WHY
---
Two of the eight arm-A candidates carry a warning: CsRbSe3 is dynamically
unstable under our own phonons (one imaginary mode), and Cs4Sb4PdPt cannot be
processed by our phonon pipeline at all. Neither should go into a DFT request
that is meant to be clean.

The arm-A pool is far larger than eight, so this reproduces step 61's selection
exactly and takes the next-ranked crystals as candidate replacements. They are
NOT accepted here - they are only nominated. Each one still has to survive a
phonon calculation before it can take a place in the queue, which is the whole
point of having found the problem in the first two.

RUN
    ml_env/bin/python scripts/cgcnn/68_replacement_candidates.py

OUTPUTS
    results/cgcnn/68_replacement_pool.csv   the next-ranked arm-A crystals
    data/replacement_cifs/*.cif             their structures, ready to test
"""
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
CIF_OUT = os.path.join(ROOT, "data", "replacement_cifs")

# Reproduced verbatim from step 61 so the ranking is the same one, not a new one.
MODELS = {"ALIGNN": "Kappa_alignn", "CGCNN-ens": "Kappa_cal_derived_matbench",
          "round9": "Kappa_r9_gamma", "tree": "Kappa_baseline"}
MAX_ATOMS = 10
LOW = 1.0
N_NOMINATE = 10          # more than the 2 needed: about half will fail the phonons


def main():
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    s = pd.read_csv(os.path.join(ROOT, "results/cgcnn/39_gnome_screen_all_gamma.csv"),
                    dtype={"material_id": str})
    sg = pd.read_csv(os.path.join(ROOT, "results/cgcnn/gnome_screen_spacegroups.csv"),
                     dtype={"material_id": str},
                     usecols=["material_id", "space_group", "crystal_system"])

    d = a.merge(s, on="material_id", suffixes=("", "_dup")).merge(sg, on="material_id", how="left")
    d = d[d["alignn_prediction_reliable"]].copy()
    for name, col in MODELS.items():
        d[name] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(MODELS))

    d["kappa_max"] = d[list(MODELS)].max(axis=1)     # the most pessimistic model
    d["kappa_min"] = d[list(MODELS)].min(axis=1)
    d["n_models_low"] = (d[list(MODELS)] <= LOW).sum(axis=1)
    small = d[d["Number of Atoms"] <= MAX_ATOMS]
    pool = small[small.n_models_low == 4].sort_values("kappa_max")

    already = set(pd.read_csv(os.path.join(RESULTS, "61_dft_validation_set.csv"),
                              dtype={"material_id": str}).material_id)
    fresh = pool[~pool.material_id.isin(already)].head(N_NOMINATE).copy()
    fresh["rank_in_pool"] = range(9, 9 + len(fresh))

    print("=" * 78)
    print("STEP 68 - NOMINATING REPLACEMENTS")
    print("=" * 78)
    print(f"\n  arm-A pool (all four models call it low, <= {MAX_ATOMS} atoms): "
          f"{len(pool)} crystals")
    print(f"  already in the queue: 8.  Nominating the next {len(fresh)}.\n")
    cols = ["rank_in_pool", "formula", "material_id", "Number of Atoms",
            "space_group", "kappa_max", "ALIGNN", "K_alignn", "G_alignn"]
    print(fresh[cols].round(3).to_string(index=False))

    # structures, from the GNoME archive, keyed by composition rather than id
    summary = pd.read_csv(os.path.join(ROOT, "gnome_data/stable_materials_summary.csv"),
                          usecols=["Composition", "MaterialId"], dtype=str)
    comp = dict(zip(summary.MaterialId, summary.Composition))
    os.makedirs(CIF_OUT, exist_ok=True)
    z = zipfile.ZipFile(os.path.join(ROOT, "gnome_data/by_composition.zip"))
    names = set(z.namelist())
    got = 0
    for r in fresh.itertuples():
        entry = f"by_composition/{comp.get(r.material_id)}.CIF"
        if entry in names:
            open(os.path.join(CIF_OUT, f"{r.material_id}.cif"), "wb").write(z.read(entry))
            got += 1
        else:
            print(f"  MISSING CIF for {r.formula} ({r.material_id})")

    out = os.path.join(RESULTS, "68_replacement_pool.csv")
    fresh[cols + ["crystal_system", "kappa_min", "CGCNN-ens", "round9", "tree"]].to_csv(out, index=False)
    print(f"\n  {got} CIFs -> {CIF_OUT}")
    print(f"  -> {out}")
    print("\n  NEXT: run the phonons before accepting any of them.")
    print("  mlip_env/bin/python scripts/01_gamma_phonon.py \\")
    print(f"      --cif-dir {os.path.relpath(CIF_OUT, ROOT)} --tag replacements \\")
    print("      --device cpu --max-atoms 200        (in ~/Desktop/data_generation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
