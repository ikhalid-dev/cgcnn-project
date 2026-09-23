#!/usr/bin/env python
"""
Step 76 - stage ranks 61-300 of the halide shortlist into Kaggle shards.
================================================================================

    mlip_env/bin/python scripts/cgcnn/76_stage_halide_shards.py

WHY 61-300 AND NOT 1-300
-------------------------
The step-72 pilot measured dynamic stability by rank band, and band 1 collapsed:

    band  ranks      usable   median imaginary modes
      1    1- 60      1/10          1725
      2   61-120      5/10             0
      3  121-180      8/10             0
      4  181-240      7/10             0
      5  241-300      7/10             0

Ranks 1-60 are where the three matbench models predicted kappa below anything
ever measured in a crystal (down to 0.0004 W/m/K from a single model). The
phonons say those are not ultralow-kappa materials, they are not stable
structures - the models were extrapolating off the end of their training data.
Buying them costs ~4 GPU-hours to learn almost nothing, so they are skipped.

Band 1's one survivor is recorded here and can be added by hand if wanted.

WHAT IS ALREADY PAID FOR
------------------------
40 of the 240 crystals in ranks 61-300 were themselves the pilot's bands 2-5
and already have a gamma. They are excluded rather than recomputed - the merge
at the end reads the pilot CSV alongside the shard CSVs.

SHARD PACKING
-------------
Cost per crystal is predicted from the workspace's own refitted law,
t = 0.058 * ndisp^1.087 * n_atoms_super^0.756, computed here with phonopy and
no GPU. The pilot measured that law over-predicting by ~27% (3.52 GPU-h actual
against 4.8 predicted), so the packing target is deliberately conservative and
the shards should come in under budget.

Crystals are packed longest-processing-time-first into the emptiest shard -
runtime spans a factor of ~50 here, so equal-COUNT shards would differ several-
fold in hours and one would hit Kaggle's 12 h wall while another idled.

OUTPUTS
    ~/Desktop/data_generation/data/halide_s00../s0N/   CIFs, one dir per shard
    results/cgcnn/76_halide_shard_plan.csv             the plan, with costs
"""
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
GEN = os.path.expanduser("~/Desktop/data_generation")
TARGET_H = 5.5          # per shard; 10.5 h deadline leaves plenty of headroom


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s08", os.path.join(GEN, "scripts/08_select_high_value_shortlist.py"))
    s08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s08)

    top = pd.read_csv(os.path.join(RESULTS, "71_top300_matbench3.csv"),
                      dtype={"material_id": str})
    band1 = top[top.rank3 <= 60]
    cand = top[(top.rank3 >= 61) & (top.rank3 <= 300)].copy()

    pilot = pd.read_csv(os.path.join(RESULTS, "72_pilot50_selection.csv"),
                        dtype={"material_id": str})
    done = set(pilot.material_id)
    cand["already_done"] = cand.material_id.isin(done)
    todo = cand[~cand.already_done].reset_index(drop=True)
    print(f"ranks 61-300            : {len(cand)}")
    print(f"already done in the pilot: {int(cand.already_done.sum())}")
    print(f"to compute               : {len(todo)}")
    print(f"(band 1 skipped: {len(band1)} crystals, 1/10 usable in the pilot)")

    # ---- extract CIFs to a staging dir, then price them --------------------
    summ = pd.read_csv(os.path.join(ROOT, "gnome_data/stable_materials_summary.csv"),
                       usecols=["Composition", "MaterialId"], dtype=str)
    comp = dict(zip(summ.MaterialId, summ.Composition))
    z = zipfile.ZipFile(os.path.join(ROOT, "gnome_data/by_composition.zip"))
    names = set(z.namelist())
    stage = os.path.join(GEN, "data", "_halide_stage")
    os.makedirs(stage, exist_ok=True)
    got = []
    for mid in todo.material_id:
        e = f"by_composition/{comp.get(mid)}.CIF"
        if e in names:
            open(os.path.join(stage, f"{mid}.cif"), "wb").write(z.read(e))
            got.append(mid)
    print(f"CIFs extracted           : {len(got)}")

    rows = []
    for mid in got:
        try:
            r = s08.cost_features(os.path.join(stage, f"{mid}.cif"))
            r["material_id"] = mid
            rows.append(r)
        except Exception as e:
            print(f"  pricing failed for {mid}: {type(e).__name__}")
    c = pd.DataFrame(rows)
    c["t_s"] = 0.058 * c.n_displacements ** 1.087 * c.n_atoms_super ** 0.756
    total_h = c.t_s.sum() / 3600
    print(f"\npredicted total          : {total_h:.1f} GPU-h "
          f"(the pilot measured this law ~27% pessimistic, so expect ~{total_h*0.73:.1f})")

    # ---- pack, longest first into the emptiest shard ----------------------
    n_shards = max(1, int(np.ceil(total_h / TARGET_H)))
    c = c.sort_values("t_s", ascending=False).reset_index(drop=True)
    loads = np.zeros(n_shards)
    assign = []
    for t in c.t_s:
        i = int(np.argmin(loads))     # emptiest shard so far
        assign.append(i)
        loads[i] += t
    c["shard"] = assign
    c["tag"] = [f"halide_s{i:02d}" for i in assign]

    print(f"\n{n_shards} shards:")
    for i in range(n_shards):
        sub = c[c.shard == i]
        d = os.path.join(GEN, "data", f"halide_s{i:02d}")
        os.makedirs(d, exist_ok=True)
        for mid in sub.material_id:
            os.rename(os.path.join(stage, f"{mid}.cif"), os.path.join(d, f"{mid}.cif"))
        print(f"   halide_s{i:02d}  {len(sub):3d} crystals  {sub.t_s.sum()/3600:.2f} GPU-h "
              f"(expect ~{sub.t_s.sum()/3600*0.73:.2f})")
    os.rmdir(stage) if not os.listdir(stage) else None

    out = c.merge(todo[["material_id", "formula", "rank3", "kappa_max3"]],
                  on="material_id")
    out.to_csv(os.path.join(RESULTS, "76_halide_shard_plan.csv"), index=False)
    print(f"\nwrote {RESULTS}/76_halide_shard_plan.csv")
    print("\nbuild and push each with:")
    for i in range(n_shards):
        print(f"   ml_env/bin/python kaggle/build_gamma_kernel.py "
              f"--cif-dir data/halide_s{i:02d} --tag halide_s{i:02d} "
              f"--title 'MLIP gamma halide s{i:02d}' --deadline-hours 10.5")


if __name__ == "__main__":
    main()
