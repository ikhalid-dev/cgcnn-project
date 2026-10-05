#!/usr/bin/env python
"""
Step 94 - each model's own predictions added to the supervisor's CSVs
=====================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/94_shortlist_model_outputs.py

WHY
---
dft/final_shortlist_15/index.csv (written by step 91) only gives a SUMMARY of
the three moduli models: kappa_max3 is the highest of their three kappas, and
kappa_typical / kappa_stress are stress-tested versions of it. The supervisor
should be able to see what EACH model predicted, and what the direct kappa
model predicted, without opening the results folder. This step adds those
columns. No neural network is run: every number was already computed and saved
by an earlier step. The one exception is newbase's moduli for crystals that
were not in the halide screen (5 of the 15 + the 4 controls). Those were never
saved, so they are re-predicted the same way steps 74 and 92 did it, with the
same saved XGBoost files, which give the same answer every time.

WHAT IS ADDED (one value per crystal; <model> = ALIGNN, CGCNN-ens, newbase)
---------------------------------------------------------------------------
    K_GPa_<model>, G_GPa_<model>  the model's predicted bulk and shear modulus
    kappa_mlip_<model>            Slack kappa_L (W/m/K, 300 K) from those moduli plus
                                  the MLIP phonon gamma, which is the chain the 15 were
                                  picked on. Highest of the three = kappa_max3
    gamma_poisson_<model>         gamma from the model's own Poisson ratio (from K/G)
    kappa_poisson_<model>         Slack kappa_L with that gamma (step 92's chain)
    kappa_poisson_max3            the highest of those three
    kappa_direct                  the direct ALIGNN model (structure -> kappa_L,
                                  trained on 6,641 PhoNIX DFT values; crosswork 05)

CHECKS (the script stops if any of them fails)
----------------------------------------------
1. Putting each model's K and G through the Slack formula gives back the
   kappa that model's screen stored (ALIGNN step 57, CGCNN-ens step 13).
2. Swapping the Poisson gamma for the MLIP gamma gives back step 83's stored
   per-model kappa. The highest of the three must equal kappa_max3 (for the 15)
   or kappa_pred_slack_300K (for the controls).
3. For the 15, the Poisson numbers equal the ones step 92 saved.
4. Every row's formula agrees between the sources. No cell is left blank.
   The columns already in each CSV are not changed (they are only reordered
   around the new ones).

OUTPUTS (rewritten in place: new columns added, existing ones untouched)
-------
    dft/final_shortlist_15/index.csv            15 rows
    dft/final_shortlist_15/controls/index.csv   4 rows
    dft/final_shortlist_15/reserves/index.csv   8 rows, built here from step 91's table
                                                and the reserve CIFs (moved in 2026-10-06)
    ~/Desktop/final_shortlist_15_cifs.zip       the whole folder, zipped
    ~/Desktop/final_shortlist_15/               the whole folder, unzipped copy

RE-RUN ORDER: step 91 rewrites the 15 and controls index.csv files WITHOUT
these columns, so if step 91 is ever re-run, run this step again afterwards.
(Step 91 can no longer run as-is: the older lists it read, dft/kappa_L_cifs/
and dft/kappa_L_low_range/, were removed on 2026-10-06 and live in git history.)
"""

import glob                              # lists files matching a pattern like "*.cif"
import os
import shutil                            # copies whole folders
import zipfile                           # writes .zip archives
# xgboost (used for newbase) crashes on this laptop if more than one OpenMP
# thread runs. os.environ is the process's environment variables; it must be
# set BEFORE numpy/xgboost are imported, so it sits right after "import os".
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from importlib import import_module      # imports a file whose name starts with a digit

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
OUT = os.path.join(ROOT, "dft", "final_shortlist_15")
# os.path.expanduser turns "~" into the home folder (/Users/mac)
DIRECT = os.path.expanduser("~/Desktop/transport_program/crosswork/data/05_dft_list_direct_klat.csv")
ZIP = os.path.expanduser("~/Desktop/final_shortlist_15_cifs.zip")      # same zip step 91 makes
DESK = os.path.expanduser("~/Desktop/final_shortlist_15")              # unzipped copy
RES_DIR = os.path.join(OUT, "reserves")

# sys.path = the folders Python searches on "import"; this lets us reuse earlier steps
sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
_s74 = import_module("74_oxide_shortlist_real_gamma")   # kappa_cal, featurize, predict_baseline
_s92 = import_module("92_shortlist_poisson_check")      # one_row_each

MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]
TOL = 1e-6         # two numbers "agree" if they differ by less than one part in a million

# The new column names, in the order they will appear. A "list comprehension"
# [f"K_GPa_{m}" for m in MODELS] builds one name per model in a single line.
NEW = ([f"K_GPa_{m}" for m in MODELS] + [f"G_GPa_{m}" for m in MODELS]
       + [f"kappa_mlip_{m}" for m in MODELS]
       + [f"gamma_poisson_{m}" for m in MODELS] + [f"kappa_poisson_{m}" for m in MODELS]
       + ["kappa_poisson_max3", "kappa_direct"])


def check(label, a, b):
    """Stop the script unless a and b agree element by element (relative TOL)."""
    # np.asarray turns a pandas column or a list into a plain array of numbers
    rel = np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1))
    print(f"  {label:58s} max relative diff {rel:.1e}")
    if not rel <= TOL:                  # written as "not <=" so that NaN also fails
        raise SystemExit(f"CHECK FAILED: {label}")


def model_outputs(ids):
    """Every new column for the crystals in `ids`, as one table, one row each."""
    one = _s92.one_row_each             # keeps one row per crystal, stops if duplicates disagree

    # step 83: the MLIP gamma, each model's MLIP-chain kappa, and which screen
    # (halide / oxide / step-61 control) each crystal came from
    q = pd.read_csv(os.path.join(RESULTS, "83_trusted_dft_queue.csv"), dtype={"material_id": str})
    q = one(q, ids, ["formula", "origin", "gamma_mlip"] + MODELS)

    # ALIGNN's moduli and the Poisson-chain kappa its screen stored (step 57)
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    a = one(a, ids, ["K_alignn", "G_alignn", "Kappa_alignn"])

    # CGCNN ensemble's moduli and its stored Poisson-chain kappa (step 13)
    cg = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"), dtype={"material_id": str})
    cg = one(cg, ids, ["K_VRH_pred", "G_VRH_pred", "Kappa_cal (W m-1 K-1)"])

    # cell volume, density and atom count: the Slack formula needs them
    st = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"), dtype={"material_id": str})
    st = one(st, ids, ["Volume (A3)", "Density (g cm-3)", "Number of Atoms"])

    # newbase moduli: halide-screen crystals use step 71's saved values; all the
    # others are re-predicted from a fresh featurisation, the way step 74 did
    # (dict(zip(...)) pairs each id with its origin: {"d7e7d897af": "halide", ...})
    origin = dict(zip(q.material_id, q.origin))
    halide = [m for m in ids if origin[m] == "halide"]
    other = [m for m in ids if origin[m] != "halide"]
    nb = pd.read_csv(os.path.join(RESULTS, "71_screen_ranked_5model.csv"), dtype={"material_id": str})
    nb = one(nb, halide, ["K_newbase", "G_newbase"])
    if other:
        print(f"  re-predicting newbase moduli for {len(other)} non-halide crystals ...")
        F = _s74.featurize(other)
        p = _s74.predict_baseline(F)
        # pd.concat stacks two tables on top of each other
        nb = pd.concat([nb, pd.DataFrame({"material_id": F.index, "K_newbase": p["K"],
                                          "G_newbase": p["G"]})], ignore_index=True)

    # the direct model's prediction (crosswork 05); its formula is kept for a check
    dr = pd.read_csv(DIRECT, dtype={"material_id": str}).rename(columns={"formula": "formula_direct"})
    dr = one(dr, ids, ["formula_direct", "klat_direct"])

    # glue it all together, one row per id, in the order given.
    # how="left" keeps every id even if a source lacks it (it becomes blank, caught below)
    d = pd.DataFrame({"material_id": ids})
    for part in [q, a, cg, st, nb, dr]:
        d = d.merge(part, on="material_id", how="left")

    # ---- each model: K, G -> Poisson gamma and kappa -> swap in MLIP gamma ----
    K = {"ALIGNN": d.K_alignn, "CGCNN-ens": d.K_VRH_pred, "newbase": d.K_newbase}
    G = {"ALIGNN": d.G_alignn, "CGCNN-ens": d.G_VRH_pred, "newbase": d.G_newbase}
    stored = {"ALIGNN": d.Kappa_alignn, "CGCNN-ens": d["Kappa_cal (W m-1 K-1)"]}
    for m in MODELS:
        # .values = the plain numbers without pandas' row labels
        kp, gp = _s74.kappa_cal(K[m].values, G[m].values, d["Volume (A3)"].values,
                                d["Density (g cm-3)"].values, d["Number of Atoms"].values)
        if m in stored:                 # newbase's screens did not store this one
            check(f"{m}: K,G -> Slack = its screen's stored kappa", kp, stored[m])
        # gamma enters the Slack formula only as exp(-gamma), so changing gamma
        # multiplies kappa by exp(old gamma - new gamma)
        check(f"{m}: same with MLIP gamma = step 83's stored kappa",
              kp * np.exp(gp - d.gamma_mlip), d[m])
        d[f"K_GPa_{m}"], d[f"G_GPa_{m}"] = K[m], G[m]
        d[f"kappa_mlip_{m}"] = d[m]     # step 83's own number (agrees with the line above)
        d[f"gamma_poisson_{m}"], d[f"kappa_poisson_{m}"] = gp, kp
    d["kappa_poisson_max3"] = d[[f"kappa_poisson_{m}" for m in MODELS]].max(axis=1)
    d["kappa_direct"] = d.klat_direct
    return d


def add_columns(path, before):
    """Read one index.csv and return (old table, new table with the model columns)."""
    old = pd.read_csv(path, dtype={"material_id": str})
    # if this step was run before, drop its columns so a re-run replaces them
    old = old[[c for c in old.columns if c not in NEW]]
    d = model_outputs(list(old.material_id))

    # the formula in every source must be the same crystal as the CSV row
    for col in ["formula", "formula_direct"]:
        bad = d[col].values != old.formula.values
        if bad.any():
            raise SystemExit(f"{path}: {col} disagrees for {list(old.formula[bad])}")
    if d[NEW].isna().any().any():       # .isna() marks blanks; .any().any() = "anywhere at all"
        raise SystemExit(f"{path}: blank cells in {list(d[NEW].columns[d[NEW].isna().any()])}")

    new = old.copy()
    for c in NEW:
        new[c] = d[c].values            # same row order as `old`, so plain values are safe
    # put the new columns just before `before` (or at the end if there is no such column)
    keep = list(old.columns)
    at = keep.index(before) if before in keep else len(keep)
    return old, new[keep[:at] + NEW + keep[at:]]


def build_reserves_index():
    """Write reserves/index.csv: one row per reserve CIF, step 91's numbers.

    The 8 reserve CIFs were moved into reserves/ by hand (git mv) when the older
    lists were removed. This rebuilds their table from step 91's saved results
    and checks each CIF really is the crystal its row says it is.
    """
    from pymatgen.core import Composition, Structure     # imported here: only this part needs it
    s = pd.read_csv(os.path.join(RESULTS, "91_final_shortlist.csv"), dtype={"material_id": str})
    r = s[s.decision.str.startswith("reserve")].copy()
    files = glob.glob(os.path.join(RES_DIR, "*.cif"))
    # the id is the last "__" part of each name: res_01__Cs2CuAgO2__7b1eb74d1b.cif -> 7b1eb74d1b
    # (basename drops the folder; [:-4] drops ".cif"; split("__")[-1] takes the last piece)
    by_id = {os.path.basename(f)[:-4].split("__")[-1]: f for f in files}
    if set(by_id) != set(r.material_id):                 # a set ignores order, so this compares contents
        raise SystemExit(f"reserve CIFs {sorted(by_id)} do not match step 91's {sorted(r.material_id)}")
    for row in r.itertuples():
        st = Structure.from_file(by_id[row.material_id])
        if (len(st) != row.n_atoms or st.composition.reduced_formula
                != Composition(row.formula).reduced_formula):
            raise SystemExit(f"{by_id[row.material_id]} is not {row.formula} ({row.n_atoms} atoms)")
    r["file"] = [os.path.basename(by_id[m]) for m in r.material_id]
    # "reserve: tier 1, rare oxidation state" -> "tier 1, rare oxidation state"
    r["reserve_reason"] = r.decision.str.replace("reserve: ", "", regex=False)
    cols = ["file", "formula", "material_id", "n_atoms", "reserve_reason", "gamma_mlip",
            "kappa_max3", "kappa_typical", "kappa_stress", "kappa_cahill"]
    path = os.path.join(RES_DIR, "index.csv")
    r.sort_values("file")[cols].to_csv(path, index=False)   # res_01, res_02, ... = order of use
    return path


def main():
    print("step 94 - model outputs into the supervisor's CSVs")
    sl_path = os.path.join(OUT, "index.csv")
    ct_path = os.path.join(OUT, "controls", "index.csv")

    print("\nthe final 15:")
    sl_old, sl = add_columns(sl_path, before="lit_status")   # keep the long text columns last
    mlip = sl[[f"kappa_mlip_{m}" for m in MODELS]]
    check("15: highest of the three kappa_mlip = kappa_max3", mlip.max(axis=1), sl.kappa_max3)
    p92 = pd.read_csv(os.path.join(RESULTS, "92_shortlist_poisson_check.csv"), dtype={"material_id": str})
    p92 = p92.set_index("material_id").loc[sl.material_id]   # step 92's rows, in our order
    for m in MODELS:
        check(f"15: kappa_poisson_{m} = step 92", sl[f"kappa_poisson_{m}"], p92[f"kP_{m}"])
        check(f"15: gamma_poisson_{m} = step 92", sl[f"gamma_poisson_{m}"], p92[f"gP_{m}"])
    check("15: kappa_poisson_max3 = step 92", sl.kappa_poisson_max3, p92.poisson_max3)

    print("\nthe 4 controls:")
    ct_old, ct = add_columns(ct_path, before=None)
    check("controls: highest kappa_mlip = kappa_pred_slack_300K",
          ct[[f"kappa_mlip_{m}" for m in MODELS]].max(axis=1), ct.kappa_pred_slack_300K)

    print("\nthe 8 reserves:")
    rs_path = build_reserves_index()
    rs_old, rs = add_columns(rs_path, before=None)
    check("reserves: highest kappa_mlip = kappa_max3",
          rs[[f"kappa_mlip_{m}" for m in MODELS]].max(axis=1), rs.kappa_max3)

    # every check passed: write the files (index=False = no extra row-number column)
    sl.to_csv(sl_path, index=False)
    ct.to_csv(ct_path, index=False)
    rs.to_csv(rs_path, index=False)
    print(f"\nwrote {sl_path}: {sl.shape[0]} rows x {sl.shape[1]} columns "
          f"({len(NEW)} new)")
    print(f"wrote {ct_path}: {ct.shape[0]} rows x {ct.shape[1]} columns")
    print(f"wrote {rs_path}: {rs.shape[0]} rows x {rs.shape[1]} columns")

    # rebuild the Desktop zip so it carries the new CSVs (and the README).
    # "**" with recursive=True also looks inside sub-folders (controls/);
    # arcname = the path the file gets INSIDE the zip
    files = sorted(glob.glob(os.path.join(OUT, "**", "*.*"), recursive=True))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.join("final_shortlist_15", os.path.relpath(f, OUT)))
    print(f"wrote {ZIP}: {len(files)} files")

    # and the same folder unzipped on the Desktop. copytree copies a folder with
    # everything inside; dirs_exist_ok=True lets it write over an older copy
    shutil.copytree(OUT, DESK, dirs_exist_ok=True)
    # a file renamed or removed here would survive in the old Desktop copy: name any such file
    rel = lambda root: {os.path.relpath(f, root) for f in
                        glob.glob(os.path.join(root, "**", "*.*"), recursive=True)}
    stale = sorted(rel(DESK) - rel(OUT))                 # "-" on sets = in the first, not the second
    print(f"wrote {DESK}/ ({len(files)} files)"
          + (f"  WARNING, old files not in the repo folder: {stale}" if stale else ""))

    # a compact view of the kappa columns, rounded for reading
    show = ["formula"] + [f"kappa_mlip_{m}" for m in MODELS] + ["kappa_max3"] \
        + [f"kappa_poisson_{m}" for m in MODELS] + ["kappa_poisson_max3", "kappa_direct"]
    short = {c: c.replace("kappa_", "").replace("CGCNN-ens", "CGCNN") for c in show}
    print("\nkappa_L at 300 K, W/m/K (mlip_* = MLIP-gamma chain, poisson_* = Poisson chain):")
    print(pd.concat([sl[show], ct[[c for c in show if c in ct.columns]], rs[show]])
          .rename(columns=short).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
