#!/usr/bin/env python
"""
Step 98 - columns for the supervisor's DFT moduli, and kappa_L from them
=========================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/98_shortlist_dft_moduli.py

WHY
---
The DFT on the final 15 will compute the elastic moduli (bulk K and shear G),
not kappa_L itself, because that is far cheaper (decided 2026-10-07). The list
does not change. kappa_L then comes from the DFT moduli through the same
physics chain the models use, so the only thing swapped is where K and G come
from.

dft/final_shortlist_15/index.csv already holds each model's predicted K and G
(step 94: K_GPa_<model>, G_GPa_<model>). This step adds empty columns for the
DFT values and fills in the kappa columns for every row where they are present.

WHAT IS ADDED (inserted just before lit_status)
------------------------------------------------
    K_GPa_DFT, G_GPa_DFT   the supervisor's DFT moduli in GPa (Voigt-Reuss-Hill
                           averages). Blank until typed in. A re-run KEEPS whatever
                           is there, so type them straight into index.csv and re-run.
    gamma_poisson_DFT      gamma from the DFT Poisson ratio (from K/G)
    kappa_poisson_DFT      Slack kappa_L (W/m/K, 300 K) from the DFT K, G and that gamma
                           (the same chain as kappa_poisson_<model>)
    kappa_mlip_DFT         the DFT K, G with the MLIP phonon gamma (the same chain
                           as kappa_mlip_<model>, which is the one the 15 were picked on)
The volume, density and atom count come from each crystal's CIF in the folder.

CHECKS (the script stops if any fails)
--------------------------------------
1. Feeding each MODEL's K and G through the DFT code path must give back that
   model's kappa_poisson_* and kappa_mlip_* already in index.csv. This proves
   the code that will turn DFT moduli into kappa is the same chain that made the
   model numbers.
2. Each CIF is the crystal its row names (formula and atom count).
3. A DFT entry needs both K and G, and both must be > 0.

OUTPUTS (rewritten in place: columns added, nothing else changed)
-------
    dft/final_shortlist_15/index.csv   + the 5 columns above
    ~/Desktop/final_shortlist_15/ and ~/Desktop/final_shortlist_15_cifs.zip
                                       refreshed, the same way step 94 does it

RE-RUN ORDER: after step 94. (Step 94 keeps these columns if it is re-run.)
"""

import glob                              # lists files matching a pattern like "*.cif"
import os
import shutil                            # copies whole folders
import sys
import zipfile                           # writes .zip archives
from importlib import import_module      # imports a file whose name starts with a digit

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dft", "final_shortlist_15")
INDEX = os.path.join(OUT, "index.csv")
ZIP = os.path.expanduser("~/Desktop/final_shortlist_15_cifs.zip")      # same zip as steps 91/94
DESK = os.path.expanduser("~/Desktop/final_shortlist_15")              # unzipped copy

sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
_s74 = import_module("74_oxide_shortlist_real_gamma")   # kappa_cal = the Slack chain step 94 used

MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]
TYPED = ["K_GPa_DFT", "G_GPa_DFT"]                       # typed in by hand, never overwritten
DERIVED = ["gamma_poisson_DFT", "kappa_poisson_DFT", "kappa_mlip_DFT"]
TOL = 1e-6         # two numbers "agree" if they differ by less than one part in a million


def geometry(files):
    """Cell volume (A^3), density (g/cm^3) and atom count of each CIF, as a table."""
    from pymatgen.core import Structure                  # imported here: only this part needs it
    rows = []
    for f in files:
        st = Structure.from_file(os.path.join(OUT, f))
        rows.append({"file": f, "volume_A3": st.volume, "density": st.density,
                     "n_atoms_cif": len(st), "formula_cif": st.composition.reduced_formula})
    return pd.DataFrame(rows)


def chain(K, G, g, gamma_mlip):
    """K, G in GPa -> (gamma_poisson, kappa_poisson, kappa_mlip), one value per crystal.

    `g` is the geometry table. _s74.kappa_cal returns the Slack kappa with gamma
    taken from the Poisson ratio, plus that gamma. gamma enters Slack only as
    exp(-gamma), so swapping in the MLIP gamma multiplies kappa by
    exp(gamma_poisson - gamma_mlip) - the same step 94 used.
    """
    kp, gp = _s74.kappa_cal(np.asarray(K, float), np.asarray(G, float),
                            g.volume_A3.values, g.density.values, g.n_atoms_cif.values)
    return gp, kp, kp * np.exp(gp - np.asarray(gamma_mlip, float))


def check(label, a, b):
    """Stop the script unless a and b agree element by element (relative TOL)."""
    rel = np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1))
    print(f"  {label:60s} max relative diff {rel:.1e}")
    if not rel <= TOL:                  # written as "not <=" so that NaN also fails
        raise SystemExit(f"CHECK FAILED: {label}")


def refresh_desktop_copies():
    """Rebuild the Desktop zip and the unzipped Desktop folder (as step 94 does)."""
    # "**" with recursive=True also looks inside sub-folders (controls/, reserves/)
    files = sorted(glob.glob(os.path.join(OUT, "**", "*.*"), recursive=True))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            # arcname = the path the file gets INSIDE the zip
            z.write(f, arcname=os.path.join("final_shortlist_15", os.path.relpath(f, OUT)))
    shutil.copytree(OUT, DESK, dirs_exist_ok=True)      # dirs_exist_ok: write over the old copy
    print(f"wrote {ZIP} and {DESK}/ ({len(files)} files)")


def main():
    print("step 98 - DFT moduli columns for the final 15")
    # float_precision="round_trip" reads every number back exactly as written, so
    # the columns this step does not touch are re-written byte for byte
    d = pd.read_csv(INDEX, dtype={"material_id": str}, float_precision="round_trip")
    if len(d) != 15:
        raise SystemExit(f"{INDEX} has {len(d)} rows, expected 15")

    # keep any DFT values already typed in; start blank columns if this is the first run.
    # pd.to_numeric(..., errors="raise") stops on anything that is not a number
    for c in TYPED:
        d[c] = pd.to_numeric(d[c], errors="raise") if c in d.columns else np.nan
    # the derived columns are always recomputed, so drop any old ones
    d = d.drop(columns=[c for c in DERIVED if c in d.columns])

    # ---- check 2: each CIF is the crystal its row says ----
    from pymatgen.core import Composition
    g = geometry(d.file)
    bad = [(f, a) for f, a, b, n, m in zip(d.file, d.formula, g.formula_cif, d.n_atoms, g.n_atoms_cif)
           if Composition(a).reduced_formula != b or n != m]
    if bad:
        raise SystemExit(f"CIF does not match its row: {bad}")
    print("  15/15 CIFs match their row's formula and atom count")

    # ---- check 1: the DFT code path reproduces every model's stored kappa ----
    for m in MODELS:
        gp, kp, km = chain(d[f"K_GPa_{m}"], d[f"G_GPa_{m}"], g, d.gamma_mlip)
        check(f"{m}'s K, G through this code = its gamma_poisson", gp, d[f"gamma_poisson_{m}"])
        check(f"{m}'s K, G through this code = its kappa_poisson", kp, d[f"kappa_poisson_{m}"])
        check(f"{m}'s K, G through this code = its kappa_mlip", km, d[f"kappa_mlip_{m}"])

    # ---- check 3, then the real thing: kappa from the DFT moduli ----
    have_K, have_G = d.K_GPa_DFT.notna(), d.G_GPa_DFT.notna()
    if (have_K != have_G).any():        # "!=" row by row: one given without the other
        raise SystemExit(f"K and G must both be given: {list(d.formula[have_K != have_G])}")
    if (d.loc[have_K, TYPED] <= 0).any().any():
        raise SystemExit("a DFT modulus is <= 0 GPa")
    gp, kp, km = chain(d.K_GPa_DFT, d.G_GPa_DFT, g, d.gamma_mlip)
    # np.where(condition, a, b) = a where the condition holds, else b (blank here)
    d["gamma_poisson_DFT"] = np.where(have_K, gp, np.nan)
    d["kappa_poisson_DFT"] = np.where(have_K, kp, np.nan)
    d["kappa_mlip_DFT"] = np.where(have_K, km, np.nan)

    # put the new columns just before lit_status (the long text columns stay last)
    rest = [c for c in d.columns if c not in TYPED + DERIVED]
    at = rest.index("lit_status")
    d = d[rest[:at] + TYPED + DERIVED + rest[at:]]
    d.to_csv(INDEX, index=False)
    print(f"\nwrote {INDEX}: {d.shape[0]} rows x {d.shape[1]} columns; "
          f"DFT moduli present for {int(have_K.sum())} of 15")
    refresh_desktop_copies()

    # ---- what to look at once DFT values exist ----
    if have_K.any():
        s = d[have_K]
        print("\npredicted / DFT, per crystal (1.00 = exact):")
        show = pd.DataFrame({"formula": s.formula})
        for m in MODELS:
            show[f"K {m}"] = s[f"K_GPa_{m}"] / s.K_GPa_DFT
            show[f"G {m}"] = s[f"G_GPa_{m}"] / s.G_GPa_DFT
        print(show.round(2).to_string(index=False))
        print("\nkappa_L at 300 K, W/m/K, from the DFT moduli:")
        print(s[["formula", "K_GPa_DFT", "G_GPa_DFT", "kappa_poisson_DFT", "kappa_mlip_DFT",
                 "kappa_max3"]].round(3).to_string(index=False))
    else:
        print("\nno DFT moduli yet: type them into K_GPa_DFT and G_GPa_DFT, then re-run this step")


if __name__ == "__main__":
    main()
