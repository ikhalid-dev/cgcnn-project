#!/usr/bin/env python
"""
Step 85 - which PhoNIX crystals are also matbench crystals?
===========================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/85_match_phonix_matbench.py

WHY
---
The DFT run will compute lattice thermal conductivity (kappa_L), not elastic
moduli. Steps 82-83 only measured how well our screen predicts DFT MODULI, so
they cannot say how often a kappa_L run will confirm "low".

PhoNIX (the coherent-heat project's dataset) holds DFT kappa_L at 300 K for
6,641 Materials Project crystals, as klat = kp + kc:
    kp  particle-like (Peierls) conductivity - the usual Boltzmann-transport number
    kc  wave-like (coherent) conductivity    - only appears in very low-kappa crystals

To test the screen against those numbers we must know, for each PhoNIX crystal,
whether our models have seen it. Our models were trained on matbench, which
also comes from the Materials Project but labels its crystals mb-00000,
mb-00001, ... with no MP id. So we match by STRUCTURE:

    1. bucket matbench by reduced formula  (cheap: only same-formula pairs are compared)
    2. pymatgen's StructureMatcher decides whether two cells are the same crystal

Then every PhoNIX crystal gets one of three labels:
    test        in our 1,648 held-out test crystals - ALIGNN, CGCNN and newbase
                predictions all exist and none of the models trained on it
    train/val   in matbench, but the ALIGNN/CGCNN models trained on it
                (newbase still has an out-of-fold prediction for it)
    not in mb   no model has a prediction yet

This step only matches and COUNTS. How many low-kappa_L crystals land in the
test set decides whether a kappa_L confusion matrix can say anything at all -
that is the question to answer before building one.

OUTPUTS
    results/cgcnn/85_phonix_matbench_match.csv   one row per PhoNIX crystal
    results/cgcnn/85_phonix_census.csv           the counts table printed below
    results/cgcnn/85_phonix_census.png
"""
import ast                     # turns the text "{'numbers': [8, 8], ...}" back into a real dict
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # draw to a file, no window
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from pymatgen.core import Lattice, Structure                  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from matminer.datasets import load_dataset                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
PHONIX = os.path.expanduser("~/Desktop/a new project/data/phonix/data_all.csv")

LOW, VERY_LOW = 1.0, 0.5       # the screen's two "low" lines, W/m/K at 300 K


def load_phonix():
    """One row per material, with a pymatgen Structure.

    690 materials were calculated more than once. The coherent-heat project
    collapses repeats to the MEDIAN of each group (less sensitive to one odd
    repeat than the mean), so the same is done here.
    """
    cols = ["mp_id", "natoms_prim", "structure", "volume", "kp", "kc", "klat"]
    df = pd.read_csv(PHONIX, usecols=cols)
    print(f"PhoNIX: {len(df)} rows, {df.mp_id.nunique()} materials")

    # the dataset's own guarantee - checked rather than trusted
    assert np.allclose(df.kp + df.kc, df.klat), "kp + kc should equal klat"

    # groupby("mp_id") splits the table into one small table per material;
    # .agg({...}) then reduces each small table to one row, column by column
    per = df.groupby("mp_id").agg({"kp": "median", "kc": "median", "klat": "median",
                                   "natoms_prim": "first", "volume": "first",
                                   "structure": "first"}).reset_index()

    structures = []
    for text in per.structure:
        s = ast.literal_eval(text)              # safe parsing: only plain Python literals allowed
        # positions are CARTESIAN (Angstrom) - checked on a sample: converting
        # them with the cell gives fractions between 0 and 1
        structures.append(Structure(Lattice(s["cell"]), s["numbers"], s["positions"],
                                    coords_are_cartesian=True))
    per["structure"] = structures
    per["n_atoms"] = [len(s) for s in structures]
    assert (per.n_atoms == per.natoms_prim).all(), "PhoNIX cells should be primitive"
    per["formula"] = [s.composition.reduced_formula for s in structures]
    return per.drop(columns="natoms_prim")


def load_matbench():
    """matbench_log_kvrh, with the SAME mb ids as data_full/labels.csv.

    Step 01b named each crystal by its ROW NUMBER in this dataset
    (row 42 -> "mb-00042"). If the row order ever changed, every id would
    point at the wrong crystal, so the formula and atom count of every row
    are compared with labels.csv before anything is matched.
    """
    mb = load_dataset("matbench_log_kvrh")      # read from matminer's local cache
    mb = pd.DataFrame({"mb_id": [f"mb-{i:05d}" for i in range(len(mb))],
                       "structure": mb.structure})
    mb["formula"] = [s.composition.reduced_formula for s in mb.structure]
    mb["n_sites"] = [len(s) for s in mb.structure]

    labels = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))
    check = labels.merge(mb, on="mb_id", suffixes=("", "_mb"))
    assert len(check) == len(labels), "a labels.csv id is missing from matbench"
    assert (check.formula == check.formula_mb).all(), "formula disagrees - row order changed"
    assert (check.n_sites == check.n_sites_mb).all(), "atom count disagrees - row order changed"
    print(f"matbench: {len(mb)} crystals, ids agree with labels.csv on all {len(labels)}")
    return mb


def match(phonix, mb):
    # A dict maps a key to a value. Here: formula -> list of matbench row numbers.
    buckets = {}
    for i, f in enumerate(mb.formula):
        buckets.setdefault(f, []).append(i)    # setdefault: make the list the first time a formula is seen

    # primitive_cell=True  reduce both cells first, so a doubled cell still matches
    # scale=True (default) compare shapes at equal volume - PhoNIX re-relaxed
    #                      every crystal, so its volume can differ by a few %
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)

    mb_ids, n_hits = [], []
    start = time.time()
    for k, row in enumerate(phonix.itertuples()):
        hits = [i for i in buckets.get(row.formula, [])
                if matcher.fit(row.structure, mb.structure[i])]
        # keep the first hit; count all of them so duplicates are visible
        mb_ids.append(mb.mb_id[hits[0]] if hits else None)
        n_hits.append(len(hits))
        if (k + 1) % 1000 == 0:
            print(f"  matched {k + 1}/{len(phonix)}  ({(time.time() - start) / 60:.1f} min)")
    phonix["mb_id"] = mb_ids
    phonix["n_mb_hits"] = n_hits
    return phonix


def main():
    phonix = load_phonix()
    mb = load_matbench()
    phonix = match(phonix, mb)

    # which split each matched crystal sits in
    test_ids = set(pd.read_csv(os.path.join(RES, "82_test_predictions.csv")).mb_id)
    phonix["split"] = np.where(phonix.mb_id.isna(), "not in mb",
                      np.where(phonix.mb_id.isin(test_ids), "test", "train/val"))

    # quality check on the matches: volume per atom, PhoNIX vs matbench.
    # Same crystal, different DFT settings -> expect a few % difference, not 30 %.
    mb_vol = dict(zip(mb.mb_id, [s.volume / len(s) for s in mb.structure]))
    phonix["vol_ratio"] = [v / n / mb_vol[m] if isinstance(m, str) else np.nan
                           for v, n, m in zip(phonix.volume, phonix.n_atoms, phonix.mb_id)]

    m = phonix[phonix.mb_id.notna()]
    print(f"\nmatched to matbench: {len(m)} of {len(phonix)}")
    print(f"  PhoNIX crystals matching >1 matbench row: {(phonix.n_mb_hits > 1).sum()}")
    print(f"  matbench rows claimed by >1 PhoNIX crystal: {m.mb_id.duplicated().sum()}")
    print(f"  volume/atom PhoNIX / matbench: median {m.vol_ratio.median():.3f}, "
          f"outside 0.9-1.1: {((m.vol_ratio < 0.9) | (m.vol_ratio > 1.1)).sum()}")

    # ---- the census --------------------------------------------------------
    sets = {
        "all PhoNIX": phonix,
        "in matbench (any split)": m,
        "  of which train/val": phonix[phonix.split == "train/val"],
        "  of which TEST": phonix[phonix.split == "test"],
        "  TEST and <=20 atoms": phonix[(phonix.split == "test") & (phonix.n_atoms <= 20)],
        "not in matbench": phonix[phonix.split == "not in mb"],
    }
    rows = []
    for name, d in sets.items():
        rows.append({"set": name, "n": len(d),
                     f"klat<={LOW}": int((d.klat <= LOW).sum()),
                     f"klat<={VERY_LOW}": int((d.klat <= VERY_LOW).sum()),
                     "kc>kp (wave-like dominant)": int((d.kc > d.kp).sum()),
                     "median klat": d.klat.median(),
                     "median atoms": d.n_atoms.median()})
    census = pd.DataFrame(rows)
    print("\n" + census.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = phonix.drop(columns="structure")
    out.to_csv(os.path.join(RES, "85_phonix_matbench_match.csv"), index=False)
    census.to_csv(os.path.join(RES, "85_phonix_census.csv"), index=False)

    # ---- figure: where the low-kappa crystals are -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-2, 3.5, 45)
    for name, colour in [("not in mb", "#bbbbbb"), ("train/val", "#3b6ea8"), ("test", "#c9803a")]:
        d = phonix[phonix.split == name]
        ax[0].hist(d.klat, bins=bins, histtype="step", lw=2, color=colour, label=f"{name} ({len(d)})")
    ax[0].axvline(LOW, color="k", ls="--", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("DFT lattice thermal conductivity klat at 300 K (W/m/K)")
    ax[0].set_ylabel("crystals")
    ax[0].set_title("PhoNIX, split by what our models have seen")
    ax[0].legend()
    ax[1].hist(m.vol_ratio, bins=np.linspace(0.8, 1.2, 41), color="#3b6ea8")
    ax[1].set_xlabel("volume per atom, PhoNIX / matbench")
    ax[1].set_title(f"match check ({len(m)} matched pairs)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "85_phonix_census.png"), dpi=150)
    print("\nwrote 85_phonix_matbench_match.csv, 85_phonix_census.csv, 85_phonix_census.png")


if __name__ == "__main__":
    main()
