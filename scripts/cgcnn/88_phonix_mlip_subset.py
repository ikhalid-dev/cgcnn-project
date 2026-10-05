#!/usr/bin/env python
"""
Step 88 - choose the PhoNIX crystals for MLIP gamma, price them, package them for Kaggle.
=========================================================================================

    ~/miniconda3/envs/mlip_env/bin/python scripts/cgcnn/88_phonix_mlip_subset.py

    (mlip_env, because counting phonon displacements needs phonopy, which only
    that environment has. Nothing here runs a force calculation - that happens
    on Kaggle.)

WHERE THIS SITS
---------------
Step 87 scored the screen with the OLD gamma: each model's gamma came from its
own predicted K and G through the Poisson relation. Against PhoNIX DFT kappa_L,
on 2,520 crystals no model has seen:

    max3 <= 1.0   precision 0.747 [0.704, 0.785]

The step-86 DFT list was built with the NEW gamma instead - MLIP phonon gamma
(MACE, three volumes, ~/Desktop/data_generation). Step 87 says nothing about
that chain. This step prepares the same test for it: compute MLIP gamma for a
subset of the SAME 2,520 crystals, then (step 89) score both chains on the
crystals that have both.

WHICH CRYSTALS - a stratified random sample
-------------------------------------------
Running all 2,520 is possible but the user asked for a subset first. A plain
random subset would waste most of the GPU on crystals far from the 1 W/m/K line,
whose call cannot change. So the pool is cut into four bands by step 87's
prediction, and a different FRACTION is drawn from each band:

    band          step-87 max3      why
    A  low        <= 1.0            the crystals the old chain calls low
    B  near       1.0 - 3.0         MLIP gamma can push these below 1
    C  mid        3.0 - 10          a check: does MLIP ever pull these below 1?
    D  high       > 10              same check, far end

Inside a band the pick is random (fixed seed), so a band's sample stands for
the whole band. Each crystal then carries a WEIGHT = 1 / (fraction drawn from
its band): a crystal from a band sampled 1-in-4 counts as 4 crystals when step
89 adds things up. That turns the sample back into an estimate for all 2,520.

HOW STEP 89 WILL BE JUDGED - written before any gamma exists
------------------------------------------------------------
Both chains use the SAME three models' K and G and the SAME crystals; only the
gamma differs. So the comparison is paired.

primary: precision at klat <= 1.0, max3 rule, MLIP chain minus Poisson chain,
         weighted, with a stratified paired bootstrap 95% CI
    >= +0.05, CI above 0     MLIP gamma makes the screen more trustworthy;
                             the step-86 list rests on the better chain
    CI includes 0            no measurable gain; judge step 86 by step 87's
                             Poisson calibration, MLIP gamma bought nothing here
    <= -0.05, CI below 0     MLIP gamma makes the screen WORSE; step 86's
                             list must be re-ranked on the Poisson chain

secondary: median DFT / predicted under each chain. Step 86 assumed MLIP gamma
         runs ~10% too HIGH (from a 38-material benchmark), so its kappa comes
         out too LOW. If that is right, the MLIP chain's median DFT/pred is
         clearly above the Poisson chain's (0.92 in step 87); if both sit near
         1, step 86's correction was not needed.

usable gamma = status ok, dynamically stable (no mode below -0.1 THz),
               0 < gamma < 10 - the production filter from steps 74/86.
               Production gamma = gamma_c300 (300 K, 0.3 THz cutoff).
Crystals MLIP calls unstable drop out of BOTH chains, and step 89 reports the
Poisson precision on the dropped crystals too, so the drop cannot hide a bias.

OUTPUTS
    data/phonix_cifs/<mp_id>.cif                     all 2,520, for pricing (gitignored)
    results/cgcnn/88_phonix_cost.csv                 displacements + supercell per crystal
    results/cgcnn/88_phonix_mlip_plan.csv            the sample: band, weight, shard, cost
    results/cgcnn/88_phonix_mlip_plan.png
    ~/Desktop/data_generation/data/phonix_cal_sNN/   one CIF folder per Kaggle shard
"""
import os
import shutil
import sys
import time
import warnings
from importlib import import_module   # imports a file whose name starts with a digit

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # draw to a file, no window
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")      # pymatgen warns about CIF rounding on every file
from pymatgen.core import Structure                       # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
CIF_ALL = os.path.join(ROOT, "data", "phonix_cifs")
DATAGEN = os.path.expanduser("~/Desktop/data_generation")

# sys.path is the list of folders Python searches on "import". Adding these two
# lets import_module find step 87 here and step 08 in the sibling repo.
sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
sys.path.insert(0, os.path.join(DATAGEN, "scripts"))
_s87 = import_module("87_phonix_poisson_calibration")    # load_pool(): structures + DFT kappa
_s08 = import_module("08_select_high_value_shortlist")   # cost_features(), fit_cost_law()

SEED = 88
# band name -> (lowest max3, highest max3, fraction to draw)
BANDS = {"A low":  (0.0, 1.0, 0.50),
         "B near": (1.0, 3.0, 0.35),
         "C mid":  (3.0, 10.0, 0.10),
         "D high": (10.0, np.inf, 0.05)}
SAFETY = 1.5           # the cost law is a fit; budget every shard as if it ran 1.5x slower
# the user: "run on all kaggle accounts, two shards" each. 3 accounts x 2 GPU
# sessions (Kaggle's per-account limit) = 6 shards running side by side.
ACCOUNTS = ["kaggle1", "kaggle2", "kaggle3"]
N_SHARDS = 2 * len(ACCOUNTS)
TAG = "phonix_cal_s{:02d}"

# runs made with the CURRENT kernel code (per-mode mesh saved, five cutoffs).
# The tell-tale is a gamma_hist_err column. Only these price what will run.
MESH_ERA = ["halide_recut_s00", "halide_recut_s01", "oxide_recut_s01",
            "t1_cut001", "t1_cut001_asr", "t1_cut030", "t1_cut030_asr"]


def load_pool():
    """Step 87's 2,520 scored crystals, with structures attached."""
    pool = _s87.load_pool()                                   # 2,615 crystals <= 20 atoms
    scored = pd.read_csv(os.path.join(RES, "87_phonix_poisson_predictions.csv"))
    # how="inner" keeps only crystals in BOTH tables: the 95 that newbase could
    # not featurise were never scored, so they stay out here too
    pool = pool.merge(scored[["mp_id", "kappa_max3", "kappa_min3", "spread3"]],
                      on="mp_id", how="inner")
    assert len(pool) == len(scored), "every scored crystal must have a structure"
    print(f"pool: {len(pool)} crystals scored in step 87")
    return pool


def write_cifs(pool):
    """One CIF per crystal, read back and checked - the kernel reads exactly these files."""
    os.makedirs(CIF_ALL, exist_ok=True)
    bad = []
    for r in pool.itertuples():
        path = os.path.join(CIF_ALL, f"{r.mp_id}.cif")
        if not os.path.exists(path):
            r.structure.to(filename=path)                     # pymatgen writes the CIF
        back = Structure.from_file(path)
        # same atoms, and the same space group at the kernel's own tolerance (1e-3)
        sg_mem = SpacegroupAnalyzer(r.structure, symprec=1e-3).get_space_group_number()
        sg_cif = SpacegroupAnalyzer(back, symprec=1e-3).get_space_group_number()
        if len(back) != r.n_atoms or sg_mem != sg_cif:
            bad.append((r.mp_id, len(back), r.n_atoms, sg_mem, sg_cif))
    if bad:
        raise SystemExit(f"{len(bad)} CIFs do not round-trip, first: {bad[:3]}")
    print(f"CIFs: {len(pool)} written to {os.path.relpath(CIF_ALL, ROOT)}/, "
          f"atom count and space group survive the round trip for all")


def price(pool):
    """Displacements + supercell for every crystal (symmetry only, no forces). Cached."""
    cache = os.path.join(RES, "88_phonix_cost.csv")
    done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["mp_id"])
    todo = [m for m in pool.mp_id if m not in set(done.mp_id)]
    rows, t0 = [], time.time()
    for i, mp in enumerate(todo, 1):
        try:
            # ** unpacks a dict into the new one: {"mp_id": mp, "n_atoms_prim": 8, ...}
            rows.append({"mp_id": mp, **_s08.cost_features(os.path.join(CIF_ALL, f"{mp}.cif"))})
        except Exception as exc:
            rows.append({"mp_id": mp, "error": str(exc)[:80]})
        if i % 500 == 0:
            print(f"  priced {i}/{len(todo)} ({time.time() - t0:.0f} s)")
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(cache, index=False)
    return pool.merge(done, on="mp_id", how="left")


def cost_law():
    """t = a * ndisp^b * natoms_super^c, refitted on runs of the code that will run."""
    m = pd.concat([pd.read_csv(os.path.join(DATAGEN, "results", f"01_gamma_phonon_{t}.csv"))
                   for t in MESH_ERA], ignore_index=True)
    m = m[m.status == "ok"]
    print(f"cost law, fitted on {len(m)} runs of the current kernel code:")
    return _s08.fit_cost_law(m)


def sample(pool):
    """Draw each band's fraction at random; every drawn crystal gets weight 1/fraction."""
    rng = np.random.default_rng(SEED)
    parts = []
    for band, (lo, hi, frac) in BANDS.items():
        # (lo, hi] - a crystal at exactly 1.0 belongs to band A, like step 87's "<= 1"
        b = pool[(pool.kappa_max3 > lo) & (pool.kappa_max3 <= hi)].copy()
        n = int(round(frac * len(b)))
        # rng.choice picks n DIFFERENT row positions (replace=False) at random
        pick = rng.choice(len(b), size=n, replace=False)
        s = b.iloc[np.sort(pick)].copy()
        s["band"] = band
        s["band_size"] = len(b)
        s["weight"] = len(b) / n           # exact, so the weights add back to the band size
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def make_shards(sel):
    """Longest-first packing: each crystal goes to the shard with the least work so far."""
    load = np.zeros(N_SHARDS)                             # hours placed on each shard so far
    shard = np.empty(len(sel), dtype=int)
    order = np.argsort(-sel.hours_budget.values)          # minus sign: most expensive first
    for i in order:
        k = int(np.argmin(load))                          # the lightest shard right now
        shard[i] = k
        load[k] += sel.hours_budget.values[i]
    sel["shard"] = shard
    sel["tag"] = [TAG.format(k) for k in shard]
    # shards 0,1 -> kaggle1; 2,3 -> kaggle2; 4,5 -> kaggle3 (// is whole-number division)
    sel["account"] = [ACCOUNTS[k // 2] for k in shard]
    return sel, load


def main():
    pool = load_pool()
    write_cifs(pool)
    pool = price(pool)
    if pool.n_displacements.isna().any():
        print("could not price:", pool[pool.n_displacements.isna()][["mp_id", "error"]].to_string())
    law = cost_law()
    pool["pred_s"] = _s08.predict_seconds(pool, law)
    pool["hours_budget"] = SAFETY * pool.pred_s / 3600

    print(f"\nwhole pool: {len(pool)} crystals, {pool.pred_s.sum() / 3600:.1f} GPU-h predicted "
          f"({pool.hours_budget.sum():.1f} with the x{SAFETY} margin); "
          f"median {pool.n_displacements.median():.0f} displacements, "
          f"{pool.pred_s.median():.0f} s per crystal")

    sel = sample(pool.dropna(subset=["pred_s"]))
    sel, load = make_shards(sel)

    # ---- the plan, band by band -------------------------------------------
    rows = []
    for band, g in sel.groupby("band"):
        rows.append({"band": band, "in band": int(g.band_size.iloc[0]), "drawn": len(g),
                     "weight": g.weight.iloc[0], "DFT klat<=1 in sample": int((g.klat <= 1).sum()),
                     "GPU-h predicted": g.pred_s.sum() / 3600})
    plan = pd.DataFrame(rows)
    print("\n" + plan.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"total: {len(sel)} crystals, {sel.pred_s.sum() / 3600:.1f} GPU-h predicted, "
          f"{len(load)} shards, budgeted {load.min():.2f}-{load.max():.2f} h each (x{SAFETY})")
    # the weighted sample must reproduce the pool's own DFT base rate - a check on the draw
    w_rate = np.average(sel.klat <= 1, weights=sel.weight)
    print(f"DFT klat<=1: pool {(pool.klat <= 1).mean():.3f}, weighted sample {w_rate:.3f}")

    cols = ["tag", "account", "mp_id", "formula", "band", "weight", "n_atoms", "spacegroup_no",
            "n_displacements", "supercell", "n_atoms_super", "pred_s", "klat", "kp",
            "kappa_max3", "kappa_min3", "spread3"]
    sel = sel.sort_values(["tag", "pred_s"], ascending=[True, False])
    sel[cols].to_csv(os.path.join(RES, "88_phonix_mlip_plan.csv"), index=False)

    # ---- one CIF folder per shard, where run_on_account.sh looks ----------
    for tag, g in sel.groupby("tag"):
        d = os.path.join(DATAGEN, "data", tag)
        if os.path.isdir(d):
            shutil.rmtree(d)                  # start clean, so no stale CIF survives a re-plan
        os.makedirs(d)
        for mp in g.mp_id:
            shutil.copyfile(os.path.join(CIF_ALL, f"{mp}.cif"), os.path.join(d, f"{mp}.cif"))
        print(f"  {tag}: {len(g)} CIFs, {g.pred_s.sum() / 3600:.2f} h predicted")

    # ---- figure: where the sample sits, and what it costs ------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-2, 3, 41)
    ax[0].hist(pool.kappa_max3, bins=bins, color="#cccccc", label=f"pool ({len(pool)})")
    ax[0].hist(sel.kappa_max3, bins=bins, color="#3b6ea8", label=f"MLIP sample ({len(sel)})")
    for lo, hi, _ in BANDS.values():
        ax[0].axvline(hi if np.isfinite(hi) else lo, color="k", lw=0.6, ls=":")
    ax[0].axvline(1.0, color="k", ls="--", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("step-87 prediction, max3 kappa with Poisson gamma (W/m/K)")
    ax[0].set_ylabel("crystals")
    ax[0].set_title("which crystals get MLIP gamma")
    ax[0].legend(fontsize=8)
    ax[1].scatter(pool.n_displacements, pool.pred_s, s=6, color="#cccccc", label="pool")
    ax[1].scatter(sel.n_displacements, sel.pred_s, s=8, color="#3b6ea8", label="sample")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("symmetry-distinct displacements")
    ax[1].set_ylabel("predicted GPU seconds")
    ax[1].set_title(f"cost law t = {law['a']:.2f} ndisp^{law['b']:.2f} N^{law['c']:.2f}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "88_phonix_mlip_plan.png"), dpi=150)
    print("\nwrote 88_phonix_cost.csv, 88_phonix_mlip_plan.csv, 88_phonix_mlip_plan.png")


if __name__ == "__main__":
    main()
