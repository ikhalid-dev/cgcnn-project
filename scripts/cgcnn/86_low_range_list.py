#!/usr/bin/env python
"""
Step 86 - the DFT list for kappa_L in the LOW RANGE, not for the lowest number.
===============================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/86_low_range_list.py

WHAT THE USER ASKED FOR
-----------------------
The DFT run computes kappa_L. The predicted value does not have to come out
exactly; the crystal has to come out LOW (kappa_L <= 1 W/m/K at 300 K). The
MLIP gamma was computed for exactly this: to make the screen pick crystals that
really are low.

So the question for each crystal changes from "how low is the prediction?" to
"how SAFELY is it below 1 - would it still be below 1 if the inputs are off?"

WHICH WAY THE INPUTS ARE OFF
----------------------------
moduli   max3 = the HIGHEST of the three models' kappa, so the most pessimistic
         model already sets the number. Step 82: DFT moduli usually give a LOWER
         kappa than max3 (median DFT/pred 0.73). Errors here mostly help.

gamma    The 45-material benchmark (~/Desktop/data_generation, script 02)
         compares MLIP gamma with the gamma PINK's Slack formula was built on.
         On the 38 stable, clean materials MLIP gamma is HIGHER in 74% of
         cases (median +0.14). A gamma that is too high makes kappa too LOW -
         the dangerous direction for a low-range list. Worst cases are soft,
         anharmonic crystals (AgCl 4.5 vs 1.9), the same kind as our candidates.

THE TWO RE-SCORES
-----------------
In the Slack formula kappa contains exp(-gamma), so a different gamma just
rescales kappa:      kappa(gamma_new) = kappa(gamma_old) * exp(gamma_old - gamma_new)

    typical    gamma x (median of reference/MLIP on the benchmark)   ~ x0.90
    stress     gamma x (10th percentile of reference/MLIP)            ~ x0.68
               = MLIP overshooting as badly as 1 benchmark material in 10

Both are then raised to the crystal's amorphous floor if they fall below it,
because a crystal cannot be relied on to conduct heat worse than its own glass.

TIERS (only cells of <= 20 atoms - the DFT budget)
    1 robust low   still <= 1.0 under the STRESS re-score
    2 likely low   <= 1.0 under the typical re-score, not under stress
    3 at risk      above 1.0 once the typical overshoot is removed - its low
                   prediction leans on a gamma that is probably too high

LIMIT, SAID UP FRONT: the benchmark crystals are simple (2-16 atoms, gamma
mostly < 2). Our candidates reach gamma 3.4. Scaling the correction with gamma
is an assumption - the AgCl case suggests real overshoots at high gamma can be
larger, which would push more crystals toward tier 3, not fewer.

OUTPUTS
    results/cgcnn/86_low_range_list.csv
    results/cgcnn/86_low_range_list.png
    dft/kappa_L_low_range/*.cif + index.csv    (copied from step 84's verified files)
"""
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
BENCH = os.path.expanduser("~/Desktop/data_generation/results/02_gamma_comparison.csv")
STEP84 = os.path.join(ROOT, "dft", "kappa_L_cifs")
OUT = os.path.join(ROOT, "dft", "kappa_L_low_range")

LOW = 1.0          # W/m/K at 300 K - "the low range"
MAX_ATOMS = 20     # the DFT budget the user set


def gamma_corrections():
    """How much smaller the reference gamma is than MLIP gamma, from the benchmark."""
    b = pd.read_csv(BENCH)
    # stable = no imaginary modes; clean = the structure is the right polymorph
    b = b[b.stable & b.clean].dropna(subset=["gamma_paper", "gamma_mlip"])
    ratio = b.gamma_paper / b.gamma_mlip         # below 1 -> MLIP gamma was too high
    typical = ratio.median()
    stress = ratio.quantile(0.10)                # 9 benchmark materials in 10 overshoot less than this
    print(f"gamma benchmark: {len(b)} stable+clean materials, MLIP higher in "
          f"{(ratio < 1).mean():.0%} of them")
    print(f"  reference/MLIP ratio: typical (median) {typical:.3f}, stress (10th pct) {stress:.3f}")
    return typical, stress


def main():
    typical, stress = gamma_corrections()

    q = pd.read_csv(os.path.join(RES, "83_trusted_dft_queue.csv"), dtype={"material_id": str})
    idx84 = pd.read_csv(os.path.join(STEP84, "index.csv"), dtype={"material_id": str})

    # step 84 already holds every crystal <= 20 atoms, verified file by file;
    # .isin() keeps queue rows whose id appears in that index
    cand = q[q.material_id.isin(idx84.material_id) & ~q.group_label.str.startswith("C")].copy()
    cand["n_atoms"] = cand["Number of Atoms"].astype(int)
    assert (cand.n_atoms <= MAX_ATOMS).all()
    print(f"\ncandidates with <= {MAX_ATOMS} atoms: {len(cand)}")

    g = cand.gamma_mlip
    # exp(g - g*factor) = exp(g * (1 - factor)): how much kappa rises when gamma shrinks
    cand["kappa_typical"] = cand.kappa_max3 * np.exp(g * (1 - typical))
    cand["kappa_stress"] = cand.kappa_max3 * np.exp(g * (1 - stress))
    # np.maximum compares two columns row by row and keeps the larger value
    cand["kappa_typical"] = np.maximum(cand.kappa_typical, cand.kappa_cahill)
    cand["kappa_stress"] = np.maximum(cand.kappa_stress, cand.kappa_cahill)

    # np.select: the first condition that is True picks the label
    cand["tier"] = np.select([cand.kappa_stress <= LOW, cand.kappa_typical <= LOW],
                             ["1 robust low", "2 likely low"], default="3 at risk")
    # most safety margin first: lowest stress-test kappa at the top of each tier
    cand = cand.sort_values(["tier", "kappa_stress"]).reset_index(drop=True)
    cand["pos_in_tier"] = cand.groupby("tier").cumcount() + 1

    print("\ntier counts:")
    print(cand.tier.value_counts().sort_index().to_string())
    print("\nhow the step-83 groups land:")
    print(pd.crosstab(cand.group_label, cand.tier, margins=True).to_string())

    cols = ["tier", "pos_in_tier", "formula", "material_id", "n_atoms", "gamma_mlip",
            "kappa_max3", "kappa_typical", "kappa_stress", "kappa_cahill", "spread3", "group_label"]
    table = cand[cols]
    table.to_csv(os.path.join(RES, "86_low_range_list.csv"), index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- CIFs: copy step 84's verified files under tier names ----------------
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    src = dict(zip(idx84.material_id, idx84.file))     # material_id -> step-84 filename
    names = []
    for r in cand.itertuples():
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.formula)
        name = f"t{r.tier[0]}_{r.pos_in_tier:02d}__{safe}__{r.material_id}.cif"
        shutil.copyfile(os.path.join(STEP84, src[r.material_id]), os.path.join(OUT, name))
        names.append(name)
    # the four negative controls ride along unchanged - DFT should find them HIGH
    ctrl = idx84[idx84.group_label.str.startswith("C")]
    for k, r in enumerate(ctrl.itertuples(), start=1):
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.formula)
        shutil.copyfile(os.path.join(STEP84, r.file),
                        os.path.join(OUT, f"ctrl_{k:02d}__{safe}__{r.material_id}.cif"))
    index = table.assign(file=names)
    index.to_csv(os.path.join(OUT, "index.csv"), index=False)
    print(f"\nwrote {len(index)} candidate CIFs + {len(ctrl)} controls + index.csv "
          f"to {os.path.relpath(OUT, ROOT)}/")

    # ---- figure: each crystal's prediction, typical and stress values --------
    colours = {"1 robust low": "#3b6ea8", "2 likely low": "#c9803a", "3 at risk": "#b03a3a"}
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(cand) + 1.5))
    y = np.arange(len(cand))[::-1]                     # first row at the top
    ax.hlines(y, cand.kappa_max3, cand.kappa_stress, color=[colours[t] for t in cand.tier], lw=2)
    ax.scatter(cand.kappa_max3, y, marker="|", color="k", s=60, label="screen prediction (max3)", zorder=3)
    ax.scatter(cand.kappa_typical, y, marker="o", color="k", s=14, label="typical gamma correction", zorder=3)
    ax.scatter(cand.kappa_stress, y, marker=">", color="k", s=18, label="stress test (1-in-10 gamma)", zorder=3)
    ax.axvline(LOW, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"t{t[0]} {f}" for t, f in zip(cand.tier, cand.formula)], fontsize=7)
    ax.set_xlabel("kappa_L at 300 K (W/m/K), log scale")
    ax.set_title(f"<= {MAX_ATOMS}-atom candidates: does each stay below {LOW} W/m/K?")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "86_low_range_list.png"), dpi=150)


if __name__ == "__main__":
    main()
