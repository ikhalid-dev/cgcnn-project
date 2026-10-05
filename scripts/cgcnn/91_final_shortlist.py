#!/usr/bin/env python
"""
Step 91 - the FINAL DFT shortlist: up to 15 crystals, MLIP gamma on every one.
===============================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/91_final_shortlist.py

WHAT THE USER ASKED FOR
-----------------------
"a final short list of up to 15 materials that we have run the MLIP on, and
their CIF files". The 256-row trusted queue (step 83) and the 53-row low-range
list (step 86) are too long to hand to a DFT run. The user also said: drop the
four crystals step 90 flagged (their charges do not balance).

THE RULE - four filters, applied in this order, nothing hand-picked
--------------------------------------------------------------------
    1. start from step 86's 53 crystals (<= 20 atoms, all with MLIP gamma)
    2. drop the 4 whose charges do not balance (step 90)        -> 49
    3. keep TIER 1 only: still <= 1 W/m/K under the stress test  -> 23
       (the gamma correction that only 1 benchmark crystal in 10 needs)
    4. clean chemistry before rare oxidation states (Mn+6, Fe+5,
       Au-1 ...): those three go to the back of the queue         -> 20 clean
    then take the first 15 by stress-test kappa = the most safety margin.

Everything not chosen is kept in the CSV with the reason, so the next crystal
in line is always known if DFT on one of the 15 fails.

PROOF THAT MLIP WAS RUN ON EACH ONE
-----------------------------------
Each crystal is looked up in the RAW MLIP output files
(~/Desktop/data_generation/results/01_gamma_phonon_*_recut_*.csv). The script
stops if any of the 15 is missing, did not finish, has imaginary phonon modes,
or carries a gamma different from the one the ranking used.

OUTPUTS
    results/cgcnn/91_final_shortlist.csv     all 49, with the decision for each
    results/cgcnn/91_final_shortlist.png
    dft/final_shortlist_15/*.cif + index.csv the 15 candidates
    dft/final_shortlist_15/controls/         the 4 HIGH-kappa controls (+ index.csv)
    ~/Desktop/final_shortlist_15_cifs.zip    the folder above, zipped
"""
import glob
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Composition

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
STEP86 = os.path.join(ROOT, "dft", "kappa_L_low_range")
STEP84 = os.path.join(ROOT, "dft", "kappa_L_cifs")
OUT = os.path.join(ROOT, "dft", "final_shortlist_15")
ZIP = os.path.expanduser("~/Desktop/final_shortlist_15_cifs.zip")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")

N_MAX = 15          # the user's cap
LOW = 1.0           # W/m/K - the "low range" gate


def family(formula, chem_check):
    """A plain-English structure family, read off which anions the formula has."""
    # Composition("K4HfSnI12").elements -> [K, Hf, Sn, I]; .symbol turns each into "K", "Hf", ...
    els = {e.symbol for e in Composition(formula).elements}
    if chem_check.startswith("ok: iodine cations"):
        return "iodine-cation iodofluoride"
    if "O" not in els and els & {"I"}:          # & = the elements in BOTH sets
        return "A2BI6-type iodide"
    if "O" in els and els & {"Cl", "Br", "I"}:
        return "oxyhalide"
    if "O" in els and "F" in els:
        return "oxyfluoride"
    return "oxide"


def main():
    # ---- 1. step 86's list + step 90's literature/chemistry columns --------
    # dtype={"material_id": str} stops pandas reading an id like "043233727a"
    # as a number and throwing away its leading zero
    lst = pd.read_csv(os.path.join(STEP86, "index.csv"), dtype={"material_id": str})
    lit = pd.read_csv(os.path.join(RES, "90_dft_list_literature.csv"), dtype={"material_id": str})
    d = lst.merge(lit[["material_id", "lit_status", "lit_note", "chem_check"]],
                  on="material_id", how="left")
    assert d.chem_check.notna().all(), "a crystal is missing from step 90 - rerun it"

    # ---- 2. the raw MLIP records -------------------------------------------
    # sorted(glob.glob(pattern)) = every file whose name fits the pattern, in order
    files = sorted(glob.glob(os.path.join(GAMMA_DIR, "01_gamma_phonon_oxide_recut_s*.csv"))) + \
            sorted(glob.glob(os.path.join(GAMMA_DIR, "01_gamma_phonon_halide_recut_s*.csv")))
    # pd.concat stacks the tables on top of each other; a crystal computed in two
    # shards appears twice, and drop_duplicates keeps the first
    mlip = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in files],
                     ignore_index=True).drop_duplicates("mp_id")
    keep = ["mp_id", "status", "dynamically_stable", "n_imaginary", "min_freq_THz",
            "gamma_c300", "gamma_mlip_1000K"]
    d = d.merge(mlip[keep], left_on="material_id", right_on="mp_id", how="left")

    # ---- 3. the four filters -----------------------------------------------
    d["family"] = [family(f, c) for f, c in zip(d.formula, d.chem_check)]
    flagged = d.chem_check.str.startswith("FLAG")
    tier1 = d.tier.str.startswith("1")
    rare = d.chem_check.str.startswith("rare")

    # sort key: tier-1 first, then clean before rare, then lowest stress kappa.
    # In Python False sorts before True, so "not tier1" puts tier 1 on top.
    d["_k1"], d["_k2"] = ~tier1, rare
    pool = d[~flagged].sort_values(["_k1", "_k2", "tier", "kappa_stress"]).reset_index(drop=True)
    # ~ means NOT: rows that are tier 1 AND clean; .head(15) keeps the first 15
    chosen = pool[(~pool._k1) & (~pool._k2)].head(N_MAX)

    # the decision column: one plain sentence per crystal, in the order tested
    def decide(r):
        if r.chem_check.startswith("FLAG"):
            return "dropped: charge does not balance (step 90)"
        if r.material_id in set(chosen.material_id):
            return "SHORTLIST"
        if not r.tier.startswith("1"):
            return f"not chosen: tier {r.tier[0]} (fails the stress test)"
        if r.chem_check.startswith("rare"):
            return "reserve: tier 1, rare oxidation state"
        return "reserve: tier 1, beyond the first 15"
    d["decision"] = d.apply(decide, axis=1)       # axis=1 -> run decide() once per ROW
    # rank 1..15 for the shortlist, blank for everything else
    rank = {m: k for k, m in enumerate(chosen.material_id, start=1)}
    # ids not in the dict become blank; "Int64" (capital I) = whole numbers that may be blank
    d["rank"] = d.material_id.map(rank).astype("Int64")

    # ---- 4. the MLIP gate: every chosen crystal must pass ------------------
    s = d[d.decision == "SHORTLIST"]
    problems = s[s.mp_id.isna() | (s.status != "ok") | ~s.dynamically_stable.astype(bool)
                 | (s.n_imaginary != 0) | ((s.gamma_mlip - s.gamma_c300).abs() > 1e-9)]
    if len(problems):
        raise SystemExit(f"MLIP record problem:\n{problems[['formula', 'material_id']]}")

    # ---- 5. CSV with every crystal and its decision ------------------------
    cols = ["rank", "decision", "formula", "material_id", "n_atoms", "family",
            "tier", "pos_in_tier", "gamma_mlip", "gamma_mlip_1000K", "min_freq_THz",
            "kappa_max3", "kappa_typical", "kappa_stress", "kappa_cahill", "spread3",
            "lit_status", "chem_check", "lit_note"]
    # reserves in the order the rule would use them: clean ones first, rare states after
    order = {"SHORTLIST": 0, "reserve: tier 1, beyond the first 15": 1,
             "reserve: tier 1, rare oxidation state": 2}
    d["_o"] = d.decision.map(lambda x: order.get(x, 3 if x.startswith("not") else 4))
    table = d.sort_values(["_o", "rank", "tier", "kappa_stress"])[cols]
    table.to_csv(os.path.join(RES, "91_final_shortlist.csv"), index=False)

    # ---- 6. copy the CIFs ---------------------------------------------------
    os.makedirs(os.path.join(OUT, "controls"), exist_ok=True)
    # a re-run must not leave yesterday's files behind: delete only .cif files
    # inside THIS step's own folder, never anything else
    for old in glob.glob(os.path.join(OUT, "*.cif")) + glob.glob(os.path.join(OUT, "controls", "*.cif")):
        os.remove(old)

    s = table[table.decision == "SHORTLIST"].copy()
    names = []
    for r in s.itertuples():                  # itertuples: one row at a time, columns as r.name
        # keep letters/digits, turn ( ) into _ so the file name is safe everywhere
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.formula)
        name = f"{int(r.rank):02d}__{safe}__{r.material_id}.cif"
        src = os.path.join(STEP86, lst.set_index("material_id").file[r.material_id])
        shutil.copyfile(src, os.path.join(OUT, name))
        # read the copy back and check it is the crystal the row says it is
        st = Structure.from_file(os.path.join(OUT, name))
        assert len(st) == r.n_atoms, f"{name}: {len(st)} sites, expected {r.n_atoms}"
        assert st.composition.reduced_formula == Composition(r.formula).reduced_formula, name
        names.append(name)
    s.insert(0, "file", names)                # insert(position, column name, values)
    s.drop(columns=["decision"]).to_csv(os.path.join(OUT, "index.csv"), index=False)

    # the four controls: crystals the screen calls HIGH. DFT should agree.
    i84 = pd.read_csv(os.path.join(STEP84, "index.csv"), dtype={"material_id": str})
    ctrl = i84[i84.group_label.str.startswith("C")].copy()
    cnames = []
    for k, r in enumerate(ctrl.itertuples(), start=1):
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.formula)
        name = f"ctrl_{k:02d}__{safe}__{r.material_id}.cif"
        shutil.copyfile(os.path.join(STEP84, r.file), os.path.join(OUT, "controls", name))
        cnames.append(name)
    ctrl = ctrl.assign(file=cnames)[["file", "formula", "material_id", "n_atoms",
                                     "gamma_phonon_mlip", "kappa_pred_slack_300K"]]
    ctrl.to_csv(os.path.join(OUT, "controls", "index.csv"), index=False)

    # zip the whole folder for the Desktop; arcname = the path INSIDE the zip
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(OUT, "**", "*.*"), recursive=True)):
            z.write(f, arcname=os.path.join("final_shortlist_15", os.path.relpath(f, OUT)))

    # ---- 7. print ------------------------------------------------------------
    print("=" * 100)
    print("STEP 91 - FINAL DFT SHORTLIST")
    print("=" * 100)
    print(f"\nstep 86 list {len(d)}  -> drop flagged {int(flagged.sum())} -> {int((~flagged).sum())}"
          f"  -> tier 1 {int((tier1 & ~flagged).sum())}"
          f"  -> clean chemistry {int((tier1 & ~flagged & ~rare).sum())}"
          f"  -> shortlist {len(s)}")
    print(f"\nMLIP check on the {len(s)}: status ok, dynamically stable, 0 imaginary modes, "
          f"gamma identical to the raw file - all pass")
    print(f"lowest phonon frequency among them: {s.min_freq_THz.min():.4f} THz "
          f"(tiny negatives at Gamma are numerical noise, not instability)")
    show = ["rank", "formula", "n_atoms", "family", "gamma_mlip", "kappa_max3",
            "kappa_stress", "kappa_cahill", "lit_status"]
    print("\nTHE SHORTLIST")
    print(s[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nfamilies:", s.family.value_counts().to_dict())
    below = s[s.kappa_max3 < s.kappa_cahill]
    print(f"\nscreen prediction below the crystal's own glass limit: {len(below)} of {len(s)} "
          f"({', '.join(below.formula)}) - ranked on the floored value, as in step 86")
    print("\nRESERVES, in the order to use them:")
    r = table[table.decision.str.startswith("reserve")]
    print(r[["formula", "decision", "kappa_stress", "chem_check"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(s)} CIFs (each re-read and checked) -> {os.path.relpath(OUT, ROOT)}/")
    print(f"{len(ctrl)} controls -> {os.path.relpath(OUT, ROOT)}/controls/")
    print(f"zip -> {ZIP}")

    # ---- 8. figure ------------------------------------------------------------
    colours = {"A2BI6-type iodide": "#3b6ea8", "iodine-cation iodofluoride": "#7a4fa3",
               "oxyhalide": "#2f8f5b", "oxyfluoride": "#c9803a", "oxide": "#b03a3a"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.6, 1]})

    y = np.arange(len(s))[::-1]                       # rank 1 at the top
    for fam, c in colours.items():
        m = (s.family == fam).values                  # True/False per row
        if m.any():
            a1.hlines(y[m], s.kappa_max3[m], s.kappa_stress[m], color=c, lw=3, label=fam)
    a1.scatter(s.kappa_max3, y, marker="|", color="k", s=80, zorder=3, label="screen prediction (max3)")
    a1.scatter(s.kappa_stress, y, marker=">", color="k", s=25, zorder=3, label="stress test")
    a1.scatter(s.kappa_cahill, y, marker="x", color="grey", s=20, zorder=3, label="glass limit (Cahill)")
    a1.axvline(LOW, color="k", ls="--", lw=1)
    a1.set_xscale("log")
    a1.set_yticks(y)
    a1.set_yticklabels([f"{int(k)}. {f}" for k, f in zip(s["rank"], s.formula)], fontsize=9)
    a1.set_xlabel("kappa_L at 300 K (W/m/K), log scale")
    a1.set_title("The 15: every one stays below 1 W/m/K under the stress test")
    a1.legend(fontsize=7, loc="upper right")    # top-right is empty: rows 1-4 end below 0.32

    steps = ["step 86 list", "charges balance", "tier 1 (stress <= 1)", "clean chemistry", "shortlist"]
    counts = [len(d), int((~flagged).sum()), int((tier1 & ~flagged).sum()),
              int((tier1 & ~flagged & ~rare).sum()), len(s)]
    yy = np.arange(len(steps))[::-1]
    a2.barh(yy, counts, color="#3b6ea8")
    for yi, c in zip(yy, counts):
        a2.text(c + 0.5, yi, str(c), va="center")
    a2.set_yticks(yy)
    a2.set_yticklabels(steps)
    a2.set_xlim(0, max(counts) * 1.15)
    a2.set_xlabel("crystals")
    a2.set_title("How 53 became 15")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "91_final_shortlist.png"), dpi=150)


if __name__ == "__main__":
    main()
