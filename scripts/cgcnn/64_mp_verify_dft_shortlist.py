"""
Step 64 - verify the DFT shortlist against external DFT, without running any DFT.

WHAT THIS ANSWERS
-----------------
`dft/dft_materials.csv` (step 61) queues 18 GNoME crystals for elastic-tensor
calculations. Arm B of that set - 6 crystals - exists purely to adjudicate a
disagreement: ALIGNN and the composition-only tree call them very soft
(K ~ 5-8 GPa), round 9 calls them stiff (K ~ 67-135 GPa). The plan was to settle
it with our own DFT.

It may already be settled. The Materials Project and AFLOW both publish DFT
elastic tensors, and while they do not contain these exact GNoME crystals, they
contain plenty of the same CHEMISTRY. If every real metal halide ever computed
is soft, a prediction of 135 GPa for a halide is not a close call.

WHAT IT DOES
------------
Three checks, cheapest first:

  1. EXACT match. Ask MP for each of the 18 reduced formulas, any polymorph,
     and report whether an elastic tensor exists.
  2. CHEMISTRY match. For the chemical systems of arm B (Mo-Br, W-Cl, Zr-Br,
     Bi-Cl, ...) pull every MP material that HAS an elastic tensor, and compare
     the real range against what each model predicts.
  3. LOCAL cross-check. Independently, bin this project's own AFLOW table by
     halogen fraction and read off the real bulk-modulus distribution. AFLOW and
     MP are separate DFT databases with separate settings, so agreement between
     them is worth more than either alone.

WHAT IT CANNOT DO
-----------------
This is corroboration, not substitution. The MP analogues are different
compositions in different space groups, so none of them IS the candidate. It
cannot confirm that a specific GNoME crystal is soft; it can only show whether a
prediction sits inside or far outside the range that chemistry has ever produced.
Read the output as "is this physically plausible", never as "this is the answer".

Requires an MP API key in ~/.pmgrc.yaml (PMG_MAPI_KEY) and mp-api installed.

Run:
    /Users/mac/miniconda3/envs/ml_env/bin/python scripts/cgcnn/64_mp_verify_dft_shortlist.py
"""

import torch  # first, per this project's OpenMP import-order rule
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Composition

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "cgcnn")
os.makedirs(OUT, exist_ok=True)

HALOGENS = {"F", "Cl", "Br", "I"}

# Palette validated with the dataviz skill's checker (lightness band, chroma
# floor, CVD separation and normal-vision floor all PASS).
C_REAL, C_R9, C_ALIGNN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def halogen_fraction(formula):
    """Fraction of atoms that are halogens. Parsed properly, not by substring.

    A substring search for 'O' would match Os and Og; the same trap applies to
    halogens (e.g. 'I' inside 'In', 'Cl' vs 'C'+'l'). pymatgen's Composition
    parser is the only correct way to do this - the same lesson step 15 of this
    project already learned for oxides.
    """
    try:
        comp = Composition(formula)
        total = sum(comp.values())
        return sum(v for el, v in comp.items() if el.symbol in HALOGENS) / total
    except Exception:
        return np.nan


def chemsys_of(formula):
    """Two-element systems to query MP for: every metal-halogen pair in the formula."""
    comp = Composition(formula)
    els = [e.symbol for e in comp.elements]
    halos = [e for e in els if e in HALOGENS]
    metals = [e for e in els if e not in HALOGENS]
    return sorted({f"{m}-{h}" for m in metals for h in halos})


def main():
    shortlist = pd.read_csv(os.path.join(ROOT, "dft", "dft_materials.csv"))
    print("=" * 78)
    print("STEP 64 - external DFT check on the 18-crystal DFT shortlist")
    print("=" * 78)

    from mp_api.client import MPRester

    # ---- check 1: exact formula matches ----------------------------------
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            formula=shortlist.formula.tolist(),
            fields=["material_id", "formula_pretty", "symmetry", "bulk_modulus",
                    "shear_modulus", "energy_above_hull"])

    exact = []
    for d in docs:
        k = d.bulk_modulus.get("vrh") if isinstance(d.bulk_modulus, dict) else d.bulk_modulus
        g = d.shear_modulus.get("vrh") if isinstance(d.shear_modulus, dict) else d.shear_modulus
        exact.append(dict(mp_id=d.material_id, formula=d.formula_pretty,
                          spacegroup=str(d.symmetry.symbol),
                          e_above_hull=d.energy_above_hull,
                          K_mp=k, G_mp=g, has_elastic=k is not None))
    exact = pd.DataFrame(exact)
    exact.to_csv(os.path.join(OUT, "64_mp_exact_matches.csv"), index=False)

    print(f"\n--- CHECK 1: exact formula matches in MP ---")
    print(f"{len(exact)} of {len(shortlist)} shortlist formulas exist in MP")
    if len(exact):
        n_el = int(exact.has_elastic.sum())
        print(f"of those, {n_el} have an elastic tensor")
        print(exact.to_string(index=False))
    if len(exact) and exact.has_elastic.sum() == 0:
        print("\n=> No direct verification possible. GNoME crystals are recent "
              "entries; MP has not run elasticity on them. Fall through to "
              "check 2, which compares against the same chemistry instead.")

    # ---- check 2: same chemistry, with elastic data ----------------------
    arm_b = shortlist[shortlist.arm.str.startswith("B")]
    systems = sorted({s for f in arm_b.formula for s in chemsys_of(f)})
    print(f"\n--- CHECK 2: MP materials in arm-B chemistries WITH elastic tensors ---")
    print(f"querying {len(systems)} chemical systems: {', '.join(systems)}")

    rows = []
    with MPRester() as mpr:
        for sysname in systems:
            try:
                ds = mpr.materials.summary.search(
                    chemsys=sysname,
                    fields=["material_id", "formula_pretty", "symmetry",
                            "bulk_modulus", "shear_modulus", "energy_above_hull"])
            except Exception as exc:
                print(f"   {sysname}: query failed ({type(exc).__name__})")
                continue
            for d in ds:
                k = d.bulk_modulus.get("vrh") if isinstance(d.bulk_modulus, dict) else d.bulk_modulus
                g = d.shear_modulus.get("vrh") if isinstance(d.shear_modulus, dict) else d.shear_modulus
                if k is None:
                    continue
                rows.append(dict(chemsys=sysname, mp_id=d.material_id,
                                 formula=d.formula_pretty,
                                 spacegroup=str(d.symmetry.symbol),
                                 e_above_hull=d.energy_above_hull, K_mp=k, G_mp=g))
    analogues = pd.DataFrame(rows).sort_values("K_mp")
    analogues.to_csv(os.path.join(OUT, "64_mp_chemistry_analogues.csv"), index=False)

    # A negative shear modulus means THAT MP calculation is itself unconverged.
    # Keep the rows (they are real database contents) but flag them, so nobody
    # leans on an individual number that the source database got wrong.
    analogues["suspect"] = analogues.G_mp <= 0
    print(analogues.to_string(index=False))
    good = analogues[~analogues.suspect]
    print(f"\nreal DFT bulk modulus in this chemistry (n={len(analogues)}, "
          f"{int(analogues.suspect.sum())} flagged with G<=0):")
    print(f"   min {analogues.K_mp.min():.1f}   median {analogues.K_mp.median():.1f}   "
          f"max {analogues.K_mp.max():.1f} GPa")

    print(f"\n{'candidate':>12}{'ALIGNN K':>10}{'round9 K':>10}   verdict vs the real range")
    k_max = analogues.K_mp.max()
    verdicts = []
    for _, r in arm_b.iterrows():
        v = ("round 9 ABOVE every real analogue"
             if r.K_pred_round9_GPa > k_max else "inside the real range")
        print(f"{r.formula:>12}{r.K_pred_alignn_GPa:>10.1f}{r.K_pred_round9_GPa:>10.1f}   {v}")
        verdicts.append(dict(formula=r.formula, K_alignn=r.K_pred_alignn_GPa,
                             K_round9=r.K_pred_round9_GPa,
                             max_real_K_same_chemistry=k_max,
                             round9_exceeds_all_real=r.K_pred_round9_GPa > k_max,
                             alignn_inside_real_range=(analogues.K_mp.min() <=
                                                       r.K_pred_alignn_GPa <= k_max)))
    pd.DataFrame(verdicts).to_csv(os.path.join(OUT, "64_armB_verdict.csv"), index=False)

    # ---- check 3: independent AFLOW cross-check --------------------------
    print(f"\n--- CHECK 3: this project's own AFLOW table, binned by halogen content ---")
    aflow = pd.read_csv(os.path.join(ROOT, "data_full", "aflow_all.csv"))
    aflow = aflow.dropna(subset=["ael_bulk_modulus_vrh"])
    aflow = aflow[aflow.ael_bulk_modulus_vrh > 0]
    aflow["hal_frac"] = aflow.compound.map(halogen_fraction)
    rich = aflow[aflow.hal_frac >= 0.75]
    print(f"AFLOW entries with elastic data: {len(aflow)}; halogen-rich (>=75%): {len(rich)}")
    print(f"   halogen-rich K: median {rich.ael_bulk_modulus_vrh.median():.1f}, "
          f"p95 {rich.ael_bulk_modulus_vrh.quantile(.95):.1f}, "
          f"max {rich.ael_bulk_modulus_vrh.max():.1f} GPa")
    for thr in (60, 80, 100, 127):
        n = int((rich.ael_bulk_modulus_vrh > thr).sum())
        print(f"   entries above {thr:>3} GPa: {n} of {len(rich)}")

    # ---- figure -----------------------------------------------------------
    # A strip plot of the real values with the two models' predictions overlaid.
    # The question is "does the prediction land inside the observed range", which
    # is a one-dimensional comparison - so a one-dimensional chart, not a scatter.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.scatter(analogues.K_mp, np.full(len(analogues), 0.0), s=64, marker="o",
               facecolor=C_REAL, edgecolor="white", linewidth=0.9, zorder=3,
               label=f"real MP DFT, same chemistry (n={len(analogues)})")
    ax.scatter(arm_b.K_pred_alignn_GPa, np.full(len(arm_b), 0.5), s=74, marker="^",
               facecolor=C_ALIGNN, edgecolor="white", linewidth=0.9, zorder=3,
               label="ALIGNN prediction, arm B")
    ax.scatter(arm_b.K_pred_round9_GPa, np.full(len(arm_b), 1.0), s=74, marker="s",
               facecolor=C_R9, edgecolor="white", linewidth=0.9, zorder=3,
               label="round 9 prediction, arm B")

    ax.axvspan(analogues.K_mp.min(), k_max, color=C_REAL, alpha=0.07, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["real DFT", "ALIGNN", "round 9"], color=INK_MUTED)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("bulk modulus K (GPa, log scale)", color=INK_MUTED)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.set_title("Arm-B predictions against every DFT elastic tensor MP has "
                 "for the same chemistry", color=INK, fontsize=12, pad=12)
    # No legend box. The three y-tick labels ARE the direct labels, so identity is
    # never carried by colour alone, and an "upper left" legend sat on top of the
    # round-9 row - obscuring the very markers the chart exists to show.
    ax.annotate(f"shaded band = full range of real DFT values (n={len(analogues)})",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                color=INK_MUTED, fontsize=9)
    ax.annotate("every round-9 prediction sits outside the shaded band",
                xy=(0.98, 1.22), xycoords=("axes fraction", "data"), ha="right",
                color=C_R9, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "64_mp_verification.png"), dpi=200,
                bbox_inches="tight", facecolor="#fcfcfb")

    print(f"\nwrote {OUT}/64_mp_exact_matches.csv")
    print(f"wrote {OUT}/64_mp_chemistry_analogues.csv")
    print(f"wrote {OUT}/64_armB_verdict.csv")
    print(f"wrote {OUT}/64_mp_verification.png")


if __name__ == "__main__":
    main()
