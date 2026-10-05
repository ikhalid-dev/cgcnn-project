#!/usr/bin/env python
"""
Step 90 - has anyone already made or measured the 53 crystals on the DFT list?
==============================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/90_dft_list_literature.py

(Step 89 is kept for the MLIP-vs-Poisson comparison that step 88 pre-registered.)

WHAT IT CHECKS, crystal by crystal, for dft/kappa_L_low_range/ (step 86)
----------------------------------------------------------------------
1. Materials Project   Is the formula in MP, and does MP mark any entry as
                       EXPERIMENTAL (theoretical = False, or an ICSD id)? An
                       ICSD id means a crystal structure was measured in a lab.
2. Literature          A web search per formula (done by hand on 2026-10-06,
                       written into the LIT table below with its source). Most
                       GNoME formulas are written in an odd reduced form, so
                       the search also used the chemist's way of writing them,
                       e.g. K4Os2Cl10O = K4[Os2OCl10].
3. Chemistry           Do the formal charges add up to zero? If oxygen has to
                       carry less than -2 and there is NO O-O bond in the
                       structure (a peroxide bond is ~1.5 A), the electrons are
                       simply missing - "oxygen holes". Plain DFT (PBE) is known
                       to make such crystals look more stable than they are.

WHAT IT DOES NOT DO
-------------------
It does not change step 86's ranking. Whether to drop or demote the flagged
crystals is the user's call.

OUTPUTS
    results/cgcnn/90_mp_status.csv            (the MP answers; cached, so MP is asked once)
    results/cgcnn/90_dft_list_literature.csv
    results/cgcnn/90_dft_list_literature.png
    dft/kappa_L_low_range/literature_check.csv  (same table, beside the CIFs;
                                                 re-run this after step 86, which
                                                 empties that folder)
"""
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition, Structure

warnings.filterwarnings("ignore")      # pymatgen warns about every CIF's rounding

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
DFT = os.path.join(ROOT, "dft", "kappa_L_low_range")
MP_CACHE = os.path.join(RES, "90_mp_status.csv")

PEROXIDE = 1.6       # A - an O-O pair closer than this is a real O-O bond

# ---------------------------------------------------------------------------
# The hand-searched literature. A Python dict maps a key to a value:
#   "formula": (status, what was found, source)
# Any formula NOT in this dict gets status "not found".
# ---------------------------------------------------------------------------
IODIDE = ("parent family reported",
          "an ordered 50/50 mix of two A2BI6 vacancy-ordered perovskites; end members "
          "such as Rb2PtI6 and K2TeI6 have been made; Cs2BI6 relatives have computed and "
          "measured kappa_L ~0.2-0.3 W/m/K",
          "https://www.crystallography.net/cod/7222602.html ; https://ar5iv.labs.arxiv.org/html/2209.08559")
IODINE_CATION = ("parent family reported",
                 "the CIF holds I-I pairs at 2.63 A = iodine cations (I2+); salts of iodine "
                 "cations with MF6- anions are known (e.g. I4(AsF6)2), this one is not",
                 "Gillespie, Passmore et al., J. Chem. Soc. Chem. Commun. 1983 (I4(AsF6)2)")
LIT = {
    "K4Os2Cl10O": ("anion reported",
                   "the [Os2OCl10]4- anion and its salts are made from HCl solution and studied "
                   "by Raman; the ammonium salt's structure is solved. The potassium salt's "
                   "crystal structure was not found",
                   "https://connectsci.au/ch/article-lookup/doi/10.1071/CH9680589 ; "
                   "https://pubs.rsc.org/en/content/articlelanding/1980/f2/f29807601103 ; "
                   "https://repository.rudn.ru/en/records/article/record/2152/"),
    "Rb4Ru2Cl10O": ("anion reported",
                    "the [Ru2OCl10]4- anion is known: K salt as a monohydrate (tetragonal), "
                    "Li salt as a decahydrate. The water-free Rb salt was not found",
                    "https://par.nsf.gov/biblio/10195473 ; "
                    "https://pubs.rsc.org/en/Content/ArticleLanding/1986/C3/C39860000483"),
    "Rb4HfPtI12": IODIDE, "Rb4ZrPtI12": IODIDE, "K4HfTeI12": IODIDE, "K4HfMnI12": IODIDE,
    "K4HfSnI12": IODIDE, "K4ZrSnI12": IODIDE, "Cs4NbCrI12": IODIDE,
    "K3Rb(TeI6)2": ("parent family reported",
                    "both end members K2TeI6 and Rb2TeI6 are made, and K/Rb tellurium-halide "
                    "solid solutions have been studied; this ordered 3:1 version was not found",
                    "https://publications.lnu.edu.ua/chemetal/ejournal27/online_abstracts/CMA0400_online_abstract.htm"),
    "Os(IF3)2": IODINE_CATION, "TaAs(IF3)4": IODINE_CATION,
    "Cs2RbAuO": ("parent family reported",
                 "Cs3AuO and Rb3AuO (gold as the Au- anion) are made; the mixed Cs/Rb one was not found",
                 "https://crystallography.io/structure/1510109 ; "
                 "https://pubs.rsc.org/en/content/articlelanding/1994/c3/c39940001045"),
    # below: nothing for the compound itself, but a note on the nearest known thing
    "KRb(CO3)2": ("not found", "NOT a peroxydicarbonate (K2C2O6 etc. are known) - this CIF has no O-O bond", ""),
    "CsK(CO3)2": ("not found", "NOT a peroxydicarbonate (Cs2C2O6 etc. are known) - this CIF has no O-O bond", ""),
    "KRb4Au3O4": ("not found", "mixed-valent Rb5Au3O2 is known, a different compound",
                  "https://www.crystallography.net/cod/1510508.html"),
    "LiHg2BrO2": ("not found", "mercury oxyhalides (Hg3O2Cl2, Hg2OCl) are known; no Li ones", ""),
    "LiHg2ClO2": ("not found", "mercury oxyhalides (Hg3O2Cl2, Hg2OCl) are known; no Li ones", ""),
    "Cs2KFeO4": ("not found", "needs Fe(V); the known caesium ferrate Cs2FeO4 is Fe(VI)", ""),
    "Cs2NaFeO4": ("not found", "needs Fe(V); the known caesium ferrate Cs2FeO4 is Fe(VI)", ""),
    "Rb2Ru(OF2)2": ("not found", "Rb2RuO2F4; the Mo analogue Rb2MoO2F4 is known", ""),
    "NaAg(NO3)2": ("not found", "NaAg(NO2)2 - the NITRITE - is known; the nitrate was not found", ""),
}


def mp_status(index):
    """Ask Materials Project once per formula; later runs read the cached CSV."""
    if os.path.exists(MP_CACHE):
        return pd.read_csv(MP_CACHE, dtype={"gnome_id": str})
    from mp_api.client import MPRester          # imported here: only a fresh query needs it
    rows = []
    with MPRester() as mpr:
        for r in index.itertuples():
            f = Composition(r.formula).reduced_formula
            docs = mpr.materials.summary.search(
                formula=f, fields=["material_id", "theoretical", "database_IDs", "energy_above_hull"])
            rows.append({"formula": r.formula, "gnome_id": r.material_id, "n_mp": len(docs),
                         "mp_ids": ";".join(str(d.material_id) for d in docs),
                         # any() is True if at least one entry passes the test
                         "any_experimental": any(d.theoretical is False for d in docs),
                         "icsd_ids": ";".join(";".join(map(str, (d.database_IDs or {}).get("icsd", [])))
                                              for d in docs).strip(";"),
                         "min_ehull": min((d.energy_above_hull for d in docs
                                           if d.energy_above_hull is not None), default=None)})
    out = pd.DataFrame(rows)
    out.to_csv(MP_CACHE, index=False)
    return out


def shortest(s, element):
    """Shortest distance between two DIFFERENT atoms of one element (periodic images included)."""
    idx = [k for k, site in enumerate(s) if site.specie.symbol == element]
    return min((s.get_distance(a, b) for a in idx for b in idx if a < b), default=np.nan)


def chemistry(formula, cif):
    """Formal-charge check plus the shortest O-O distance in the actual crystal."""
    comp = Composition(formula)
    # pymatgen's "common" list for gold is only +3; Au(+1) is just as ordinary, so add it
    common = comp.oxi_state_guesses(oxi_states_override={"Au": [1, 3]}, max_sites=-1)
    rare = comp.oxi_state_guesses(all_oxi_states=True, max_sites=-1)   # every state ever seen
    s = Structure.from_file(cif)
    oo = shortest(s, "O")
    ii = shortest(s, "I")
    states = rare[0] if rare else {}
    if ii < 2.8:
        # an I-I bond (I2 itself is 2.67 A) means iodine is a cation here, e.g. I2+ [OsF6]-;
        # a formula-only guess cannot see that and invents Os(+8)
        check = "ok: iodine cations (I-I bond in the CIF)"
    elif common:
        check = "ok"
    elif "O" in states and states["O"] > -2 and not oo < PEROXIDE:
        check = "FLAG: charge does not balance (oxygen holes, no O-O bond)"
    elif rare:
        check = "rare oxidation state: " + ", ".join(f"{e}{v:+g}" for e, v in states.items()
                                                      if e != "O")
    else:
        check = "FLAG: no charge-balanced assignment"
    return check, oo


def main():
    index = pd.read_csv(os.path.join(DFT, "index.csv"), dtype={"material_id": str})
    mp = mp_status(index)
    t = index[["tier", "pos_in_tier", "formula", "material_id", "n_atoms", "kappa_max3", "kappa_stress"]]
    t = t.merge(mp[["gnome_id", "n_mp", "mp_ids", "any_experimental", "icsd_ids"]],
                left_on="material_id", right_on="gnome_id").drop(columns="gnome_id")

    # .get(key, default) returns the default when the formula is not in LIT
    lit = [LIT.get(f, ("not found", "", "")) for f in t.formula]
    t["lit_status"] = [x[0] for x in lit]
    t["lit_note"] = [x[1] for x in lit]
    t["lit_source"] = [x[2] for x in lit]

    checks, oos = [], []
    for r in t.itertuples():
        # glob finds the CIF whose name ends in this crystal's id
        cif = glob.glob(os.path.join(DFT, f"*__{r.material_id}.cif"))[0]
        c, oo = chemistry(r.formula, cif)
        checks.append(c)
        oos.append(oo)
    t["chem_check"] = checks
    t["min_OO_A"] = np.round(oos, 3)
    t["kappa_measured_anywhere"] = False    # no search hit reported kappa for any exact compound

    t.to_csv(os.path.join(RES, "90_dft_list_literature.csv"), index=False)
    t.to_csv(os.path.join(DFT, "literature_check.csv"), index=False)

    print(f"{len(t)} crystals on the DFT list")
    print(f"  in Materials Project at all     : {(t.n_mp > 0).sum()}  (all as GNoME imports, mp-3xxxxxx)")
    print(f"  MP says EXPERIMENTAL / ICSD id  : {int(t.any_experimental.sum())}")
    print("\nliterature, by tier:")
    print(pd.crosstab(t.tier, t.lit_status, margins=True).to_string())
    print("\nchemistry flags:")
    flag = t[t.chem_check.str.startswith("FLAG")]
    print(flag[["tier", "pos_in_tier", "formula", "chem_check", "min_OO_A"]].to_string(index=False))
    print("\nrare oxidation states (real in some known compounds, unusual here):")
    rare = t[t.chem_check.str.startswith("rare")]
    print(rare[["tier", "pos_in_tier", "formula", "chem_check"]].to_string(index=False))

    # ---- figure: two panels, one bar per tier --------------------------------
    tiers = sorted(t.tier.unique())
    lit_order = ["anion reported", "parent family reported", "not found"]
    lit_col = {"anion reported": "#2e7d4f", "parent family reported": "#8fbf6a", "not found": "#c8c8c8"}
    t["chem_group"] = np.select([t.chem_check.str.startswith("FLAG"), t.chem_check.str.startswith("rare")],
                                ["charge does not balance", "rare oxidation state"], default="ok")
    chem_order = ["ok", "rare oxidation state", "charge does not balance"]
    chem_col = {"ok": "#3b6ea8", "rare oxidation state": "#e0a24a", "charge does not balance": "#b03a3a"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, col, order, colours, title in [
            (axes[0], "lit_status", lit_order, lit_col, "Literature (web search + Materials Project)"),
            (axes[1], "chem_group", chem_order, chem_col, "Chemistry check (formal charges + O-O bonds)")]:
        left = np.zeros(len(tiers))
        for k in order:
            n = np.array([((t.tier == tr) & (t[col] == k)).sum() for tr in tiers])
            ax.barh(tiers, n, left=left, color=colours[k], label=k)
            for y, (l, v) in enumerate(zip(left, n)):
                if v:
                    ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=8)
            left += n
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("crystals")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].invert_yaxis()                          # tier 1 on top
    fig.suptitle("The 53-crystal DFT list: none is experimentally known; none has a measured kappa",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "90_dft_list_literature.png"), dpi=150)


if __name__ == "__main__":
    main()
