#!/usr/bin/env python
"""
Step 96 - the double-perovskite family: our predictions against the papers
==========================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/96_double_perovskite_check.py

THE QUESTION
------------
Most of the final 15 (step 91) are A2BX6-type iodides: "vacancy-ordered double
perovskites" such as Cs2SnI6, where A = K/Rb/Cs, B = one metal, X = a halogen.
None of the 15 has been measured, but their PARENT compounds have published
lattice thermal conductivities:
  - PhoNIX (phonon DFT, 3-phonon) has 26 of them;
  - Cao et al. 2026 computed 18 Cs2BX6 with a higher-level DFT method;
  - one, Cs2SnI6, has been MEASURED.
Where we already have a prediction that was made WITHOUT seeing the crystal,
how close are our models to those published numbers?

HOW
---
This step re-runs no model. It only collects predictions that already exist:
1. Find every A2BX6 crystal in PhoNIX (step 85's PhoNIX-matbench match table,
   which also says whether matbench holds the crystal and in which split).
2. Physics-chain kappa (moduli -> Poisson gamma -> kappa_cal), HELD OUT only:
     - crystals NOT in matbench: step 87 scored them (2,520 PhoNIX crystals);
     - crystals in matbench's TEST split: step 82's test predictions.
   Crystals in matbench's training split have no held-out prediction: the
   networks saw their moduli. Their cells are left empty, not filled in.
3. Direct ALIGNN (structure -> kappa_L in one step): the 5-fold out-of-fold
   predictions of CoherentHeat step 25, where each crystal was held out once.
4. The published values, typed in LITERATURE below with their source.

OUTPUTS
-------
    results/cgcnn/96_double_perovskite_check.csv   one row per compound
    results/cgcnn/96_double_perovskite_check.png   predicted vs PhoNIX, and
                                                    the two compounds with papers
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # draw straight to a file, no window
import matplotlib.pyplot as plt
from pymatgen.core import Composition    # reads a formula like "Cs2SnI6"

# ----------------------------------------------------------------------------
# CONFIG - every input file and every typed number lives here
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
COH = os.path.expanduser("~/Desktop/a new project")     # the CoherentHeat project

PHONIX_MATCH = os.path.join(RESULTS, "85_phonix_matbench_match.csv")   # 6,641 PhoNIX crystals
CHAIN_87 = os.path.join(RESULTS, "87_phonix_poisson_predictions.csv")  # not in matbench
CHAIN_82 = os.path.join(RESULTS, "82_test_predictions.csv")            # matbench test split
DIRECT_OOF = os.path.join(COH, "results", "25_oof_klat_predictions.csv")  # log10 kappa_L
SHORTLIST = os.path.join(RESULTS, "91_final_shortlist.csv")

OUT_CSV = os.path.join(RESULTS, "96_double_perovskite_check.csv")
OUT_PNG = os.path.join(RESULTS, "96_double_perovskite_check.png")

A_SITE = {"K", "Rb", "Cs"}               # the large cation
X_SITE = {"Cl", "Br", "I"}               # the halogen
LOW = 1.0                                # "low" = kappa_L <= 1 W/m/K, as everywhere else

# Cao et al. 2026 computed all 18 Cs2BX6 with these B and X, and reports every
# one below 1 W/m/K at 300 K (abstract and p.4). Only two values are stated in
# the text; the rest are in a bar chart, which we do not read numbers off.
CAO_B = ["Zr", "Pd", "Sn", "Te", "Hf", "Pt"]
CAO_X = ["Cl", "Br", "I"]

# Published kappa_L at 300 K, W/m/K. One row per number, each with its source.
CAO = "Cao, Wang, Xia, He, arXiv:2602.06501 (2026), p.4"
LITERATURE = [
    # formula,    kind,       value, source
    ("Cs2SnI6",  "measured",  0.29, "Bhui et al., Chem. Mater. 34, 3301 (2022); quoted in " + CAO),
    ("Cs2SnI6",  "DFT_cao",   0.28, CAO + " (3+4-phonon, renormalised, + coherent term)"),
    ("Cs2SnBr6", "DFT_cao",   0.47, CAO + " (3+4-phonon, renormalised, + coherent term)"),
    ("Cs2SnBr6", "DFT_other", 0.43, "Cheng et al., Phys. Rev. B 109, 054305 (2024); quoted in " + CAO),
]

CHAIN_MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]   # newbase = the descriptor XGBoost


def a2bx6_parts(formula):
    """Return (A, B, X) if the formula is A2BX6 with A in A_SITE, X in X_SITE; else None.

    Composition(...).reduced_composition turns e.g. "Cs4Sn2I12" into Cs2SnI6,
    so a crystal written with a bigger cell still counts. It behaves like a
    dictionary {element: amount}.
    """
    try:
        comp = Composition(formula).reduced_composition
    except Exception:                    # an unreadable formula is simply not a match
        return None
    amounts = {str(el): n for el, n in comp.items()}   # {"Cs": 2.0, "Sn": 1.0, "I": 6.0}
    a = [el for el in amounts if el in A_SITE]
    x = [el for el in amounts if el in X_SITE]
    b = [el for el in amounts if el not in A_SITE and el not in X_SITE]
    if len(a) == 1 and len(b) == 1 and len(x) == 1 \
            and amounts[a[0]] == 2 and amounts[b[0]] == 1 and amounts[x[0]] == 6:
        return a[0], b[0], x[0]
    return None


def main():
    # ---- 1. the A2BX6 crystals in PhoNIX ------------------------------------
    phonix = pd.read_csv(PHONIX_MATCH)
    parts = phonix.formula.map(a2bx6_parts)          # None for every other crystal
    fam = phonix[parts.notna()].copy()               # keep only the family
    fam["formula"] = fam.formula.map(lambda f: Composition(f).reduced_formula)
    if fam.mp_id.duplicated().any():
        raise SystemExit("a PhoNIX crystal appears twice - stop, step 85 should be one row each")
    print(f"A2BX6 crystals in PhoNIX: {len(fam)}")

    # ---- 2a. physics chain, crystals NOT in matbench (step 87) ---------------
    c87 = pd.read_csv(CHAIN_87)
    c87 = c87[["mp_id"] + CHAIN_MODELS + ["kappa_max3", "klat"]]
    c87 = c87.rename(columns={m: f"chain_{m}" for m in CHAIN_MODELS} | {"kappa_max3": "chain_max3",
                                                                          "klat": "klat_87"})
    c87["chain_source"] = "step 87 (not in matbench)"

    # ---- 2b. physics chain, crystals in matbench's TEST split (step 82) ------
    c82 = pd.read_csv(CHAIN_82)
    keep = {f"{m}_full_chain": f"chain_{m}" for m in CHAIN_MODELS}
    keep |= {"max3_full_chain": "chain_max3", "kappa_DFT": "chain_on_dft_moduli"}
    c82 = c82[["mb_id"] + list(keep)].rename(columns=keep)
    c82["chain_source"] = "step 82 (matbench test split)"

    # Join each source on its own key. "how='left'" keeps every family row,
    # with empty cells where that source has nothing.
    from_87 = fam[["mp_id"]].merge(c87, on="mp_id", how="inner")
    from_82 = fam[["mp_id", "mb_id"]].merge(c82, on="mb_id", how="inner").drop(columns="mb_id")
    both = set(from_87.mp_id) & set(from_82.mp_id)
    if both:
        raise SystemExit(f"{sorted(both)} are in both chain sources - they should be disjoint")
    chain = pd.concat([from_87, from_82], ignore_index=True)

    # A safety check: step 87 stored PhoNIX's klat next to its predictions.
    # It must equal the klat in step 85's table, or the rows are mismatched.
    chk = from_87.merge(fam[["mp_id", "klat"]], on="mp_id")
    if (chk.klat_87 - chk.klat).abs().max() > 1e-9:
        raise SystemExit("step 87's klat disagrees with step 85's - rows are mismatched")
    chain = chain.drop(columns="klat_87")

    # ---- 3. direct ALIGNN, out-of-fold (CoherentHeat step 25) ----------------
    oof = pd.read_csv(DIRECT_OOF)                     # columns: mp_id, fold, true_log_klat, oof_pred
    oof = fam[["mp_id", "klat"]].merge(oof, on="mp_id", how="inner")
    # its stored truth is log10(klat); check it against PhoNIX before trusting the join.
    # The two files round klat differently (step 85 keeps 6 decimals), so they
    # differ by ~1e-6 in log10; 1e-4 (0.02%) still catches any real mismatch.
    if (oof.true_log_klat - np.log10(oof.klat)).abs().max() > 1e-4:
        raise SystemExit("step 25's stored truth disagrees with PhoNIX klat - rows are mismatched")
    oof["direct_oof"] = 10 ** oof.oof_pred            # back from log10 to W/m/K
    oof = oof[["mp_id", "direct_oof"]]

    # ---- put one table together ----------------------------------------------
    table = fam[["formula", "mp_id", "n_atoms", "split", "klat", "kp", "kc"]].rename(
        columns={"split": "matbench_split", "klat": "phonix_klat",
                 "kp": "phonix_kp", "kc": "phonix_kc"})
    table = table.merge(chain, on="mp_id", how="left").merge(oof, on="mp_id", how="left")
    # A crystal with no chain prediction is missing for one of two reasons:
    #   in matbench (train/val) -> its moduli were training data, nothing held out exists
    #   not in matbench         -> step 87 skipped it for having more than 20 atoms
    # np.where(condition, a, b) picks a where the condition is True, b elsewhere.
    why_missing = np.where(table.matbench_split == "not in mb",
                           "not scored: step 87 kept <= 20 atoms",
                           "none held out: moduli in matbench training")
    table["chain_source"] = table.chain_source.fillna(pd.Series(why_missing, index=table.index))

    # Cao's 18: is the compound one of them (so Cao says kappa_L < 1)?
    cao_set = {f"Cs2{b}{x}6" for b in CAO_B for x in CAO_X}
    table["in_cao_18"] = table.formula.isin(cao_set)

    # the typed literature values, one column per kind
    lit = pd.DataFrame(LITERATURE, columns=["formula", "kind", "value", "source"])
    wide = lit.pivot(index="formula", columns="kind", values="value").add_prefix("lit_")
    table = table.merge(wide, left_on="formula", right_index=True, how="left")

    # Cao compounds that PhoNIX does not have: list them too, empty, so the
    # table shows all 18 and what is missing.
    missing = sorted(cao_set - set(table.formula))
    extra = pd.DataFrame({"formula": missing, "in_cao_18": True,
                          "chain_source": "not in PhoNIX"})
    table = pd.concat([table, extra], ignore_index=True)
    table = table.sort_values(["phonix_klat", "formula"], na_position="last")
    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(table)} rows)")

    # ---- the printed summary ------------------------------------------------
    have = table[table.chain_max3.notna()]
    print(f"\nPhoNIX DFT: {(table.phonix_klat <= LOW).sum()} of "
          f"{table.phonix_klat.notna().sum()} A2BX6 crystals have kappa_L <= {LOW}")
    print(f"Physics chain, held out: {len(have)} crystals "
          f"({(have.chain_source.str.startswith('step 87')).sum()} not in matbench, "
          f"{(have.chain_source.str.startswith('step 82')).sum()} matbench test)")
    for col in [f"chain_{m}" for m in CHAIN_MODELS] + ["chain_max3"]:
        r = have.phonix_klat / have[col]               # > 1 means DFT is HIGHER than predicted
        print(f"  {col:16s} calls <= 1: {(have[col] <= LOW).sum():2d}/{len(have)};  "
              f"median DFT/predicted {r.median():.2f}  (range {r.min():.2f}-{r.max():.2f})")
    d = table[table.direct_oof.notna()]
    err = (np.log10(d.direct_oof) - np.log10(d.phonix_klat)).abs()
    print(f"Direct ALIGNN, out-of-fold: {len(d)} crystals; calls <= 1: "
          f"{(d.direct_oof <= LOW).sum()}/{len(d)}; MAE log10 {err.mean():.3f}; "
          f"median DFT/predicted {(d.phonix_klat / d.direct_oof).median():.2f}")

    print("\nCompounds with a published value:")
    cols = ["formula", "phonix_klat", "lit_measured", "lit_DFT_cao", "lit_DFT_other",
            "chain_ALIGNN", "chain_CGCNN-ens", "chain_newbase", "direct_oof", "chain_source"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(table[table.lit_DFT_cao.notna()][cols].to_string(index=False))

    sl = pd.read_csv(SHORTLIST)
    n_fam = (sl.decision.eq("SHORTLIST") & sl.family.str.startswith("A2BI6")).sum()
    print(f"\n(For context: {n_fam} of the final {sl.decision.eq('SHORTLIST').sum()} "
          f"are labelled 'A2BI6-type iodide' in step 91.)")

    # ---- the figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: predicted against PhoNIX, log-log
    lim = [0.05, 3]
    ax1.plot(lim, lim, color="0.5", lw=1, ls="--", label="perfect")
    ax1.fill_between(lim, [v / 2 for v in lim], [v * 2 for v in lim],
                     color="0.9", label="within x2")
    ax1.axhline(LOW, color="tab:red", lw=0.8, ls=":")
    ax1.axvline(LOW, color="tab:red", lw=0.8, ls=":")
    styles = {"chain_ALIGNN": ("tab:green", "o", "chain, ALIGNN"),
              "chain_CGCNN-ens": ("tab:blue", "s", "chain, CGCNN ensemble"),
              "chain_newbase": ("tab:orange", "^", "chain, descriptor XGBoost"),
              "direct_oof": ("tab:purple", "D", "direct ALIGNN (out-of-fold)")}
    for col, (c, mk, lab) in styles.items():
        s = table[table[col].notna()]
        ax1.scatter(s.phonix_klat, s[col], color=c, marker=mk, s=30, alpha=0.8,
                    label=f"{lab}, n={len(s)}")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("PhoNIX DFT $\\kappa_L$ (W/m/K, 300 K)")
    ax1.set_ylabel("predicted $\\kappa_L$ (W/m/K)")
    ax1.set_title("A  A$_2$BX$_6$ crystals, held-out predictions only", loc="left", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper left")

    # Panel B: the two compounds with papers, every value on one line each
    rows = ["Cs2SnI6", "Cs2SnBr6"]
    marks = [("phonix_klat", "black", "o", "PhoNIX DFT (3-phonon)"),
             ("lit_DFT_cao", "tab:red", "*", "Cao 2026 DFT (3+4-phonon)"),
             ("lit_DFT_other", "tab:pink", "P", "Cheng 2024 DFT"),
             ("lit_measured", "tab:brown", "X", "measured (Bhui 2022)"),
             ("chain_ALIGNN", "tab:green", "o", "chain, ALIGNN"),
             ("chain_CGCNN-ens", "tab:blue", "s", "chain, CGCNN ensemble"),
             ("chain_newbase", "tab:orange", "^", "chain, descriptor XGBoost"),
             ("direct_oof", "tab:purple", "D", "direct ALIGNN")]
    seen = set()                         # each legend entry once, however many rows use it
    for i, f in enumerate(rows):
        r = table[table.formula == f].iloc[0]          # .iloc[0] = the first (only) matching row
        for col, c, mk, lab in marks:
            if pd.notna(r.get(col)):                   # skip values this compound does not have
                ax2.scatter(r[col], i, color=c, marker=mk, s=90 if mk == "*" else 55,
                            label=None if lab in seen else lab, zorder=3)
                seen.add(lab)
        ax2.text(2.9, i + 0.28, r.chain_source, fontsize=7, ha="right", color="0.35")
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(["Cs$_2$SnI$_6$", "Cs$_2$SnBr$_6$"])
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xscale("log"); ax2.set_xlim(0.05, 3)
    ax2.axvline(LOW, color="tab:red", lw=0.8, ls=":")
    ax2.set_xlabel("$\\kappa_L$ (W/m/K, 300 K)")
    ax2.set_title("B  the two compounds with a published number", loc="left", fontsize=10)
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
