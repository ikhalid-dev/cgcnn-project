#!/usr/bin/env python
"""
Step 83 - the DFT queue re-sorted by how likely DFT is to CONFIRM each crystal.
================================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/83_trusted_dft_queue.py
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/83_trusted_dft_queue.py --skip-mp

WHY
---
Step 81 ranked the 221 survivors purely by lowest predicted kappa (kappa_max3).
Step 82 measured, on 1,648 held-out matbench crystals with DFT moduli, WHICH
kinds of prediction a DFT elastic run confirms. Two things mattered:

    how low the prediction is   max3 <= 0.5  ->  76 of 77 confirmed as <= 1.0
                                0.5 - 1.0    ->  91 of 100
    whether the models agree    agree (spread <= 1.5) -> DFT within 2x  81%
                                disagree              -> DFT within 2x  63%

So this step changes NO prediction. It only sorts the same 221 crystals into
three groups by those two signals, and attaches to every crystal the range its
DFT kappa is expected to land in.

    group 1  real candidate         max3 <= 0.5  and spread3 <= 1.5
    group 2  low, models disagree   max3 <= 0.5  and spread3 >  1.5
    group 3  near the cutoff        0.5 < max3 <= 1.0
    C        the four negative controls, unchanged from step 81

Within each group crystals are ordered by kappa_max3 - the number a DFT elastic
run tests. kappa_floored (amorphous-limit) stays as a column, so the other
ordering is one sort away.

THE EXPECTED RANGE - WHERE IT COMES FROM
-----------------------------------------
For every test crystal in a group, step 82 recorded DFT kappa / predicted kappa.
The 10th and 90th percentiles of that ratio, within the same group, give an 80%
range. A crystal predicted at 0.10 in a group whose ratios run 0.2-1.2 is
expected to land between 0.02 and 0.12.

That ratio depends only on the MODULI (gamma cancels - prediction and truth used
the same gamma), so it transfers to the screen, where gamma is the MLIP value
and DFT will recompute only K and G. It is NOT a range for the real, measured
kappa: it says nothing about MLIP gamma error or whether Slack holds at all.

OUTPUTS (results/cgcnn/)
    83_trusted_dft_queue.csv      the queue: group, rank, expected DFT range
    83_group_calibration.csv      the test-set numbers each group's range comes from
    83_prior_calculations.csv     Materials Project lookup for group 1 + controls
    83_group1_ranges.png          every group-1 crystal: prediction, range, floor
"""
# ---- imports -------------------------------------------------------------------
import argparse                # reads command-line switches like --skip-mp
import os                      # file paths
import numpy as np             # arithmetic on whole columns
import pandas as pd            # tables
import matplotlib
matplotlib.use("Agg")          # draw to a file, no window
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
MB = ["ALIGNN", "CGCNN-ens", "newbase"]

# The rule. These two numbers are the ONLY tunables, and they are the same
# ones step 82 measured with - changing them here without re-reading step 82's
# trust table would make the ranges below describe a different rule.
LOW = 0.5                      # W/m/K: "deep" prediction
AGREE = 1.5                    # highest / lowest of the three model kappas

GROUPS = {1: "1: real candidate",
          2: "2: low, models disagree",
          3: "3: near the cutoff"}


def assign_group(max3, spread3):
    """Return 1, 2 or 3 for every row. np.select checks conditions in order and
    takes the first that is True - like an if / elif / else for a whole column."""
    return np.select([(max3 <= LOW) & (spread3 <= AGREE),
                      (max3 <= LOW) & (spread3 > AGREE)],
                     [1, 2], default=3)


def wilson(k, n, z=1.96):
    """95% range for a success rate k out of n (same helper as step 82)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# =============================================================================
#  1. calibrate each group on the step-82 test crystals
# =============================================================================
def calibrate():
    t = pd.read_csv(os.path.join(RES, "82_test_predictions.csv"))
    t = t[t.max3_moduli_only <= 1.0].copy()        # only crystals the screen would send
    t["group"] = assign_group(t.max3_moduli_only, t.spread3_moduli_only)
    t["ratio"] = t.kappa_DFT / t.max3_moduli_only  # > 1 means DFT came out HIGHER

    rows = []
    for g, sub in t.groupby("group"):              # one pass per group number
        k = int((sub.kappa_DFT <= 1.0).sum())
        lo, hi = wilson(k, len(sub))
        rows.append({"group": g, "label": GROUPS[g], "n_test": len(sub),
                     "confirmed_low": k, "confirmed_pct": 100 * k / len(sub),
                     "confirmed_ci95": f"{100 * lo:.0f}-{100 * hi:.0f}%",
                     "within_2x_pct": 100 * sub.ratio.between(0.5, 2).mean(),
                     "ratio_p10": sub.ratio.quantile(0.10),
                     "ratio_median": sub.ratio.median(),
                     "ratio_p90": sub.ratio.quantile(0.90)})
    cal = pd.DataFrame(rows)
    # the group counts must reproduce step 82's trust table - same rule, same data
    expect = {1: 18 + 32, 2: 13 + 14, 3: 100}
    got = dict(zip(cal.group, cal.n_test))
    if got != expect:
        raise SystemExit(f"group sizes {got} do not match step 82's table {expect}")
    return cal


# =============================================================================
#  2. Materials Project lookup - has anyone already computed this?
# =============================================================================
def prior_calculations(rows):
    """Same query as step 81, but reuses step 81's answers where they exist so
    no crystal is asked about twice."""
    old_path = os.path.join(RES, "81_prior_calculations.csv")
    old = pd.read_csv(old_path, dtype={"gnome_id": str}) if os.path.exists(old_path) else pd.DataFrame()
    done = set(old.gnome_id) if len(old) else set()
    todo = rows[~rows.material_id.isin(done)]
    print(f"\nMaterials Project: {len(rows) - len(todo)} already checked in step 81, "
          f"querying {len(todo)} more")

    new = []
    if len(todo):
        from mp_api.client import MPRester            # imported here: only this branch needs it
        from pymatgen.core import Composition
        import warnings
        warnings.filterwarnings("ignore")              # mp_api prints a progress bar per query
        with MPRester() as mpr:
            for r in todo.itertuples():                # one row at a time, as a named tuple
                comp = Composition(r.formula)
                chemsys = "-".join(sorted(e.symbol for e in comp.elements))
                docs = mpr.materials.summary.search(
                    chemsys=chemsys, fields=["material_id", "formula_pretty"],
                    num_chunks=None, chunk_size=1000)
                same = [x for x in docs if Composition(x.formula_pretty).reduced_formula
                        == comp.reduced_formula]
                n_el = 0
                if same:
                    n_el = len(mpr.materials.elasticity.search(
                        material_ids=[x.material_id for x in same], fields=["material_id"]))
                new.append({"formula": r.formula, "gnome_id": r.material_id,
                            "chemsys": chemsys, "mp_entries_in_chemsys": len(docs),
                            "mp_same_formula": len(same),
                            "mp_id": same[0].material_id if same else "",
                            "mp_has_elastic": n_el})
    keep = ["formula", "gnome_id", "chemsys", "mp_entries_in_chemsys",
            "mp_same_formula", "mp_id", "mp_has_elastic"]
    pc = pd.concat([old.reindex(columns=keep)[old.gnome_id.isin(rows.material_id)] if len(old)
                    else pd.DataFrame(columns=keep), pd.DataFrame(new, columns=keep)],
                   ignore_index=True)
    pc = rows[["group_label", "material_id"]].merge(pc, left_on="material_id", right_on="gnome_id")
    pc = pc.drop(columns="material_id")
    pc.to_csv(os.path.join(RES, "83_prior_calculations.csv"), index=False)
    print(f"   structure already in MP : {(pc.mp_same_formula > 0).sum()} of {len(pc)}")
    print(f"   ELASTIC TENSORS in MP   : {int((pc.mp_has_elastic > 0).sum())}"
          f"   <- a hit means DFT moduli already exist for that formula")
    if (pc.mp_has_elastic > 0).any():
        print(pc[pc.mp_has_elastic > 0][["group_label", "formula", "gnome_id", "mp_id"]]
              .to_string(index=False))


# =============================================================================
#  main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mp", action="store_true", help="skip the Materials Project lookup")
    args = ap.parse_args()

    cal = calibrate()
    cal.to_csv(os.path.join(RES, "83_group_calibration.csv"), index=False)
    print("WHAT EACH GROUP DID ON THE 1,648 TEST CRYSTALS (step 82 data)")
    print(cal.drop(columns="group").round(2).to_string(index=False))

    q81 = pd.read_csv(os.path.join(RES, "81_combined_dft_queue.csv"), dtype={"material_id": str})
    q = q81[q81.status == "QUEUE"].copy()
    ctl = q81[q81.status == "CONTROL"].copy()

    # spread3 must be exactly max/min of the three model columns, or the rule is
    # being applied to something other than what step 82 measured
    worst = np.max(np.abs(q[MB].max(axis=1) / q[MB].min(axis=1) / q.spread3 - 1))
    if worst > 1e-9:
        raise SystemExit(f"spread3 is not max/min of {MB} (worst {worst:.1e})")

    q["group"] = assign_group(q.kappa_max3, q.spread3)
    q["group_label"] = q.group.map(GROUPS)              # 1 -> "1: real candidate", ...
    q["old_tier_81"] = q.tier                           # keep step 81's tier to show what moved

    # attach each group's test-set numbers to every crystal in it
    c = cal.set_index("group")
    q["test_confirmed_pct"] = q.group.map(c.confirmed_pct).round(1)
    q["test_within_2x_pct"] = q.group.map(c.within_2x_pct).round(1)
    q["dft_expected_lo"] = q.kappa_max3 * q.group.map(c.ratio_p10)
    q["dft_expected_mid"] = q.kappa_max3 * q.group.map(c.ratio_median)
    q["dft_expected_hi"] = q.kappa_max3 * q.group.map(c.ratio_p90)

    # order: group first, then lowest prediction; rank restarts inside each group
    q = q.sort_values(["group", "kappa_max3"]).reset_index(drop=True)
    q["rank_in_group"] = q.groupby("group").cumcount() + 1   # cumcount counts 0,1,2... per group
    q["queue_position"] = np.arange(1, len(q) + 1)

    ctl["group_label"] = "C: negative control"
    cols = (["queue_position", "group_label", "rank_in_group", "old_tier_81", "origin",
             "formula", "material_id", "Number of Atoms", "gamma_mlip"] + MB
            + ["kappa_max3", "spread3", "dft_expected_lo", "dft_expected_mid",
               "dft_expected_hi", "test_confirmed_pct", "test_within_2x_pct",
               "kappa_cahill", "below_floor", "kappa_floored", "rank_floored"])
    out = pd.concat([q.reindex(columns=cols), ctl.reindex(columns=cols)], ignore_index=True)
    out.to_csv(os.path.join(RES, "83_trusted_dft_queue.csv"), index=False)

    # ---- report ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("THE QUEUE, RE-SORTED")
    print("=" * 96)
    for g, label in GROUPS.items():
        sub = q[q.group == g]
        # value_counts() tallies each distinct origin; "oxide+halide" is its own
        # bucket, so the three crystals in both lists are not counted twice
        origins = ", ".join(f"{k} {v}" for k, v in sub.origin.value_counts().items())
        print(f"{label:<26} {len(sub):>4} crystals   ({origins})   "
              f"below amorphous floor {int(sub.below_floor.sum()):>3}")

    g1 = q[q.group == 1]
    show = ["rank_in_group", "old_tier_81", "formula", "material_id", "kappa_max3", "spread3",
            "dft_expected_lo", "dft_expected_hi", "kappa_cahill"]
    print(f"\ngroup 1 - real candidates ({len(g1)})")
    print(g1[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # what happened to step 81's top tier
    a1 = q[q.old_tier_81 == "A1: request"]
    print("\nstep 81's A1 (20) now sits in: "
          + ", ".join(f"{GROUPS[k]} {v}" for k, v in a1.group.value_counts().sort_index().items()))
    print("A1 crystals leaving group 1 (models disagree):")
    print(a1[a1.group != 1][["formula", "material_id", "kappa_max3", "spread3"] + MB]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- figure: every group-1 crystal, prediction + expected DFT range + floor ---
    g1p = g1.iloc[::-1]                                  # reverse so rank 1 is at the top
    y = np.arange(len(g1p))
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(g1p) + 1.8))
    ax.hlines(y, g1p.dft_expected_lo, g1p.dft_expected_hi, color="#9db7d5", lw=4,
              label="expected DFT range (80%)")
    ax.plot(g1p.kappa_max3, y, "o", color="#1f4e8c", ms=5, label="screen prediction (max3)")
    ax.plot(g1p.kappa_cahill, y, "|", color="#c0392b", ms=11, mew=2, label="amorphous floor")
    ax.set_yticks(y, [f"{r.formula}  ({r.origin})" for r in g1p.itertuples()], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("kappa at 300 K (W/m/K), log scale")
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_title(f"Step 83 - group 1, real candidates ({len(g1)}): predicted <= {LOW} "
                 f"and models agree within {AGREE}x", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "83_group1_ranges.png"), dpi=130)
    print(f"\nwrote 83_trusted_dft_queue.csv ({len(out)} rows), 83_group_calibration.csv, "
          f"83_group1_ranges.png")

    if not args.skip_mp:
        live = pd.concat([g1[["material_id", "formula", "group_label"]],
                          ctl[["material_id", "formula", "group_label"]]])
        prior_calculations(live)


if __name__ == "__main__":
    main()
