#!/usr/bin/env python
"""
Step 69 - re-score DFT candidates using the REAL Gruneisen parameter instead of
the shortcut one.
================================================================================

    ml_env/bin/python scripts/cgcnn/69_replacement_kappa_real_gamma.py
    ml_env/bin/python scripts/cgcnn/69_replacement_kappa_real_gamma.py --set dft_queue

Two candidate sets go through the identical physics:

    --set replacements   the ten step-68 nominees (six usable)   [default]
    --set dft_queue      the original step-61 queue - arm A's eight consensus-low
                         crystals plus arm C's four stiff negative controls

Arm B never had phonons computed, so it cannot be re-scored here. Arm C is
included deliberately: those four were chosen because all four models call them
STIFF, so if the real gamma pushes them around as violently as it pushes arm A,
the effect is an artifact of the swap rather than a property of soft crystals.
They are the control on this script's own conclusion.

WHY THIS SCRIPT EXISTS
----------------------
Step 68 nominated ten replacement crystals and ranked them by `kappa_max` - the
most pessimistic of four ML kappa predictions. Every one of those four kappas
was built on a gamma that nobody had ever checked: three of them use the
empirical Poisson relation gamma = 3(1+nu)/(2(2-3nu)), which the sibling
workspace measured at Pearson r = 0.24 against 38 real Grueneisen parameters.
In other words the ranking is built on a number that carries almost none of the
ordering of the quantity it is named after.

The MLIP phonon run (data_generation, tag `replacements`) has now produced a
REAL gamma for all ten - mode Grueneisen parameters from phonons at three
volumes, which measured r = 0.735 on the same 38 crystals. Six of the ten are
dynamically stable and therefore have a usable label.

This script puts the real gamma through the same physics and asks whether the
ranking survives.

WHY THE SUBSTITUTION IS EXACT, NOT A RE-RUN
-------------------------------------------
PINK's Equation (2) is

    kappa_cal = G * v_sound * V^(1/3) / (n * T) * exp(-gamma)

Gamma enters through ONE factor, exp(-gamma), and nothing else - not the sound
velocity, not the Debye temperature, not the volume. So swapping one gamma for
another is closed form:

    kappa(g_new) = kappa(g_old) * exp(g_old - g_new)

No phonons, no re-predicting moduli, no refitting. Everything else in the chain
is held fixed by construction, which is exactly what makes this a clean
controlled comparison rather than a new screen.

Classic Slack is reported beside it as a robustness check. There gamma enters
TWICE - once in the empirical prefactor and once as 1/gamma^2 - so it weights
gamma differently, and a conclusion that survives both functional forms is not
an artifact of either one. PINK Eq. (2) is the primary formula; that is the
project's stated convention and Slack never overrides it.

THE FOUR CHAINS, AND WHERE EACH ONE'S OLD GAMMA COMES FROM
-----------------------------------------------------------
The swap needs a matched pair (kappa_old, gamma_old) per model. Getting this
pairing wrong would silently scale a kappa by the wrong exponential, so each is
taken from the file that actually produced it:

    ALIGNN      Kappa_alignn                 <- 57_gnome_screen_alignn.csv
                gamma DERIVED here from K_alignn/G_alignn, because that screen
                never wrote a gamma column - the same Poisson relation
                07_predict_kappa.py uses, reproduced below rather than imported
                so this script has no import-time dependency on that module.

    CGCNN-ens   Kappa_cal (W m-1 K-1)        <- 13_gnome_screen_all.csv
                "Gruneisen parameter"        <- same file, explicit column

    round9      Kappa_r9_gamma               <- 39_gnome_screen_all_gamma.csv
                gamma_r9_pred                <- same file, a TRAINED head, not
                the Poisson relation - the one chain whose gamma was fitted

    tree        Kappa_baseline               <- 39_gnome_screen_all_gamma.csv
                gamma_baseline_rf            <- same file. 45's line 262 sets
                Kappa_baseline = Kappa_baseline_rf, so the RF gamma is the
                matching one, NOT the xgboost one.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not change which crystals go to DFT. Step 68's ranking was fixed before
any gamma existed, and re-ranking on a variable after seeing its values is
post-hoc. This produces the evidence; the decision stays a human one. It also
does not touch the four dynamically unstable nominees - a gamma from a structure
with imaginary modes is not a measurement of anything.

OUTPUTS
    results/cgcnn/69_replacement_kappa_real_gamma.csv      (--set replacements)
    results/cgcnn/69_dft_queue_kappa_real_gamma.csv        (--set dft_queue)
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")

# Each candidate set: where its real gammas are, which file lists the crystals,
# and what to call the output. Keeping these in one dict rather than in branching
# code means the physics below runs identically for both - which is the whole
# point of re-using this script instead of writing a second one.
SETS = {
    "replacements": {
        "gamma_csv": "01_gamma_phonon_replacements.csv",
        "pool_csv": "68_replacement_pool.csv",
        "out_csv": "69_replacement_kappa_real_gamma.csv",
        "extra_cols": ["rank_in_pool"],
    },
    "dft_queue": {
        "gamma_csv": "01_gamma_phonon_dft_queue.csv",
        "pool_csv": "61_dft_validation_set.csv",
        "out_csv": "69_dft_queue_kappa_real_gamma.csv",
        "extra_cols": ["arm"],
    },
}

# The Slack prefactor's empirical constant, copied from 07_predict_kappa.py
# line 104 so the two cannot silently drift apart.
SLACK_A = 2.43e-8


def derived_gamma(k_gpa, g_gpa):
    """The empirical Poisson relation - gamma from the K/G ratio alone.

    This is the 'shortcut' gamma the whole screen has been running on. It is a
    smooth function of one number (K/G), which is precisely why it cannot carry
    real gamma's ordering: two crystals with the same stiffness ratio get the
    same gamma no matter how differently their phonons actually behave.

    np.errstate(all="ignore") suppresses numpy's divide-by-zero warnings for
    degenerate ratios; the caller checks for finiteness afterwards instead.
    """
    with np.errstate(all="ignore"):
        x2 = np.asarray(k_gpa, float) / np.asarray(g_gpa, float) + 4.0 / 3.0
        nu = (x2 - 2.0) / (2.0 * x2 - 2.0)          # Poisson's ratio from K/G
        return 3.0 * (1.0 + nu) / (2.0 * (2.0 - 3.0 * nu))


def swap_pink(kappa_old, gamma_old, gamma_new):
    """PINK Eq. (2): kappa is proportional to exp(-gamma), so the swap is exact."""
    return kappa_old * np.exp(gamma_old - gamma_new)


def swap_slack(kappa_old, gamma_old, gamma_new):
    """Classic Slack: gamma appears in the prefactor AND as 1/gamma^2.

    Still exact, but the ratio does not collapse to one exponential - both
    factors have to move, so they are written out separately here.
    """
    def pref(g):
        return SLACK_A / (1 - 0.514 / g + 0.228 / g ** 2)
    return kappa_old * (pref(gamma_new) / pref(gamma_old)) * (gamma_old / gamma_new) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="replacements", choices=list(SETS),
                    help="which candidate set to re-score")
    args = ap.parse_args()
    cfg = SETS[args.which]

    # ---- the real gammas, from the MLIP phonon run -------------------------
    gamma_csv = os.path.join(GAMMA_DIR, cfg["gamma_csv"])
    if not os.path.exists(gamma_csv):
        raise SystemExit(f"missing {gamma_csv}\n"
                         f"Run the data_generation `{args.which}` kernel first.")
    gam = pd.read_csv(gamma_csv, dtype={"mp_id": str})

    # A row that failed outright (e.g. Cs4Sb4PdPt's Niggli reduction error) has
    # no gamma at all, so drop it before the stability filter rather than let a
    # NaN propagate silently into the comparison.
    gam = gam[gam["status"] == "ok"].copy()

    # The project's production filter, applied verbatim: a gamma counts only if
    # the crystal is dynamically stable AND the finite-difference ratio did not
    # blow up. Very soft cells can be stable and still return nonsense (the
    # pilot saw hcp Cs at gamma = 40.8), which is what the 0 < gamma < 10 guard
    # is for. `inclusive="neither"` makes both ends strict.
    gam["usable"] = (gam["dynamically_stable"]
                     & gam["gamma_mlip"].between(0, 10, inclusive="neither"))
    good = gam[gam["usable"]].copy()

    # ---- the four kappa chains, each with its own matched gamma ------------
    alignn = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                         dtype={"material_id": str},
                         usecols=["material_id", "K_alignn", "G_alignn", "Kappa_alignn"])
    alignn["gamma_old_ALIGNN"] = derived_gamma(alignn["K_alignn"], alignn["G_alignn"])

    mb = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"),
                     dtype={"material_id": str},
                     usecols=["material_id", "Kappa_cal (W m-1 K-1)", "Gruneisen parameter"])

    r9 = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"),
                     dtype={"material_id": str},
                     usecols=["material_id", "Kappa_r9_gamma", "gamma_r9_pred",
                              "Kappa_baseline", "gamma_baseline_rf"])

    pool = pd.read_csv(os.path.join(RESULTS, cfg["pool_csv"]),
                       dtype={"material_id": str},
                       usecols=cfg["extra_cols"] + ["formula", "material_id",
                                                    "space_group", "kappa_max"])
    # The two pool files label their rows differently - 68 by rank, 61 by arm.
    # Normalise to one column so every print below is set-agnostic.
    label = cfg["extra_cols"][0]
    pool["label"] = pool[label]

    # Chained merges keep only rows present in all four - here that is exactly
    # the six usable survivors, because `good` is the left-most frame.
    d = (good[["mp_id", "formula", "gamma_mlip", "gamma_mlip_1000K"]]
         .rename(columns={"mp_id": "material_id", "formula": "formula_phonon"})
         .merge(pool, on="material_id")
         .merge(alignn, on="material_id")
         .merge(mb, on="material_id")
         .merge(r9, on="material_id"))

    # (column label, kappa_old column, gamma_old column)
    CHAINS = [("ALIGNN",    "Kappa_alignn",           "gamma_old_ALIGNN"),
              ("CGCNN-ens", "Kappa_cal (W m-1 K-1)",  "Gruneisen parameter"),
              ("round9",    "Kappa_r9_gamma",         "gamma_r9_pred"),
              ("tree",      "Kappa_baseline",         "gamma_baseline_rf")]

    for name, kcol, gcol in CHAINS:
        d[f"{name}_gamma_old"] = d[gcol]
        d[f"{name}_kappa_old"] = d[kcol]
        d[f"{name}_kappa_new"] = swap_pink(d[kcol], d[gcol], d["gamma_mlip"])
        d[f"{name}_kappa_new_slack"] = swap_slack(d[kcol], d[gcol], d["gamma_mlip"])

    # The screen's decision statistic: the most pessimistic model, before and
    # after. axis=1 takes the max ACROSS the four columns for each row.
    new_cols = [f"{n}_kappa_new" for n, _, _ in CHAINS]
    old_cols = [f"{n}_kappa_old" for n, _, _ in CHAINS]
    d["kappa_max_new"] = d[new_cols].max(axis=1)
    d["kappa_max_old"] = d[old_cols].max(axis=1)
    d["kappa_max_new_slack"] = d[[f"{n}_kappa_new_slack" for n, _, _ in CHAINS]].max(axis=1)

    d = d.sort_values(["label", "kappa_max"]).reset_index(drop=True)
    d["rank_new"] = d["kappa_max_new"].rank().astype(int)   # 1 = best low-kappa candidate

    # ---- report ------------------------------------------------------------
    print("=" * 96)
    print(f"STEP 69 [{args.which}] - RE-SCORED WITH REAL GAMMA (PINK Eq. 2), n={len(d)}")
    print("=" * 96)
    print("\nShortcut gamma vs real gamma, per crystal:\n")
    show = d[["label", "formula", "material_id", "gamma_mlip",
              "ALIGNN_gamma_old", "CGCNN-ens_gamma_old", "round9_gamma_old",
              "tree_gamma_old"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n\nkappa_max, the statistic the shortlist ranked on:\n")
    tab = d[["label", "formula", "kappa_max_old", "kappa_max_new",
             "kappa_max_new_slack", "rank_new"]].copy()
    tab["change"] = tab["kappa_max_new"] / tab["kappa_max_old"]
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Survival of the 1.0 W/m/K low-kappa line, which is what the DFT request
    # is actually asserting about these crystals.
    print("\n\nStill below the 1.0 W/m/K low-kappa line:")
    for col, lab in [("kappa_max_old", "shortcut gamma"),
                     ("kappa_max_new", "real gamma, PINK Eq.2"),
                     ("kappa_max_new_slack", "real gamma, Slack")]:
        print(f"   {lab:24s} {(d[col] <= 1.0).sum()}/{len(d)}")

    print("\n\nPer-model kappa, old -> new (PINK Eq. 2):\n")
    for name, _, _ in CHAINS:
        # iterrows() rather than itertuples(): itertuples renames any column
        # that is not a valid Python identifier, and "CGCNN-ens_kappa_old"
        # contains a hyphen, so attribute access on it fails. iterrows gives a
        # Series per row, which is indexed by the literal column name.
        line = "  ".join(f"{row['formula']:>14s} {row[f'{name}_kappa_old']:5.3f}->"
                         f"{row[f'{name}_kappa_new']:5.3f}"
                         for _, row in d.iterrows())
        print(f"{name:10s} {line}")

    old_order = list(d.sort_values("kappa_max_old").formula)
    new_order = list(d.sort_values("kappa_max_new").formula)
    print(f"\n\nranking on shortcut gamma : {old_order}")
    print(f"ranking on real gamma     : {new_order}")
    print(f"order changed             : {old_order != new_order}")

    out = os.path.join(RESULTS, cfg["out_csv"])
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
