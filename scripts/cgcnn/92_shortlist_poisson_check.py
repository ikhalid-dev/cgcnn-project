#!/usr/bin/env python
"""
Step 92 - the final 15 (and the 8 reserves) scored on the POISSON-gamma chain
=============================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/92_shortlist_poisson_check.py

THE QUESTION
------------
The final 15 (step 91) were chosen on the MLIP chain: real-phonon gamma, then
step 86's stress test. Step 89 then found MLIP gamma gives NO measurable
precision gain over the cheaper Poisson chain on PhoNIX - the gain of tier 1
came from being strict, not from the gamma. So the 15 should also be checked
on the chain that HAS been calibrated against DFT (step 87). A crystal that
both chains call safely low is the strongest kind of pick.

HOW
---
1. Rebuild each crystal's Poisson-chain kappa for the three models, from the
   values the screen stored (no model is re-run except newbase on the 11
   oxygen-containing crystals, whose moduli were never saved - step 74 re-
   predicted them the same way, with the same deterministic XGBoost files).
2. PROVE they are the right numbers: swapping each model's Poisson gamma for
   the MLIP gamma must reproduce step 83's stored MLIP kappa exactly, because
   gamma enters the Slack formula only as exp(-gamma).
3. Poisson max3 = the highest of the three (the cautious reading, as before).
4. Read step 87's precision-vs-cutoff curve at each crystal's own max3: of the
   PhoNIX crystals the Poisson chain predicts at or below that value, what
   fraction does DFT really put at kappa_L <= 1?
5. Flag BOTH-CHAIN picks: tier 1 on MLIP (stress kappa <= 1) AND Poisson max3
   <= 0.54, the Poisson cutoff step 89 found to be exactly as strict as tier 1.

OUTPUTS
-------
    results/cgcnn/92_shortlist_poisson_check.csv
    results/cgcnn/92_shortlist_poisson_check.png
"""

import os
import sys
from importlib import import_module      # imports a file whose name starts with a digit

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # draw straight to a file, no window
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")

# sys.path = the folders Python searches on "import"; this lets us reuse earlier steps
sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
_s74 = import_module("74_oxide_shortlist_real_gamma")   # kappa_cal, featurize, predict_baseline
_s82 = import_module("82_confusion_and_trust")          # wilson(): error bar for a plain rate

MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]
LOW = 1.0          # step 87's definition of "really low": DFT kappa_L <= 1 W/m/K
P_CUT = 0.54       # step 89: the Poisson cutoff exactly as strict as tier 1 (precision 0.843)
MIN_N = 10         # step 87 did not read the curve where fewer than 10 crystals sit below it


def poisson_gamma(K, G):
    """The shortcut gamma from a model's own bulk (K) and shear (G) modulus.

    Poisson's ratio nu comes from K/G; then gamma = 3(1+nu) / (2(2-3nu)).
    """
    x2 = K / G + 4 / 3                    # (v_long / v_trans)^2, written with K and G
    nu = (x2 - 2) / (2 * x2 - 2)
    return 3 * (1 + nu) / (2 * (2 - 3 * nu))


def one_row_each(df, ids, cols):
    """Keep one row per material_id - but only if its duplicates AGREE.

    Some screen files list a crystal twice. If both copies carry the same
    numbers, either will do; if they differ, we cannot know which the screen
    used, so stop rather than guess.
    """
    sub = df[df.material_id.isin(ids)][["material_id"] + cols]
    # nunique() counts distinct values per column inside each id's group
    clash = sub.groupby("material_id")[cols].nunique().gt(1).any(axis=1)
    if clash.any():
        raise SystemExit(f"conflicting duplicate rows for: {list(clash[clash].index)}")
    return sub.drop_duplicates("material_id")


def curve_at(x, pool):
    """Step 87's curve read at cutoff x: of the PhoNIX crystals predicted <= x,
    how many does DFT put at kappa_L <= 1? Returns n, k, k/n and its 95% range."""
    sel = pool[pool.kappa_max3 <= x]
    n, k = len(sel), int((sel.klat <= LOW).sum())
    lo, hi = _s82.wilson(k, n)
    return n, k, (k / n if n else np.nan), lo, hi


def main():
    # ---- 1. the 23 tier-1 crystals: the final 15 + 8 reserves --------------
    sl = pd.read_csv(os.path.join(RESULTS, "91_final_shortlist.csv"),
                     dtype={"material_id": str})          # keep ids as text: "043233727a" must not become a number
    # | means "or"; .str.startswith tests the start of each text cell
    sl = sl[(sl.decision == "SHORTLIST") | sl.decision.str.startswith("reserve")].copy()
    sl["group"] = np.where(sl.decision == "SHORTLIST", "final 15", "reserve")
    ids = list(sl.material_id)
    print(f"crystals checked: {len(sl)}  ({(sl.group == 'final 15').sum()} final + "
          f"{(sl.group == 'reserve').sum()} reserves)")

    # ---- 2. each model's stored Poisson-chain kappa and moduli -------------
    # ALIGNN: its kappa and its own K, G (step 57)
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    a = one_row_each(a, ids, ["Kappa_alignn", "K_alignn", "G_alignn"])

    # CGCNN ensemble: its kappa and the gamma it already stored (step 13)
    cg = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"), dtype={"material_id": str},
                     usecols=["material_id", "Kappa_cal (W m-1 K-1)", "Gruneisen parameter"])
    cg = one_row_each(cg, ids, ["Kappa_cal (W m-1 K-1)", "Gruneisen parameter"])

    # cell volume, density and atom count - the Slack formula needs them for newbase
    st = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"), dtype={"material_id": str},
                     usecols=["material_id", "Volume (A3)", "Density (g cm-3)", "Number of Atoms"])
    st = one_row_each(st, ids, ["Volume (A3)", "Density (g cm-3)", "Number of Atoms"])

    # newbase moduli. Step 83 took each crystal from ONE of two screens (its `origin`):
    #   halide -> step 79, which used step 71's stored newbase moduli
    #   oxide  -> step 74, which re-predicted newbase from a fresh featurisation
    # A crystal in both screens (Rb5Au(BrO)2) gets slightly different moduli from
    # the two featurisations (0.7% in kappa), so each must come from its own path.
    q = pd.read_csv(os.path.join(RESULTS, "83_trusted_dft_queue.csv"), dtype={"material_id": str},
                    usecols=["material_id", "origin", "gamma_mlip"] + MODELS)
    origin = dict(zip(q.material_id, q.origin))           # id -> "halide" / "oxide"
    halide_ids = [m for m in ids if origin[m] == "halide"]
    need = [m for m in ids if origin[m] == "oxide"]
    nb = pd.read_csv(os.path.join(RESULTS, "71_screen_ranked_5model.csv"), dtype={"material_id": str})
    nb = one_row_each(nb, halide_ids, ["K_newbase", "G_newbase", "newbase"])
    nb = nb.rename(columns={"newbase": "newbase_71"})     # step 71's own kappa, kept for a check
    print(f"newbase moduli: {len(nb)} halide-origin from step 71; "
          f"re-predicting {len(need)} oxide-origin ones the way step 74 did ...")
    F = _s74.featurize(need)
    p = _s74.predict_baseline(F)
    nb = pd.concat([nb, pd.DataFrame({"material_id": F.index, "K_newbase": p["K"],
                                      "G_newbase": p["G"]})], ignore_index=True)

    # join everything onto the 23 rows; an inner merge drops a crystal that is missing anywhere
    d = sl
    for part in [a, cg, st, nb]:
        d = d.merge(part, on="material_id", suffixes=("", "_scr"))
    if len(d) != len(sl):
        raise SystemExit(f"lost crystals while merging: {len(sl)} -> {len(d)}")

    # each model's Poisson-chain kappa (kappa_P) and Poisson gamma (gamma_P)
    d["kP_ALIGNN"] = d.Kappa_alignn
    d["gP_ALIGNN"] = poisson_gamma(d.K_alignn, d.G_alignn)
    d["kP_CGCNN-ens"] = d["Kappa_cal (W m-1 K-1)"]
    d["gP_CGCNN-ens"] = d["Gruneisen parameter"]
    kp, gp = _s74.kappa_cal(d.K_newbase.values, d.G_newbase.values, d["Volume (A3)"].values,
                            d["Density (g cm-3)"].values, d["Number of Atoms"].values)
    d["kP_newbase"], d["gP_newbase"] = kp, gp

    # step 71 already holds newbase kappa for the oxygen-free ones: they must match
    has71 = d.newbase_71.notna()
    print(f"newbase kappa vs step 71's stored value (n={has71.sum()}): "
          f"max relative diff {np.max(np.abs(d.kP_newbase[has71] / d.newbase_71[has71] - 1)):.1e}")

    # ---- 3. PROOF these are the screen's numbers --------------------------
    # Swap each model's Poisson gamma for the MLIP gamma. Because gamma enters
    # only as exp(-gamma), that must land EXACTLY on step 83's stored MLIP kappa.
    q = q.rename(columns={m: f"kM83_{m}" for m in MODELS}).rename(columns={"gamma_mlip": "gamma_mlip_83"})
    d = d.merge(q, on="material_id")
    worst = 0.0
    for m in MODELS:
        swapped = d[f"kP_{m}"] * np.exp(d[f"gP_{m}"] - d.gamma_mlip_83)
        rel = np.abs(swapped / d[f"kM83_{m}"] - 1)
        worst = max(worst, rel.max())
        print(f"  swap check {m:9s}: Poisson kappa x exp(gP - gMLIP) vs step 83, "
              f"max relative diff {rel.max():.1e}")
        if rel.max() > 1e-6:                           # name the crystals that miss, and by how much
            print(d.loc[rel > 1e-6, ["formula", "family"]].assign(rel_diff=rel[rel > 1e-6]).to_string())
    if worst > 1e-6:
        raise SystemExit("swap check FAILED - these are not the screen's Poisson numbers")
    print("  swap check PASSED: the Poisson numbers below are the screen's own")

    # ---- 4. the Poisson-chain max3 ----------------------------------------
    kP = d[[f"kP_{m}" for m in MODELS]]
    d["poisson_max3"] = kP.max(axis=1)                    # highest of the three, crystal by crystal
    d["poisson_min3"] = kP.min(axis=1)
    # idxmax gives the COLUMN name of the max; strip "kP_" to leave the model name
    d["poisson_binding"] = kP.idxmax(axis=1).str.replace("kP_", "", regex=False)
    d["mlip_max3"] = d.kappa_max3                         # step 91's MLIP-chain max3
    d["poisson_over_mlip"] = d.poisson_max3 / d.mlip_max3
    d["gP_binding"] = [d.loc[i, f"gP_{m}"] for i, m in zip(d.index, d.poisson_binding)]

    # ---- 5. step 87's calibration curve, read at each crystal --------------
    pool = pd.read_csv(os.path.join(RESULTS, "87_phonix_poisson_predictions.csv"))
    # sanity: recomputing the curve here must give step 87's saved numbers
    saved = pd.read_csv(os.path.join(RESULTS, "87_precision_vs_cutoff.csv"))
    saved = saved[saved.model == "kappa_max3"]
    same = all(curve_at(c, pool)[:2] == (n, k) for c, n, k in
               zip(saved.pred_cutoff, saved.n_selected, saved.n_dft_low))
    print(f"\nstep 87's curve recomputed from its predictions file: "
          f"{'identical at all 25 cutoffs' if same else 'MISMATCH'}")
    if not same:
        raise SystemExit("cannot reproduce step 87's curve")

    rows = [curve_at(x, pool) for x in d.poisson_max3]
    d["curve_n"], d["curve_k"], d["curve_prec"], d["curve_lo95"], d["curve_hi95"] = zip(*rows)
    d["curve_readable"] = d.curve_n >= MIN_N

    # the curve is CUMULATIVE (everything below x). Local reading too: the band x sits in.
    bands = [(0.0, 0.2), (0.2, P_CUT)]
    print("\nPoisson chain on PhoNIX, LOCAL bands (not cumulative):")
    band_prec = {}
    for lo_, hi_ in bands:
        s = pool[(pool.kappa_max3 > lo_) & (pool.kappa_max3 <= hi_)]
        k, n = int((s.klat <= LOW).sum()), len(s)
        band_prec[(lo_, hi_)] = (k, n, k / n, *_s82.wilson(k, n))
        print(f"  predicted ({lo_:.2f}, {hi_:.2f}]:  {k:3d}/{n:3d} DFT-low = {k/n:.3f}  "
              f"[{band_prec[(lo_, hi_)][3]:.3f}, {band_prec[(lo_, hi_)][4]:.3f}]")
    s = pool[pool.kappa_max3 <= P_CUT]
    k, n = int((s.klat <= LOW).sum()), len(s)
    print(f"  everything <= {P_CUT} (the both-chain cutoff): {k}/{n} = {k/n:.3f}  "
          f"[{_s82.wilson(k, n)[0]:.3f}, {_s82.wilson(k, n)[1]:.3f}]")

    def band_of(x):
        for (lo_, hi_), v in band_prec.items():
            if lo_ < x <= hi_:
                return f"({lo_:.2f},{hi_:.2f}]", v[2], v[3], v[4]
        return "above 0.54", np.nan, np.nan, np.nan
    d["band"], d["band_prec"], d["band_lo95"], d["band_hi95"] = zip(*[band_of(x) for x in d.poisson_max3])

    # ---- 6. the both-chain flag -------------------------------------------
    d["mlip_safe"] = d.kappa_stress <= 1.0                # tier 1 (true for all 23 by construction)
    d["poisson_safe"] = d.poisson_max3 <= P_CUT
    d["both_chains"] = d.mlip_safe & d.poisson_safe
    d["below_cahill_poisson"] = d.poisson_max3 < d.kappa_cahill

    # merge() keeps the left table's row order, so d is still in step 91's order:
    # the 15 by rank, then the reserves in their reserve order
    d = d.reset_index(drop=True)

    # ---- 7. print the table -----------------------------------------------
    pd.set_option("display.width", 250)
    show = d[["rank", "formula", "group", "mlip_max3", "kappa_stress", "poisson_max3",
              "poisson_binding", "gP_binding", "gamma_mlip", "curve_n", "curve_prec",
              "curve_lo95", "curve_hi95", "band_prec", "both_chains"]].copy()
    show["rank"] = show["rank"].astype("Int64")         # whole numbers, blank for reserves
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    f15 = d[d.group == "final 15"]
    print(f"\nfinal 15 safe on BOTH chains: {f15.both_chains.sum()}/15   "
          f"reserves: {d[d.group == 'reserve'].both_chains.sum()}/{(d.group == 'reserve').sum()}")
    print(f"Poisson max3 range on the 15: {f15.poisson_max3.min():.3f} - {f15.poisson_max3.max():.3f}  "
          f"(MLIP max3: {f15.mlip_max3.min():.3f} - {f15.mlip_max3.max():.3f})")
    print(f"Poisson higher than MLIP (more cautious) on {(f15.poisson_over_mlip > 1).sum()}/15; "
          f"median Poisson/MLIP = {f15.poisson_over_mlip.median():.2f}")
    print(f"curve readable (>= {MIN_N} PhoNIX crystals below the value) for "
          f"{f15.curve_readable.sum()}/15")
    print(f"Poisson max3 below the Cahill floor: {f15.below_cahill_poisson.sum()}/15")
    print("binding model on the Poisson chain (final 15):",
          f15.poisson_binding.value_counts().to_dict())

    cols = (["rank", "group", "formula", "material_id", "family", "n_atoms", "gamma_mlip"]
            + [f"gP_{m}" for m in MODELS] + [f"kP_{m}" for m in MODELS]
            + ["poisson_max3", "poisson_min3", "poisson_binding", "mlip_max3", "kappa_stress",
               "kappa_cahill", "poisson_over_mlip", "curve_n", "curve_k", "curve_prec",
               "curve_lo95", "curve_hi95", "curve_readable", "band", "band_prec", "band_lo95",
               "band_hi95", "mlip_safe", "poisson_safe", "both_chains", "below_cahill_poisson"])
    out = os.path.join(RESULTS, "92_shortlist_poisson_check.csv")
    d[cols].to_csv(out, index=False)
    print(f"\nwrote {out}")

    # ---- 8. the figure ----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})

    # left: one row per crystal, MLIP and Poisson max3 joined by a line
    y = np.arange(len(d))[::-1]                          # first crystal at the top
    labels = [f"{int(r):2d} {f}" if pd.notna(r) else f"res {f}" for r, f in zip(d["rank"], d.formula)]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax[0].plot([r.mlip_max3, r.poisson_max3], [yi, yi], color="0.7", lw=1.5, zorder=1)
    ax[0].scatter(d.mlip_max3, y, color="tab:blue", label="MLIP chain max3", zorder=3)
    ax[0].scatter(d.kappa_stress, y, color="tab:blue", marker="|", s=150,
                  label="MLIP chain, stress-tested (tier 1 needs <= 1)", zorder=3)
    ax[0].scatter(d.poisson_max3, y, color="tab:orange", marker="s", label="Poisson chain max3", zorder=3)
    ax[0].scatter(d.kappa_cahill, y, color="k", marker="x", s=25, label="Cahill floor", zorder=2)
    ax[0].axvline(P_CUT, color="tab:orange", ls="--", lw=1, label=f"Poisson cutoff {P_CUT} (as strict as tier 1)")
    ax[0].axvline(LOW, color="k", ls=":", lw=1)
    ax[0].axhline(y[(d.group == "final 15").sum() - 1] - 0.5, color="0.5", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].set_xlabel("predicted kappa_L (W/m/K, log scale)")
    ax[0].set_title("final 15 (top) and reserves (below the line):\nboth chains' max3 per crystal")
    ax[0].legend(fontsize=8, loc="upper right")         # top right is empty: rows 1-6 sit far left

    # right: step 87's curve, finely sampled, with the 15 marked on it
    xs = np.geomspace(0.08, 1.5, 120)
    cur = pd.DataFrame([curve_at(x, pool) for x in xs], columns=["n", "k", "p", "lo", "hi"])
    ok = cur.n >= MIN_N
    ax[1].plot(xs[ok], cur.p[ok], color="tab:orange", lw=2, label="step 87: precision of 'predicted <= x'")
    ax[1].fill_between(xs[ok], cur.lo[ok], cur.hi[ok], color="tab:orange", alpha=0.2, label="95% range")
    ax[1].axvline(P_CUT, color="tab:orange", ls="--", lw=1)
    f = d[d.group == "final 15"]
    ax[1].scatter(f.poisson_max3, np.where(f.curve_readable, f.curve_prec, np.nan),
                  color="k", zorder=3, label="the final 15 at their Poisson max3")
    ax[1].plot(f.poisson_max3[~f.curve_readable], [0.32] * (~f.curve_readable).sum(), "kv",
               label=f"below the curve's readable range (< {MIN_N} PhoNIX crystals)")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0.3, 1.0)
    ax[1].set_xlabel("Poisson-chain max3 cutoff x (W/m/K, log scale)")
    ax[1].set_ylabel("fraction with DFT kappa_L <= 1")
    # the blind-pick rate (0.23) sits below this y-axis, so it goes in the title instead of as a line
    ax[1].set_title("PhoNIX calibration (step 87), read at each crystal\n"
                    f"(a blind pick is DFT-low {(pool.klat <= LOW).mean():.2f} of the time)")
    ax[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    png = os.path.join(RESULTS, "92_shortlist_poisson_check.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
