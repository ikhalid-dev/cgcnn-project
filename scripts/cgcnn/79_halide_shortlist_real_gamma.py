#!/usr/bin/env python
"""
Step 79 - the halide shortlist, finished: real gamma at the FIXED cutoff.
================================================================================

    ml_env/bin/python scripts/cgcnn/79_halide_shortlist_real_gamma.py

WHAT THIS IS
------------
The halide analogue of step 74. Step 74 did this for the oxides; the halides
have been sitting with real gammas computed (steps 72, 76) and never turned
into a shortlist, so the halide screen still ranked on the Poisson shortcut
gamma. This replaces that shortcut with the measured phonon value.

It is also the first halide list built on the FIXED frequency cutoff. Every
earlier halide gamma used cutoff 0.001 THz, which left the near-Gamma region in
the heat-capacity average - and the mode Grueneisen carries a 1/omega, so that
region is amplified, not merely included. The recut (steps 77-78) recomputed all
138 at 0.3 THz and banked the whole cutoff ladder, so BOTH lists can be built
from the same file and compared directly.

THE IDENTITY THAT MAKES THIS FREE
----------------------------------
PINK Eq. (2) has gamma in exactly one place:

    kappa = G * v_s * V^(1/3) / (n * T) * exp(-gamma)

so swapping gamma is a multiplication, not a recalculation:

    kappa(g_new) = kappa(g_old) * exp(g_old - g_new)

Each model gets its OWN g_old, because each predicted its own K and G and the
shortcut gamma is a function of K/G. Using one model's shortcut gamma on
another's kappa would silently corrupt that model's column.

A multiplication by exp(g_old - g_new) is positive, and the same positive
number for all three models of a given crystal is NOT what happens here
(g_old differs per model) - so kappa_max3 must be recomputed as a max AFTER
the swap, never swapped itself.

WHAT IS PREDICTED AND WHAT IS COMPUTED
---------------------------------------
    K and G            PREDICTED   ALIGNN, CGCNN-ens, new matminer baseline
    gamma              COMPUTED    real phonons at three volumes, MACE-OMAT-0
    dynamic stability  COMPUTED    same phonon run, 0 imaginary modes
    kappa              DERIVED     predicted K,G + computed gamma

OUTPUTS
    results/cgcnn/79_halide_shortlist_real_gamma.csv
    results/cgcnn/79_halide_cutoff_effect.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")
THRESHOLD = 1.0
MB = ["ALIGNN", "CGCNN-ens", "newbase"]


def poisson_gamma(K, G):
    """The shortcut gamma every stored kappa in the screen was built on.

    Reads as: longitudinal-over-transverse sound speed ratio -> Poisson ratio
    -> gamma. All three steps are closed-form, no phonons involved. This is the
    quantity the real calculation replaces.

    K/G is all that survives: v_l/v_t = sqrt((K + 4G/3)/G), so the density and
    volume cancel out of the ratio entirely.
    """
    x2 = K / G + 4.0 / 3.0          # this is (v_l/v_t)^2
    nu = (x2 - 2) / (2 * x2 - 2)    # Poisson ratio from that ratio
    return 3 * (1 + nu) / (2 * (2 - 3 * nu))


def swap(kappa_old, g_old, g_new):
    """Apply the exact identity. np.exp on a Series returns a Series, elementwise."""
    return kappa_old * np.exp(g_old - g_new)


def build(d, gcol, tag):
    """Produce the three model kappas for one choice of gamma column.

    `d` already carries, per crystal: each model's stored kappa, each model's
    K and G, and several gamma columns. `gcol` picks which gamma to believe.
    Returns a copy so the two lists (old cutoff, new cutoff) never share state.
    """
    o = d.copy()
    g_new = o[gcol]
    o["ALIGNN"] = swap(o.ALIGNN_old, o.g_short_alignn, g_new)
    o["CGCNN-ens"] = swap(o.CGCNN_old, o.g_short_cgcnn, g_new)
    o["newbase"] = swap(o.newbase_old, o.g_short_newbase, g_new)
    o["kappa_max3"] = o[MB].max(axis=1)      # axis=1: across columns, per row
    o["kappa_min3"] = o[MB].min(axis=1)
    o["binding3"] = o[MB].idxmax(axis=1)     # WHICH model is the pessimist
    o["survives"] = o.kappa_max3 <= THRESHOLD
    o["gamma_used"] = g_new
    o["which"] = tag
    return o.sort_values("kappa_max3").reset_index(drop=True)


def main():
    # ---- 1. the recut gammas ----------------------------------------------
    rc = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in
                    sorted(glob.glob(os.path.join(
                        GAMMA_DIR, "01_gamma_phonon_halide_recut_s*.csv")))],
                   ignore_index=True).drop_duplicates("mp_id")
    print(f"recut rows: {len(rc)}   status {rc.status.value_counts().to_dict()}")
    print(f"cutoff written into the file: {rc.gamma_cutoff_THz.unique()} THz")

    # ---- 2. the consistency check that makes the recut auditable ----------
    # gamma_c001 reproduces the OLD definition exactly. The original shards
    # computed gamma that way. If the recut is the same calculation with only
    # the cutoff changed, these two must agree.
    old = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in
                     sorted(glob.glob(os.path.join(GAMMA_DIR,
                                                   "01_gamma_phonon_halide_s0*.csv")))
                     + [os.path.join(GAMMA_DIR, "01_gamma_phonon_pilot_mb3.csv")]],
                    ignore_index=True).drop_duplicates("mp_id")
    chk = rc.merge(old[["mp_id", "gamma_mlip"]], on="mp_id",
                   suffixes=("", "_banked"))
    dif = (chk.gamma_c001 - chk.gamma_mlip_banked).abs()
    print(f"\nCONSISTENCY  new gamma_c001 vs banked gamma_mlip over {len(chk)} crystals")
    print(f"   max|diff| {dif.max():.4f}   within 1%: "
          f"{int((dif <= 0.01 * chk.gamma_mlip_banked.abs()).sum())}/{len(chk)}")

    # ---- 3. the production filter ------------------------------------------
    # Applied to the NEW gamma. A crystal whose gamma is negative after the
    # cutoff fix has no physical exp(-gamma) to speak of and is dropped.
    f_new = rc[(rc.status == "ok") & rc.dynamically_stable
               & rc.gamma_c300.between(0, 10, inclusive="neither")]
    f_old = rc[(rc.status == "ok") & rc.dynamically_stable
               & rc.gamma_c001.between(0, 10, inclusive="neither")]
    print(f"\npass the filter on the OLD gamma (c001): {len(f_old)}")
    print(f"pass the filter on the NEW gamma (c300): {len(f_new)}")

    # ---- 4. the screen's stored kappas and moduli -------------------------
    top = pd.read_csv(os.path.join(RESULTS, "71_top300_matbench3.csv"),
                      dtype={"material_id": str})
    cg = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"),
                     dtype={"material_id": str},
                     usecols=["material_id", "K_VRH_pred", "G_VRH_pred",
                              "Gruneisen parameter", "Kappa_cal (W m-1 K-1)"])

    # suffixes: rc and top BOTH carry a `formula` column. Without this, pandas
    # renames them formula_x / formula_y and every later reference breaks.
    # "" on the left keeps rc's names unchanged; the screen's copy gets _scr.
    d = rc.merge(top, left_on="mp_id", right_on="material_id",
                 suffixes=("", "_scr"))
    d = d.merge(cg, on="material_id", suffixes=("", "_scr"))
    d = d.drop(columns=[c for c in d.columns if c.endswith("_scr")])
    print(f"\nrecut crystals found in the top-300 screen: {len(d)} of {len(rc)}")

    d = d.rename(columns={"ALIGNN": "ALIGNN_old", "CGCNN-ens": "CGCNN_old",
                          "newbase": "newbase_old",
                          "Kappa_cal (W m-1 K-1)": "kappa_cgcnn_raw"})
    # Each model's own shortcut gamma, from its own K and G.
    d["g_short_alignn"] = poisson_gamma(d.K_alignn, d.G_alignn)
    d["g_short_newbase"] = poisson_gamma(d.K_newbase, d.G_newbase)
    d["g_short_cgcnn"] = d["Gruneisen parameter"]

    # ---- 5. the two lists -------------------------------------------------
    keep_new = d[d.mp_id.isin(f_new.mp_id)]
    keep_old = d[d.mp_id.isin(f_old.mp_id)]
    new = build(keep_new, "gamma_c300", "cutoff 0.300 (fixed)")
    oldl = build(keep_old, "gamma_c001", "cutoff 0.001 (bug)")

    cols = (["formula", "material_id", "Number of Atoms", "rank3",
             "gamma_used", "gamma_c001", "gamma_c300", "gamma_acoustic",
             "gamma_optical", "n_imaginary", "min_freq_THz"] + MB
            + ["kappa_max3", "kappa_min3", "binding3", "survives",
               "ALIGNN_old", "CGCNN_old", "newbase_old"])
    cols = [c for c in cols if c in new.columns]
    new[cols].to_csv(os.path.join(RESULTS, "79_halide_shortlist_real_gamma.csv"),
                     index=False)

    # ---- 6. what the cutoff fix did, crystal by crystal -------------------
    cmp_ = new[["material_id", "formula", "gamma_c001", "gamma_c300",
                "kappa_max3", "survives"]].rename(
        columns={"kappa_max3": "kappa_new", "survives": "survives_new"})
    cmp_ = cmp_.merge(oldl[["material_id", "kappa_max3", "survives"]].rename(
        columns={"kappa_max3": "kappa_old", "survives": "survives_old"}),
        on="material_id", how="left")
    cmp_["d_gamma"] = cmp_.gamma_c300 - cmp_.gamma_c001
    cmp_["pct_kappa"] = 100 * (cmp_.kappa_new / cmp_.kappa_old - 1)
    cmp_.to_csv(os.path.join(RESULTS, "79_halide_cutoff_effect.csv"), index=False)

    # ---- 7. report ---------------------------------------------------------
    print("\n" + "=" * 96)
    print("STEP 79 - HALIDES ON A REAL GAMMA, FIXED CUTOFF, MATBENCH MODELS ONLY")
    print("=" * 96)
    print(f"\nscored on the NEW gamma: {len(new)}"
          f"   survive kappa_max3 <= {THRESHOLD}: {int(new.survives.sum())}"
          f"  ({100*new.survives.mean():.0f}%)")
    print(f"scored on the OLD gamma: {len(oldl)}"
          f"   survive: {int(oldl.survives.sum())}")

    sn, so = set(new[new.survives].material_id), set(oldl[oldl.survives].material_id)
    print(f"\n   stay {len(sn & so)}   LEAVE {len(so - sn)}   ENTER {len(sn - so)}")
    print(f"\ngamma change : median {cmp_.d_gamma.median():+.4f}"
          f"   range {cmp_.d_gamma.min():+.3f} to {cmp_.d_gamma.max():+.3f}")
    print(f"kappa change : median {cmp_.pct_kappa.median():+.2f}%"
          f"   >10%: {int((cmp_.pct_kappa.abs() > 10).sum())}"
          f"   >20%: {int((cmp_.pct_kappa.abs() > 20).sum())}")

    if so - sn:
        print("\nLEAVE the shortlist:")
        print(cmp_[cmp_.material_id.isin(so - sn)][
            ["formula", "material_id", "gamma_c001", "gamma_c300",
             "kappa_old", "kappa_new"]].to_string(index=False,
                                                  float_format=lambda v: f"{v:.3f}"))
    if sn - so:
        print("\nENTER the shortlist:")
        print(cmp_[cmp_.material_id.isin(sn - so)][
            ["formula", "material_id", "gamma_c001", "gamma_c300",
             "kappa_old", "kappa_new"]].to_string(index=False,
                                                  float_format=lambda v: f"{v:.3f}"))

    surv = new[new.survives]
    print(f"\nsurvivor kappa_max3 range: {surv.kappa_max3.min():.3f} - "
          f"{surv.kappa_max3.max():.3f}")
    print("\ntop 25 survivors:")
    print(surv.head(25)[["formula", "material_id", "Number of Atoms", "rank3",
                         "gamma_used"] + MB + ["kappa_max3", "binding3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbinding model among survivors:")
    print(surv.binding3.value_counts().to_string())
    print(f"\nwrote {RESULTS}/79_halide_shortlist_real_gamma.csv")
    print(f"wrote {RESULTS}/79_halide_cutoff_effect.csv")


if __name__ == "__main__":
    main()
