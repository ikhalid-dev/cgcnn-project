#!/usr/bin/env python
"""
Step 74 - the oxide shortlist, finished: real gamma, matbench-only models.
================================================================================

    ml_env/bin/python scripts/cgcnn/74_oxide_shortlist_real_gamma.py

WHY THIS EXISTS
---------------
The sibling workspace spent ~28 GPU-hours computing real Grueneisen parameters
for the GNoME oxide low-kappa shortlist and banked 355 usable labels. Those
labels were used once, to measure that the Poisson shortcut gamma was flattering
the screen, and then left alone. They have never been turned into a shortlist of
their own - which is free, because the expensive part is already paid for.

Two things changed since they were computed, and both have to be applied or the
answer is stale:

  1. REAL GAMMA replaces the Poisson shortcut, via step 69's exact identity
     kappa(g_new) = kappa(g_old) * exp(g_old - g_new).
  2. THE AFLOW-TRAINED MODELS ARE OUT. round 9 and the composition tree gated
     every previous shortlist; steps 51/55/59 showed AFLOW's labels are the weak
     link, and step 71 measured round 9's kappa spanning only a factor of 4.7
     across the whole screen against the matbench models' factor of 29 - it has
     almost no dynamic range, so whenever it binds it flattens the ranking.

So this ranks on kappa_max3 = max(ALIGNN, CGCNN-ens, new matminer/XGBoost
baseline), with the real gamma, and keeps everything at or below 1.0 W/m/K.

THE NEW BASELINE HAS TO BE COMPUTED HERE
----------------------------------------
ALIGNN and CGCNN-ens already have a kappa for these crystals from the screen.
The new baseline does not - the oxides were never in step 71's featurised pool,
because that pool was the top 3,000 by the old four-model ranking and only 1 of
its top 300 was an oxide. So this featurises them (297 matminer descriptors),
predicts K and G with the 5-fold ensemble, and pushes them through the same
physics chain as everything else.

WHAT THIS IS NOT
----------------
Not a new screen. These 355 are whatever the oxide shortlist happened to
contain and whatever survived the phonons - a selected set, not a ranked sweep
of all oxides. The honest framing is "of the oxides we already paid to measure,
these are the ones that survive", not "these are the best oxides in GNoME".

OUTPUT
    results/cgcnn/74_oxide_shortlist_real_gamma.csv
"""
import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
BASELINE = os.path.expanduser("~/Desktop/pink_reproduction/models")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")
THRESHOLD = 1.0
MB = ["ALIGNN", "CGCNN-ens", "newbase"]


def kappa_cal(K, G, V, rho, n):
    vl = ((K + 4 * G / 3) / rho) ** 0.5 * 1000
    vt = (G / rho) ** 0.5 * 1000
    vs = ((1 / vl ** 3 + 2 / vt ** 3) / 3) ** (-1 / 3)
    r = vl / vt
    nu = (r ** 2 - 2) / (2 * r ** 2 - 2)
    gam = 3 * (1 + nu) / (2 * (2 - 3 * nu))
    return G * 1e9 * vs * (V * 1e-30) ** (1 / 3) / (n * 300) * np.exp(-gam), gam


def featurize(ids):
    from pymatgen.core import Structure
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import DensityFeatures, SiteStatsFingerprint
    from matminer.featurizers.site import BondOrientationalParameter
    cf, df_ = ElementProperty.from_preset("magpie"), DensityFeatures()
    cn = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")
    bo = SiteStatsFingerprint(BondOrientationalParameter(max_l=10),
                              stats=("mean", "std_dev", "minimum", "maximum"))
    cols = (cf.feature_labels() + df_.feature_labels()
            + cn.feature_labels() + bo.feature_labels())
    summ = pd.read_csv(os.path.join(ROOT, "gnome_data/stable_materials_summary.csv"),
                       usecols=["Composition", "MaterialId"], dtype=str)
    comp = dict(zip(summ.MaterialId, summ.Composition))
    z = zipfile.ZipFile(os.path.join(ROOT, "gnome_data/by_composition.zip"))
    names = set(z.namelist())
    rows = []
    for i, mid in enumerate(ids):
        e = f"by_composition/{comp.get(mid)}.CIF"
        if e not in names:
            continue
        try:
            s = Structure.from_str(z.read(e).decode(), fmt="cif")
            rows.append(dict(zip(cols, cf.featurize(s.composition) + df_.featurize(s)
                                 + cn.featurize(s) + bo.featurize(s)), material_id=mid))
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  featurised {i+1}/{len(ids)}", flush=True)
    return pd.DataFrame(rows).set_index("material_id")


def predict_baseline(F):
    import xgboost as xgb
    out = {}
    for task, lab in [("matbench_log_kvrh", "K"), ("matbench_log_gvrh", "G")]:
        meta = json.load(open(os.path.join(
            BASELINE, f"{task}_composition+structure+angular_meta.json")))
        X = F.reindex(columns=meta["columns"]).astype(float).fillna(
            pd.Series(meta["train_medians"]))
        preds = []
        for fold in range(5):
            m = xgb.XGBRegressor()
            m.load_model(os.path.join(
                BASELINE, f"{task}_composition+structure+angular_fold{fold}.json"))
            preds.append(m.predict(X))
        out[lab] = 10 ** np.mean(preds, axis=0)
    return out


def main():
    # ---- the banked oxide gammas ------------------------------------------
    g = pd.concat([pd.read_csv(f, dtype={"mp_id": str})
                   for f in glob.glob(os.path.join(GAMMA_DIR, "01_gamma_phonon_gnome_*.csv"))
                   if not f.endswith("selected.csv")],
                  ignore_index=True).drop_duplicates("mp_id")

    # ---- the FIXED-cutoff gammas override the banked ones -----------------
    # The banked gammas above were computed with a 0.001 THz frequency cutoff,
    # which leaves the near-Gamma modes - where mode gamma diverges as 1/omega -
    # in the average. Steps 77/78 re-ran the 179 oxides that could matter at
    # 0.3 THz. Where a recut exists it wins; `gamma_source` records which one
    # each row used, so no reader has to guess.
    #
    # The 176 NOT recut all have kappa_max3 >= 1.52 on the old gamma, and the
    # largest kappa DROP the fix produced anywhere was x0.953 - so none of them
    # can reach the 1.0 gate. Leaving their old gamma cannot change the list.
    rc = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in
                    sorted(glob.glob(os.path.join(GAMMA_DIR,
                                                  "01_gamma_phonon_oxide_recut_s0?.csv")))],
                   ignore_index=True).drop_duplicates("mp_id")
    # .map() looks each mp_id up in the recut table; ids with no recut get NaN,
    # and .fillna() then falls back to the banked value for exactly those rows.
    new_g = g.mp_id.map(rc.set_index("mp_id").gamma_c300)
    new_st = g.mp_id.map(rc.set_index("mp_id").dynamically_stable)
    g["gamma_source"] = np.where(new_g.notna(), "recut 0.3 THz", "banked 0.001 THz")
    g["gamma_mlip"] = new_g.fillna(g.gamma_mlip)
    g["dynamically_stable"] = new_st.fillna(g.dynamically_stable).astype(bool)
    print(f"recut gammas applied: {int(new_g.notna().sum())} of {len(g)}")

    g = g[(g.status == "ok") & g.dynamically_stable
          & g.gamma_mlip.between(0, 10, inclusive="neither")]
    print(f"banked usable oxide gammas: {len(g)}")

    # ---- the screen's own kappa and geometry for those crystals -----------
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    s = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"),
                    dtype={"material_id": str})
    d = a.merge(s, on="material_id", suffixes=("", "_dup"))
    d = d[d["alignn_prediction_reliable"]]
    d = d.merge(g[["mp_id", "gamma_mlip", "gamma_mlip_1000K", "n_imaginary", "gamma_source"]],
                left_on="material_id", right_on="mp_id")
    print(f"of those, present in the scored screen: {len(d)}")

    # the shortcut gamma each stored kappa was built on
    x2 = d.K_alignn / d.G_alignn + 4 / 3
    nu = (x2 - 2) / (2 * x2 - 2)
    d["g_short_alignn"] = 3 * (1 + nu) / (2 * (2 - 3 * nu))
    d["ALIGNN"] = d.Kappa_alignn * np.exp(d.g_short_alignn - d.gamma_mlip)

    mb = pd.read_csv(os.path.join(RESULTS, "13_gnome_screen_all.csv"),
                     dtype={"material_id": str},
                     usecols=["material_id", "Kappa_cal (W m-1 K-1)", "Gruneisen parameter"])
    d = d.merge(mb, on="material_id")
    d["CGCNN-ens"] = (d["Kappa_cal (W m-1 K-1)"]
                      * np.exp(d["Gruneisen parameter"] - d.gamma_mlip))

    F = featurize(list(d.material_id))
    p = predict_baseline(F)
    nb = pd.DataFrame({"material_id": F.index, "K_nb": p["K"], "G_nb": p["G"]})
    d = d.merge(nb, on="material_id")
    kp, gp = kappa_cal(d.K_nb.values, d.G_nb.values, d["Volume (A3)"].values,
                       d["Density (g cm-3)"].values, d["Number of Atoms"].values)
    d["newbase"] = kp * np.exp(gp - d.gamma_mlip.values)

    d["kappa_max3"] = d[MB].max(axis=1)
    d["kappa_min3"] = d[MB].min(axis=1)
    d["binding3"] = d[MB].idxmax(axis=1)
    d["survives"] = d.kappa_max3 <= THRESHOLD
    d = d.sort_values("kappa_max3")

    cols = ["formula", "material_id", "Number of Atoms", "gamma_mlip", "gamma_source"] + MB + \
           ["kappa_max3", "kappa_min3", "binding3", "survives",
            "Kappa_alignn", "Kappa_cal (W m-1 K-1)", "has_oxygen"]
    cols = [c for c in cols if c in d.columns]
    d[cols].to_csv(os.path.join(RESULTS, "74_oxide_shortlist_real_gamma.csv"), index=False)

    print("\n" + "=" * 96)
    print("STEP 74 - OXIDES THAT SURVIVE A REAL GAMMA, ON MATBENCH MODELS ONLY")
    print("=" * 96)
    print(f"\nscored: {len(d)}   survive kappa_max3 <= {THRESHOLD}: {int(d.survives.sum())}"
          f"  ({100*d.survives.mean():.0f}%)")
    if "has_oxygen" in d:
        print(f"of the survivors, actually contain oxygen: "
              f"{int(d[d.survives].has_oxygen.sum())}/{int(d.survives.sum())}")
    print(f"\nsurvivor kappa_max3 range: {d[d.survives].kappa_max3.min():.3f} - "
          f"{d[d.survives].kappa_max3.max():.3f}")
    print("\ntop 25 survivors:")
    print(d[d.survives].head(25)[["formula", "material_id", "Number of Atoms",
                                  "gamma_mlip"] + MB + ["kappa_max3", "binding3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbinding model among survivors:")
    print(d[d.survives].binding3.value_counts().to_string())
    print(f"\nwrote {RESULTS}/74_oxide_shortlist_real_gamma.csv")


if __name__ == "__main__":
    main()
