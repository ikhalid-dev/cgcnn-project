#!/usr/bin/env python
"""
Step 73 - rebuild the DFT queue with the AFLOW-trained models removed.
================================================================================

    ml_env/bin/python scripts/cgcnn/73_matbench3_dft_queue.py

WHY THIS EXISTS
---------------
Step 70's queue gated on `kappa_max_new` = the most pessimistic of FOUR models
with the real gamma. Two of those four - round 9 and the composition tree - are
AFLOW-trained, and this project's own steps 51/55/59 established that AFLOW's
labels are the weak link: round 9 loses F1 0.400 to the tree's 0.646 on AFLOW's
own test set and sits at 2.4x its label floor.

Checked directly before writing this script, and it is not a marginal effect:

    round 9 set the gate for ALL SIX arm-A candidates and FIVE of the six
    replacements. The tree set the sixth. Not one accept/reject decision in
    the DFT queue was made by a matbench-trained model.

So the queue is rebuilt on the three matbench-trained models only:

    kappa_max3 = max(ALIGNN, CGCNN-ens, new matminer/XGBoost baseline)

round 9 and the tree are still WRITTEN to the output as columns, because the
AFLOW-vs-matbench label-noise result is worth keeping for the thesis. They
simply never gate anything again.

WHERE EACH NUMBER COMES FROM
----------------------------
ALIGNN and CGCNN-ens already have a real-gamma kappa: step 69 computed it by
the exact-swap identity kappa(g_new) = kappa(g_old) * exp(g_old - g_new).

The new baseline does not - it was never run on these crystals. So this script
featurises them (297 matminer descriptors), predicts K and G with the 5-fold
ensemble, pushes those through the same physics chain, and applies the same
gamma swap. Identical treatment, no shortcuts.

THE ADMISSION RULE IS UNCHANGED
-------------------------------
Phonons succeeded, dynamically stable with 0 < gamma < 10, and kappa_max <= 1.0
W/m/K under PINK Eq. (2) with the real gamma. Only the set of models feeding
kappa_max changes. Slack is still recorded, never used as a gate.

OUTPUT
    results/cgcnn/73_matbench3_dft_queue.csv
"""
import json
import math
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
BASELINE = os.path.expanduser("~/Desktop/pink_reproduction/models")
THRESHOLD = 1.0

h, k = 6.62607015e-34, 1.380649e-23
MB = ["ALIGNN", "CGCNN-ens", "newbase"]        # matbench-trained: these gate
AF = ["round9", "tree"]                        # AFLOW-trained: recorded only


def kappa_cal(K, G, V, rho, n):
    """PINK Eq. (2), the same chain step 07 and step 71 use."""
    vl = ((K + 4 * G / 3) / rho) ** 0.5 * 1000
    vt = (G / rho) ** 0.5 * 1000
    vs = ((1 / vl ** 3 + 2 / vt ** 3) / 3) ** (-1 / 3)
    r = vl / vt
    nu = (r ** 2 - 2) / (2 * r ** 2 - 2)
    gam = 3 * (1 + nu) / (2 * (2 - 3 * nu))
    return G * 1e9 * vs * (V * 1e-30) ** (1 / 3) / (n * 300) * np.exp(-gam), gam


def featurize(ids):
    """297 matminer descriptors for these GNoME crystals."""
    from pymatgen.core import Structure
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import DensityFeatures, SiteStatsFingerprint
    from matminer.featurizers.site import BondOrientationalParameter

    cf = ElementProperty.from_preset("magpie")
    df_ = DensityFeatures()
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
    for mid in ids:
        e = f"by_composition/{comp.get(mid)}.CIF"
        if e not in names:
            continue
        s = Structure.from_str(z.read(e).decode(), fmt="cif")
        vals = (cf.featurize(s.composition) + df_.featurize(s)
                + cn.featurize(s) + bo.featurize(s))
        rows.append(dict(zip(cols, vals), material_id=mid))
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
    d = pd.concat([pd.read_csv(os.path.join(RESULTS, f), dtype={"material_id": str})
                   .assign(origin=o)
                   for f, o in [("69_dft_queue_kappa_real_gamma.csv", "step 61 queue"),
                                ("69_replacement_kappa_real_gamma.csv", "step 68 replacement")]],
                  ignore_index=True)

    F = featurize(list(d.material_id))
    p = predict_baseline(F)
    nb = pd.DataFrame({"material_id": F.index, "K_nb": p["K"], "G_nb": p["G"]})

    s39 = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"),
                      dtype={"material_id": str},
                      usecols=["material_id", "Volume (A3)", "Density (g cm-3)",
                               "Number of Atoms"])
    m = d.merge(nb, on="material_id").merge(s39, on="material_id")

    kp, gp = kappa_cal(m.K_nb.values, m.G_nb.values, m["Volume (A3)"].values,
                       m["Density (g cm-3)"].values, m["Number of Atoms"].values)
    m["newbase_gamma_poisson"] = gp
    # the same exact-swap identity step 69 used for the other models
    m["newbase"] = kp * np.exp(gp - m.gamma_mlip.values)

    m["ALIGNN"] = m["ALIGNN_kappa_new"]
    m["CGCNN-ens"] = m["CGCNN-ens_kappa_new"]
    m["round9"] = m["round9_kappa_new"]
    m["tree"] = m["tree_kappa_new"]

    m["kappa_max3"] = m[MB].max(axis=1)
    m["kappa_max4_old"] = m["kappa_max_new"]        # what step 70 gated on
    m["binding3"] = m[MB].idxmax(axis=1)

    is_ctl = m["label"].astype(str).str.startswith("C:", na=False)
    m["status"] = np.where(is_ctl, "CONTROL",
                           np.where(m.kappa_max3 <= THRESHOLD, "QUEUE", "DROPPED"))
    m["status_step70"] = np.where(is_ctl, "CONTROL",
                                  np.where(m.kappa_max4_old <= THRESHOLD, "QUEUE", "DROPPED"))

    m = m.sort_values(["status", "kappa_max3"])
    cols = ["status", "status_step70", "origin", "formula", "material_id",
            "gamma_mlip"] + MB + ["kappa_max3", "binding3", "kappa_max4_old"] + AF
    m[cols].to_csv(os.path.join(RESULTS, "73_matbench3_dft_queue.csv"), index=False)

    print("=" * 104)
    print("STEP 73 - DFT QUEUE ON THE THREE MATBENCH MODELS (AFLOW REMOVED FROM THE GATE)")
    print("=" * 104)
    print(m[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    changed = m[m.status != m.status_step70]
    print(f"\ncandidates: {(m.status=='QUEUE').sum()}  (step 70 had "
          f"{(m.status_step70=='QUEUE').sum()})")
    print(f"decisions that CHANGED when AFLOW was removed from the gate: {len(changed)}")
    if len(changed):
        print(changed[["formula", "status_step70", "status", "kappa_max4_old",
                       "kappa_max3", "binding3"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbinding model among the three matbench models:")
    print(m[~is_ctl].binding3.value_counts().to_string())
    print(f"\nwrote {RESULTS}/73_matbench3_dft_queue.csv")


if __name__ == "__main__":
    main()
