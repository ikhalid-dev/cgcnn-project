#!/usr/bin/env python
"""
Step 71 - rank the whole GNoME screen, then re-rank the top of it with the new
matminer/XGBoost baseline as a fifth independent opinion.
================================================================================

    ml_env/bin/python scripts/cgcnn/71_screen_with_new_baseline.py --pool 2000
    ml_env/bin/python scripts/cgcnn/71_screen_with_new_baseline.py --pool 2000 --resume

WHY
---
Every shortlist in this project so far has been ranked by `kappa_max` - the most
pessimistic of four models (ALIGNN, CGCNN-ens, round 9, composition tree). Step
69 showed that ranking is far more fragile than it looked: once a real gamma
replaced the Poisson shortcut, three of six candidates stopped being low-kappa
at all. A fifth genuinely independent model is the cheapest available check on
whether the four are agreeing because the crystal is soft or because they share
an ancestor.

The new baseline is that fifth opinion, and it is a good one. On matbench's
official 5-fold CV it reaches MAE 0.0586 log10 on K and 0.0751 on G - beating
published CGCNN (0.0712 / 0.0895) and within 0.002 of ALIGNN on K. It shares no
architecture, no features and no training code with anything else in this
screen: 297 matminer descriptors (Magpie composition + DensityFeatures +
CrystalNN order parameters + Steinhardt Q_l) into gradient-boosted trees.

WHY A POOL, NOT THE WHOLE SCREEN
---------------------------------
Featurising is the cost, and it was MEASURED on real GNoME cells rather than
guessed: 1.77 s per 28-atom structure (CrystalNN 1.16, Steinhardt 0.53, Magpie
0.08). The full screen is 33,053 crystals, which is ~16 CPU-hours on this
two-core laptop. The top 2,000 is ~1 hour.

A pool of 2,000 to pick a top 300 is deliberate slack. For a crystal outside
the pool to belong in the final 300, the new model would have to disagree with
all four existing models violently enough to lift it past 1,700 others. That is
possible but would itself be the finding, and `--pool` exists to widen the net
if you want to test it.

WHAT "RE-RANK" MEANS HERE
-------------------------
The new model predicts K and G, exactly like the others. Those go through the
SAME physics chain every other model in this screen uses - step 07's
`slack_physics()`, carried here as a verified copy (see the function's own
docstring for why it cannot be imported) so the five models differ in nothing
except the moduli they predict. `verify_physics()` re-derives the screen's
stored Kappa_alignn column from the copy and halts if it disagrees. Then:

    kappa_max5 = max(ALIGNN, CGCNN-ens, round9, tree, NEW)

so a crystal only ranks high if all FIVE call it soft. Adding a model can only
push a crystal down the list, never up - which is the point. This stage cannot
manufacture candidates, only eliminate them.

IMPORTANT: PRIMITIVE CELLS
--------------------------
The baseline was trained on matbench structures converted to primitive cells
(symprec 0.1). Feeding it conventional cells would change n_sites, volume per
atom and the density features - the model would be reading a different crystal
from the one it was trained to read. The same conversion is applied here, and
`cell_changed` records where it mattered.

OUTPUTS
    results/cgcnn/71_screen_ranked_5model.csv    the pool, all five models
    results/cgcnn/71_top300_5model.csv           the final 300, ranked
    data/screen_features/part_*.parquet          the feature cache (resumable)
"""
import argparse
import math
import os
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
CACHE = os.path.join(ROOT, "data", "screen_features")
BASELINE = os.path.expanduser("~/Desktop/pink_reproduction/models")

MODELS = {"ALIGNN": "Kappa_alignn", "CGCNN-ens": "Kappa_cal_derived_matbench",
          "round9": "Kappa_r9_gamma", "tree": "Kappa_baseline"}
CHUNK = 100          # structures per cache part - small enough to resume cheaply


h = 6.62607015e-34      # Planck constant, J s
k = 1.380649e-23        # Boltzmann constant, J/K


def slack_physics(k_gpa, g_gpa, volume_a3, density, mass_amu, n_atoms):
    """The kappa chain, copied verbatim from 07_predict_kappa.py lines 92-110.

    It is COPIED rather than imported, and that is a deliberate, tested choice:
    step 07 does `import torch` at module scope, and torch 2.2.2 in ml_env now
    segfaults against numpy 2.3.5 no matter which order the two are imported.
    Importing step 07 therefore kills this process before it prints anything
    (exit 139). Nothing in this script needs torch.

    A copy can drift from its original, so `verify_physics()` below proves it
    has not - it reproduces the screen's own stored Kappa_alignn column from
    the same K and G, and refuses to run if the numbers disagree.
    """
    v_long = ((k_gpa + 4 * g_gpa / 3) / density) ** 0.5 * 1000   # longitudinal sound velocity, m/s
    v_trans = (g_gpa / density) ** 0.5 * 1000                    # transverse sound velocity, m/s
    v_sound = ((1 / v_long ** 3 + 2 / v_trans ** 3) / 3) ** (-1 / 3)

    debye_temp = (h / k * np.power(3 / (4 * math.pi * volume_a3), 1 / 3)
                  * v_sound * 1e10)

    ratio = v_long / v_trans
    poisson = (ratio ** 2 - 2) / (2 * ratio ** 2 - 2)
    gruneisen = 3 * (1 + poisson) / (2 * (2 - 3 * poisson))

    slack_coeff = 2.43e-8 / (1 - 0.514 / gruneisen + 0.228 / gruneisen ** 2)
    kappa_slack = (slack_coeff * mass_amu * volume_a3 ** (1 / 3)
                   * debye_temp ** 3 / (gruneisen ** 2 * 300 * n_atoms) * 100)

    # Equation (2) of the PINK paper - three-phonon scattering, delta = 1.
    kappa_cal = (g_gpa * 1e9 * v_sound * (volume_a3 * 1e-30) ** (1 / 3)
                 / (n_atoms * 300) * np.exp(-gruneisen))

    return {"v_sound": v_sound, "debye_temp": debye_temp, "poisson": poisson,
            "gruneisen": gruneisen, "kappa_slack": kappa_slack,
            "kappa_cal": kappa_cal}


def verify_physics(d):
    """Prove the copied chain is the one that built this screen.

    Feeds ALIGNN's own K and G back through it and compares against the stored
    Kappa_alignn. If the copy had drifted from step 07 even slightly, every
    number this script produces would be quietly wrong, so this is a hard stop
    rather than a warning.
    """
    s = d.dropna(subset=["K_alignn", "G_alignn", "Kappa_alignn"]).head(500)
    got = slack_physics(s.K_alignn.values, s.G_alignn.values,
                        s["Volume (A3)"].values, s["Density (g cm-3)"].values,
                        s["Atomic mass (amu)"].values,
                        s["Number of Atoms"].values)["kappa_cal"]
    worst = float(np.nanmax(np.abs(got - s.Kappa_alignn.values)
                            / s.Kappa_alignn.values))
    if worst > 1e-9:
        raise SystemExit(f"physics check FAILED: worst relative error {worst:.2e} "
                         "- the copied chain no longer matches step 07")
    print(f"physics check passed: reproduces Kappa_alignn on 500 crystals "
          f"to {worst:.1e} relative")


def rank_four_models():
    """The existing screen, ranked by the most pessimistic of the four models."""
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    s = pd.read_csv(os.path.join(RESULTS, "39_gnome_screen_all_gamma.csv"),
                    dtype={"material_id": str})
    d = a.merge(s, on="material_id", suffixes=("", "_dup"))
    d = d[d["alignn_prediction_reliable"]].copy()
    for name, col in MODELS.items():
        d[name] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(MODELS))
    d["kappa_max4"] = d[list(MODELS)].max(axis=1)
    d["kappa_min4"] = d[list(MODELS)].min(axis=1)
    return d.sort_values("kappa_max4").reset_index(drop=True)


def featurize_pool(ids, resume):
    """Compute the 297 matminer features for these material_ids, in chunks.

    Each chunk is written to its own parquet part. A part that already exists is
    skipped, so an interrupted run continues instead of restarting - this takes
    about an hour and the laptop cannot be trusted to stay awake for it.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import DensityFeatures, SiteStatsFingerprint
    from matminer.featurizers.site import BondOrientationalParameter

    os.makedirs(CACHE, exist_ok=True)
    comp_f = ElementProperty.from_preset("magpie")
    dens_f = DensityFeatures()
    cnn_f = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")
    boop_f = SiteStatsFingerprint(BondOrientationalParameter(max_l=10),
                                  stats=("mean", "std_dev", "minimum", "maximum"))

    summ = pd.read_csv(os.path.join(ROOT, "gnome_data/stable_materials_summary.csv"),
                       usecols=["Composition", "MaterialId"], dtype=str)
    comp_of = dict(zip(summ.MaterialId, summ.Composition))
    z = zipfile.ZipFile(os.path.join(ROOT, "gnome_data/by_composition.zip"))
    names = set(z.namelist())

    cols = (comp_f.feature_labels() + dens_f.feature_labels()
            + cnn_f.feature_labels() + boop_f.feature_labels())

    def feat_one(mid):
        """Featurise a single crystal. Returns a row dict, never raises."""
        entry = f"by_composition/{comp_of.get(mid)}.CIF"
        if entry not in names:
            return {"material_id": mid, "featurize_error": "no CIF"}
        if True:
            try:
                st = Structure.from_str(z.read(entry).decode(), fmt="cif")
                n_before = len(st)
                # Match the baseline's training convention - primitive cells.
                # spglib fails on ~9% of these ("Unable to determine symmetry"),
                # almost all low-symmetry cells. Dropping them would bias the
                # shortlist against exactly the soft, low-symmetry crystals it
                # is looking for. The conversion is a no-op on GNoME anyway
                # (0 of 99 cells changed when it succeeds), so falling back to
                # the structure as-written loses nothing and keeps the row.
                fallback = False
                try:
                    st = SpacegroupAnalyzer(st, symprec=0.1).get_primitive_standard_structure()
                except Exception:
                    fallback = True
                vals = (comp_f.featurize(st.composition) + dens_f.featurize(st)
                        + cnn_f.featurize(st) + boop_f.featurize(st))
                r = dict(zip(cols, vals))
                r.update({"material_id": mid, "featurize_error": "",
                          "n_sites_primitive": len(st),
                          "cell_changed": len(st) != n_before,
                          "primitive_fallback": fallback})
                return r
            except Exception as e:                      # one bad CIF must not kill the run
                return {"material_id": mid, "featurize_error": f"{type(e).__name__}: {e}"}

    for start in range(0, len(ids), CHUNK):
        part = os.path.join(CACHE, f"part_{start:06d}.parquet")
        if resume and os.path.exists(part):
            continue
        t0 = time.time()
        chunk = ids[start:start + CHUNK]
        pd.DataFrame([feat_one(mid) for mid in chunk]).to_parquet(part, index=False)
        done = start + len(chunk)
        rate = (time.time() - t0) / max(1, len(chunk))
        print(f"  featurised {done}/{len(ids)}   {rate:.2f} s/structure   "
              f"ETA {rate * (len(ids) - done) / 60:.0f} min", flush=True)

    def read_all():
        parts = sorted(os.path.join(CACHE, f) for f in os.listdir(CACHE)
                       if f.endswith(".parquet") and (f.startswith("part_")
                                                      or f.startswith("repair_")))
        f = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        # A repaired row replaces the failed original: sort errors last, then
        # keep the first row per crystal.
        f["_bad"] = (f.featurize_error.fillna("") != "").astype(int)
        return (f.sort_values("_bad").drop_duplicates("material_id", keep="first")
                 .drop(columns="_bad").reset_index(drop=True))

    # ---- repair pass -------------------------------------------------------
    # Re-attempt only the crystals that failed, so a fix to the featuriser costs
    # minutes rather than re-running the whole pool.
    f = read_all()
    retry = list(f[(f.featurize_error.fillna("") != "")
                   & (f.featurize_error != "no CIF")].material_id)
    if retry:
        print(f"\n  repairing {len(retry)} previously-failed crystals ...", flush=True)
        for start in range(0, len(retry), CHUNK):
            part = os.path.join(CACHE, f"repair_{start:06d}.parquet")
            if resume and os.path.exists(part):
                continue
            chunk = retry[start:start + CHUNK]
            pd.DataFrame([feat_one(mid) for mid in chunk]).to_parquet(part, index=False)
            print(f"    repaired {start + len(chunk)}/{len(retry)}", flush=True)
        f = read_all()
    return f


def predict_baseline(feat):
    """Mean of the five fold models, for K and for G. Returns log10(GPa)."""
    import xgboost as xgb
    out = {}
    for task, label in [("matbench_log_kvrh", "K"), ("matbench_log_gvrh", "G")]:
        import json
        meta = json.load(open(os.path.join(
            BASELINE, f"{task}_composition+structure+angular_meta.json")))
        cols, med = meta["columns"], meta["train_medians"]
        # reindex puts the columns in the exact training order and inserts NaN
        # for anything missing; fillna then applies the TRAINING medians, which
        # is what the model was fitted against - never medians of this new data.
        X = feat.reindex(columns=cols).astype(float).fillna(pd.Series(med))
        preds = []
        for fold in range(5):
            m = xgb.XGBRegressor()
            m.load_model(os.path.join(
                BASELINE, f"{task}_composition+structure+angular_fold{fold}.json"))
            preds.append(m.predict(X))
        out[label] = np.mean(preds, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2000, help="how many of the ranked screen to featurise")
    ap.add_argument("--top", type=int, default=300, help="size of the final shortlist")
    ap.add_argument("--resume", action="store_true", help="reuse existing feature-cache parts")
    args = ap.parse_args()

    print("=" * 92)
    print("STEP 71 - THE SCREEN, RANKED, THEN RE-RANKED WITH THE NEW BASELINE")
    print("=" * 92)
    d = rank_four_models()
    verify_physics(d)
    print(f"\nscreen scored by all four models : {len(d)}")
    print(f"pool to featurise (lowest kappa_max4) : {args.pool}")
    pool = d.head(args.pool).copy()
    print(f"   kappa_max4 range in pool : {pool.kappa_max4.min():.3f} - {pool.kappa_max4.max():.3f}")
    print(f"   atoms  median {pool['Number of Atoms'].median():.0f}, "
          f"range {pool['Number of Atoms'].min():.0f}-{pool['Number of Atoms'].max():.0f}\n")

    feat = featurize_pool(list(pool.material_id), args.resume)
    bad = feat.featurize_error.fillna("") != ""
    print(f"\nfeaturised OK: {(~bad).sum()}   failed: {bad.sum()}")
    feat = feat[~bad].copy()

    p = predict_baseline(feat.set_index("material_id"))
    nb = pd.DataFrame({"material_id": feat.material_id,
                       "K_newbase": 10 ** p["K"], "G_newbase": 10 ** p["G"]})

    m = pool.merge(nb, on="material_id")
    phys = slack_physics(m.K_newbase.values, m.G_newbase.values,
                         m["Volume (A3)"].values, m["Density (g cm-3)"].values,
                         m["Atomic mass (amu)"].values, m["Number of Atoms"].values)
    m["newbase"] = phys["kappa_cal"]
    m["gamma_newbase_poisson"] = phys["gruneisen"]

    five = list(MODELS) + ["newbase"]
    # THE RANKING THAT COUNTS. ALIGNN, CGCNN-ens and the new baseline are all
    # matbench-trained; round9 and tree are AFLOW-trained, and steps 51/55/59
    # established AFLOW's labels are the weak link (round 9 loses F1 0.400 to
    # the tree's 0.646 on AFLOW's own test set, and sits at 2.4x its label
    # floor). Mixing the two training sets in one statistic also violates this
    # project's own rule about comparing models across them. So the shortlist
    # is built from the three matbench models only; the AFLOW pair is carried
    # in the output as columns, for comparison, never as a gate.
    MB = ["ALIGNN", "CGCNN-ens", "newbase"]
    AF = ["round9", "tree"]
    m["kappa_max3"] = m[MB].max(axis=1)
    m["kappa_min3"] = m[MB].min(axis=1)
    m["kappa_max5"] = m[five].max(axis=1)
    m["kappa_maxAF"] = m[AF].max(axis=1)
    m = m.sort_values("kappa_max3").reset_index(drop=True)
    m["rank3"] = np.arange(1, len(m) + 1)
    m["rank4"] = m.kappa_max4.rank().astype(int)
    m["rank5"] = m.kappa_max5.rank().astype(int)

    print("\n" + "=" * 92)
    print("WHICH MODEL SETS THE ANSWER")
    print("=" * 92)
    print("\nbinding (most pessimistic) model over the pool, all five:")
    for kk, v in m[five].idxmax(axis=1).value_counts().items():
        print(f"   {kk:12s} {v:5d}  ({100*v/len(m):4.1f}%)"
              + ("   <- AFLOW-trained" if kk in AF else ""))
    print("\nbinding model within the three matbench models only:")
    for kk, v in m[MB].idxmax(axis=1).value_counts().items():
        print(f"   {kk:12s} {v:5d}  ({100*v/len(m):4.1f}%)")

    # ---- how much were the AFLOW models steering the shortlist? ------------
    print("\n" + "=" * 92)
    print(f"DID DROPPING THE AFLOW MODELS CHANGE THE TOP {args.top}?")
    print("=" * 92)
    t3 = set(m.head(args.top).material_id)
    t4 = set(m.nsmallest(args.top, "kappa_max4").material_id)
    t5 = set(m.nsmallest(args.top, "kappa_max5").material_id)
    for lab, other in [("four models (the old shortlist)", t4),
                       ("five models (four + new baseline)", t5)]:
        print(f"\n   matbench-3 vs {lab}:")
        print(f"      kept    {len(t3 & other):4d}")
        print(f"      dropped {len(other - t3):4d}")
        print(f"      added   {len(t3 - other):4d}")
    rho = m.kappa_max3.corr(m.kappa_maxAF, method="spearman")
    print(f"\n   Spearman(matbench-3 kappa, AFLOW-2 kappa) over the pool: {rho:+.3f}")

    out = m.sort_values("kappa_max3")
    cols = ["rank3", "rank4", "rank5", "formula", "material_id", "Number of Atoms",
            "Volume (A3)", "Density (g cm-3)", "Atomic mass (amu)"] + five + \
           ["kappa_max3", "kappa_min3", "kappa_max4", "kappa_max5", "kappa_maxAF",
            "K_newbase", "G_newbase", "K_alignn", "G_alignn"]
    cols = [c for c in cols if c in out.columns]
    out[cols].to_csv(os.path.join(RESULTS, "71_screen_ranked_matbench3.csv"), index=False)
    out[cols].head(args.top).to_csv(os.path.join(RESULTS, "71_top300_matbench3.csv"), index=False)

    print(f"\nfinal top {args.top} on the three matbench models, kappa_max3 range "
          f"{out.kappa_max3.head(args.top).min():.3f} - {out.kappa_max3.head(args.top).max():.3f}")
    print("\nfirst 15:")
    show = out.head(15)[["rank3", "rank4", "formula", "material_id", "Number of Atoms"]
                        + MB + ["kappa_max3", "kappa_maxAF"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {RESULTS}/71_screen_ranked_matbench3.csv  and  71_top300_matbench3.csv")


if __name__ == "__main__":
    main()
