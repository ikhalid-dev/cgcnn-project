#!/usr/bin/env python
"""
Step 87 - when the screen says "low", does DFT kappa_L agree?  (Poisson gamma, no MLIP)
======================================================================================

Three runs, two Python environments, in this order:

    # 1. ALIGNN + CGCNN-ens moduli  (~10 min; needs torch, which only works in infer_env)
    ~/miniconda3/envs/infer_env/bin/python scripts/cgcnn/87_phonix_poisson_calibration.py --stage torch
    # 2. newbase moduli  (~1 h; matminer + xgboost; resumable - just run it again if stopped)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/87_phonix_poisson_calibration.py --stage newbase
    # 3. kappa, confusion matrices, figure  (seconds)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/87_phonix_poisson_calibration.py --stage score

WHY TWO ENVIRONMENTS
--------------------
ml_env's torch 2.2.2 was built for numpy 1.x. numpy there is now 2.3.5, and
loading an ALIGNN model fails with "RuntimeError: Numpy is not available".
infer_env (made 2026-10-05) is a copy of mlip_env (numpy 1.26.4 with the same
torch) plus alignn and jarvis-tools at ml_env's exact versions. Re-predicting
6 GNoME crystals there gave the stored ALIGNN and CGCNN K and G back to every
printed digit, twice. The newbase model needs no torch, so it stays in ml_env.

WHY
---
The DFT run will compute kappa_L. Before spending DFT time, the question is:
when the screen calls a crystal "low" (kappa <= 1 W/m/K at 300 K), how often is
its DFT kappa_L really low?

PhoNIX holds DFT kappa_L for 6,641 Materials Project crystals. Step 85 found
4,994 of them are NOT in matbench, so none of our three models has seen them -
a fair exam. Kept here: the 2,617 with <= 20 atoms, the same cap as the DFT list.

This version uses the OLD gamma: each model's own Poisson gamma, from its own
predicted K and G (gamma = 3(1+nu)/(2(2-3nu))). It is the screen exactly as it
was before the MLIP gamma existed. The MLIP-gamma version comes next and has to
beat this one.

WHAT IS COMPARED
----------------
prediction   kappa_max3 = the HIGHEST of the three models' kappa (the screen's rule)
truth        PhoNIX klat = kp + kc, and kp alone. kp is what a standard
             Boltzmann-transport run returns; kc (the wave-like part) needs the
             Wigner formulation and only matters in very low-kappa crystals.

THE NUMBER THAT MATTERS is precision: of the crystals the screen calls low,
the fraction DFT also finds low. Read it against the base rate - 598 of the
2,617 (23%) are low anyway, so a blind pick is right 23% of the time.

WRITTEN BEFORE THE RESULT (2026-10-05)
--------------------------------------
precision at 1.0 >= 0.70   the Poisson screen already picks low crystals
                           reliably; MLIP gamma has a high bar to clear
0.40 - 0.70                useful, not safe - the precision-vs-cutoff curve
                           says how far below 1 a prediction must sit
< 0.40                     the Poisson screen over-calls "low"; the DFT list
                           then rests on the MLIP-gamma calibration

CHECKED, NOT TRUSTED
--------------------
* stages 1 and 2 each start by re-predicting GNoME crystals whose answers are
  stored from the original screen (steps 13, 57, 71), and stop if they differ
  (CGCNN and newbase to 1e-4; ALIGNN to 1% - see ALIGNN_TOL for why)
* stage 3 uses step 71's kappa formula, after step 71's own check that it
  reproduces the screen's stored kappa column

OUTPUTS
    results/cgcnn/87_torch_moduli.csv               stage 1
    results/cgcnn/87_newbase_moduli.csv             stage 2 (feature cache: data/phonix_features/)
    results/cgcnn/87_phonix_poisson_predictions.csv one row per crystal: K, G, gamma, kappa per model, DFT
    results/cgcnn/87_phonix_poisson_confusion.csv   confusion matrix + rates, per model, cutoff, truth
    results/cgcnn/87_precision_vs_cutoff.csv        "how far below 1 must the prediction be?"
    results/cgcnn/87_phonix_poisson_calibration.png
"""
import os
import sys

# Two things must happen BEFORE numpy is imported, and which one depends on the
# stage. sys.argv is the list of words typed after "python", e.g.
# ["87_phonix_poisson_calibration.py", "--stage", "torch"].
if "torch" in " ".join(sys.argv[1:]):
    import torch  # noqa: F401  -- torch first, or its OpenMP clashes with numpy's
else:
    # xgboost crashes on this machine with more than one OpenMP thread
    # (cgcnn-pink-environment memory); torch is not loaded in these stages
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse                 # reads "--stage torch" from the command line
import ast                      # turns PhoNIX's structure text back into a dict
import time
import warnings
import zipfile
from importlib import import_module   # imports a file whose name starts with a digit

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")      # pymatgen and matminer print a lot of noise
from pymatgen.core import Lattice, Structure   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path is where Python looks for modules; adding these folders lets the
# script reuse steps 04, 15, 71 and 82 instead of copying their code
for p in [ROOT, os.path.join(ROOT, "scripts", "cgcnn"), os.path.join(ROOT, "scripts", "alignn")]:
    sys.path.insert(0, p)

RES = os.path.join(ROOT, "results", "cgcnn")
PHONIX = os.path.expanduser("~/Desktop/a new project/data/phonix/data_all.csv")
GNOME_SUMMARY = os.path.join(ROOT, "gnome_data", "stable_materials_summary.csv")
GNOME_ZIP = os.path.join(ROOT, "gnome_data", "by_composition.zip")
FEAT_CACHE = os.path.join(ROOT, "data", "phonix_features")

MAX_ATOMS = 20          # the DFT budget
LOW, VERY_LOW = 1.0, 0.5
MIN_MODULUS = 0.01      # GPa. Step 57's rule: an ALIGNN K or G below this is not a prediction
CHUNK = 100             # newbase: crystals per cache file, so a stop loses at most 100
N_CHECK = 8             # GNoME crystals re-predicted at the start of each model stage
# ALIGNN keeps each atom's 12 nearest neighbours. When the 12th and 13th sit at
# the same distance, which one is kept depends on numpy's sort order, and that
# differs between environments. Measured 2026-10-05: exact on crystals without
# such a tie, off by 0.25% and 0.66% on two crystals with one. 1% is ~25x below
# ALIGNN's own typical error, so it is the tolerance for ALIGNN; CGCNN and
# newbase have no such tie and must match to 1e-4.
ALIGNN_TOL = 1e-2


# =============================================================================
#  shared: the pool of unseen PhoNIX crystals
# =============================================================================
def load_pool():
    """PhoNIX crystals no model has seen, <= 20 atoms, with structures and DFT kappa.

    Step 85 already collapsed repeated calculations to the median and labelled
    each crystal's split; its CSV just has no structures (they are big), so the
    structures are read again from data_all.csv, the first copy per material -
    the same copy step 85 matched.
    """
    m = pd.read_csv(os.path.join(RES, "85_phonix_matbench_match.csv"))
    pool = m[(m.split == "not in mb") & (m.n_atoms <= MAX_ATOMS)].copy()

    raw = pd.read_csv(PHONIX, usecols=["mp_id", "structure"])
    # drop_duplicates keeps the FIRST row of each mp_id, like step 85's "first"
    text_of = dict(raw.drop_duplicates("mp_id").set_index("mp_id").structure)

    structures = []
    for mp in pool.mp_id:
        s = ast.literal_eval(text_of[mp])
        structures.append(Structure(Lattice(s["cell"]), s["numbers"], s["positions"],
                                    coords_are_cartesian=True))
    pool["structure"] = structures

    # what the Slack formula needs, from the primitive cell (PhoNIX cells already are)
    pool["volume_a3"] = [s.volume for s in structures]
    pool["density"] = [s.density for s in structures]                      # g/cm^3
    pool["mass_amu"] = [float(s.composition.weight) for s in structures]   # whole cell
    assert (pool.n_atoms == [len(s) for s in structures]).all()
    rel = (pool.volume_a3 / pool.volume - 1).abs().max()
    assert rel < 1e-3, f"structure volume disagrees with PhoNIX's volume column ({rel:.1e})"

    print(f"pool: {len(pool)} PhoNIX crystals, not in matbench, <= {MAX_ATOMS} atoms; "
          f"DFT klat <= {LOW}: {(pool.klat <= LOW).sum()}")
    return pool.reset_index(drop=True)


def gnome_check_set(d, n):
    """n GNoME crystals from table d (stored predictions), plus their structures from the archive."""
    s = pd.read_csv(GNOME_SUMMARY, usecols=["MaterialId", "Composition"], dtype=str)
    d = d.merge(s, left_on="material_id", right_on="MaterialId").sample(n, random_state=0)
    z = zipfile.ZipFile(GNOME_ZIP)
    d["structure"] = [Structure.from_str(z.read(f"by_composition/{c}.CIF").decode(), fmt="cif")
                      for c in d.Composition]
    return d.reset_index(drop=True)


def same(a, b, tol=1e-4):
    """True when every value in a matches b to within tol (relative)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.abs(a - b) <= tol * np.abs(b)))


# =============================================================================
#  stage 1: ALIGNN and CGCNN-ens  (infer_env)
# =============================================================================
def stage_torch():
    import torch
    from jarvis.core.atoms import pmg_to_atoms
    from cgcnn_scratch.data import AtomFeaturiser, GaussianDistance, structure_to_graph
    _al = import_module("15_alignn_predict_moduli")     # load_alignn_target, predict_pair
    _pred = import_module("04_predict_moduli")          # load_model, predict_ensemble

    dev = torch.device("cpu")
    k_al, cfg = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_bulk_modulus_kv"), dev)
    g_al, _ = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_shear_modulus_gv"), dev)
    # CGCNN-ens = three CGCNN models per modulus, averaged (step 13's screen)
    k_cg = [_pred.load_model(t, RES)[:2] for t in ["K_VRH_full", "K_VRH_s1", "K_VRH_s2"]]
    g_cg = [_pred.load_model(t, RES)[:2] for t in ["G_VRH_full", "G_VRH_s1", "G_VRH_s2"]]
    ari = AtomFeaturiser(os.path.join(ROOT, "cgcnn_scratch", "atom_init.json"))
    gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)    # step 13's settings: radius 8, step 0.2

    def predict(structures, ids):
        """K and G in GPa from both models, one row per crystal."""
        rows, graphs, graph_ids = [], [], []
        start = time.time()
        for k, (st, i) in enumerate(zip(structures, ids)):
            row = {"id": i, "error": ""}
            try:
                row["K_alignn"], row["G_alignn"] = _al.predict_pair(k_al, g_al, pmg_to_atoms(st), cfg, dev)
                graphs.append(structure_to_graph(st, ari, gdf, 12, 8))
                graph_ids.append(i)
            except Exception as e:                       # one odd crystal must not stop the run
                row["error"] = f"{type(e).__name__}: {e}"[:120]
            rows.append(row)
            if (k + 1) % 250 == 0:
                rate = (time.time() - start) / (k + 1)
                print(f"  {k + 1}/{len(ids)}  {rate:.2f} s/crystal  "
                      f"ETA {rate * (len(ids) - k - 1) / 60:.0f} min", flush=True)
        out = pd.DataFrame(rows)
        ds = _pred.PredictionSet(graphs, graph_ids)
        K, _ = _pred.predict_ensemble(k_cg, ds)          # dicts: id -> GPa
        G, _ = _pred.predict_ensemble(g_cg, ds)
        out["K_cgcnn"] = out.id.map(K)                   # .map looks each id up in the dict
        out["G_cgcnn"] = out.id.map(G)
        return out

    # ---- check: the stored GNoME answers come back -------------------------
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    a = gnome_check_set(a[a.alignn_prediction_reliable], N_CHECK)   # stored K, G ~0 would prove nothing
    c = pd.read_csv(os.path.join(RES, "13_gnome_screen_all.csv"), dtype={"material_id": str})
    c = c.drop_duplicates("material_id", keep=False)   # 175 ids appear twice there - skip them all
    a = a.merge(c[["material_id", "K_VRH_pred", "G_VRH_pred"]], on="material_id")
    got = predict(a.structure, a.material_id)
    ok = (same(got.K_alignn, a.K_alignn, ALIGNN_TOL) and same(got.G_alignn, a.G_alignn, ALIGNN_TOL)
          and same(got.K_cgcnn, a.K_VRH_pred) and same(got.G_cgcnn, a.G_VRH_pred))
    # relative difference, new / stored - 1, so the size of any mismatch is visible
    diff = pd.DataFrame({"formula": a.formula,
                         "ALIGNN_K": got.K_alignn / a.K_alignn - 1, "ALIGNN_G": got.G_alignn / a.G_alignn - 1,
                         "CGCNN_K": got.K_cgcnn / a.K_VRH_pred - 1, "CGCNN_G": got.G_cgcnn / a.G_VRH_pred - 1})
    print(diff.to_string(index=False, float_format=lambda v: f"{v:+.1e}"))
    if not ok:
        raise SystemExit("CHECK FAILED: the models no longer reproduce the stored GNoME screen")
    print(f"check passed: {len(a)} GNoME crystals re-predicted; CGCNN-ens matches the stored "
          f"screen to 1e-4, ALIGNN to {ALIGNN_TOL:.0%}\n")

    # ---- the PhoNIX pool ----------------------------------------------------
    pool = load_pool()
    out = predict(pool.structure, pool.mp_id).rename(columns={"id": "mp_id"})
    out.to_csv(os.path.join(RES, "87_torch_moduli.csv"), index=False)
    print(f"\nwrote 87_torch_moduli.csv: {len(out)} rows, {(out.error != '').sum()} failed")


# =============================================================================
#  stage 2: newbase  (ml_env)
# =============================================================================
def make_featuriser():
    """Step 71's 297 matminer features, as one function that takes a Structure.

    Step 71's own version takes a GNoME id and reads the archive itself, so it
    cannot be handed a PhoNIX structure. The recipe below is the same - same
    four featurisers, same presets, same primitive-cell conversion - and the
    stage's opening check proves it by reproducing step 71's stored predictions.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import DensityFeatures, SiteStatsFingerprint
    from matminer.featurizers.site import BondOrientationalParameter

    comp_f = ElementProperty.from_preset("magpie")
    dens_f = DensityFeatures()
    cnn_f = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")
    boop_f = SiteStatsFingerprint(BondOrientationalParameter(max_l=10),
                                  stats=("mean", "std_dev", "minimum", "maximum"))
    cols = (comp_f.feature_labels() + dens_f.feature_labels()
            + cnn_f.feature_labels() + boop_f.feature_labels())

    def feat_one(st, ident):
        try:
            fallback = False
            try:   # the baseline was trained on primitive standard cells (step 71's note)
                st = SpacegroupAnalyzer(st, symprec=0.1).get_primitive_standard_structure()
            except Exception:
                fallback = True                          # keep the cell as written
            vals = (comp_f.featurize(st.composition) + dens_f.featurize(st)
                    + cnn_f.featurize(st) + boop_f.featurize(st))
            row = dict(zip(cols, vals))                  # zip pairs each name with its value
            row.update({"id": ident, "featurize_error": "", "primitive_fallback": fallback})
            return row
        except Exception as e:
            return {"id": ident, "featurize_error": f"{type(e).__name__}: {e}"[:120]}

    return feat_one


def stage_newbase():
    _s71 = import_module("71_screen_with_new_baseline")  # predict_baseline: 5 xgboost folds per modulus
    feat_one = make_featuriser()

    # ---- check: step 71's stored newbase answers come back -----------------
    chk = pd.read_csv(os.path.join(RES, "71_screen_ranked_5model.csv"), dtype={"material_id": str})
    chk = gnome_check_set(chk.dropna(subset=["K_newbase", "G_newbase"]), N_CHECK)
    f = pd.DataFrame([feat_one(st, i) for st, i in zip(chk.structure, chk.material_id)])
    p = _s71.predict_baseline(f.set_index("id"))         # log10(GPa)
    if not (same(10 ** p["K"], chk.K_newbase) and same(10 ** p["G"], chk.G_newbase)):
        print(pd.DataFrame({"K": 10 ** p["K"], "K_stored": chk.K_newbase,
                            "G": 10 ** p["G"], "G_stored": chk.G_newbase}))
        raise SystemExit("CHECK FAILED: the featuriser no longer reproduces step 71")
    print(f"check passed: {len(chk)} GNoME crystals re-featurised, newbase K and G "
          f"match step 71 to 1e-4\n")

    # ---- the PhoNIX pool, in resumable chunks -------------------------------
    pool = load_pool()
    os.makedirs(FEAT_CACHE, exist_ok=True)
    for start in range(0, len(pool), CHUNK):
        part = os.path.join(FEAT_CACHE, f"part_{start:05d}.parquet")
        if os.path.exists(part):                         # done in an earlier run
            continue
        t0 = time.time()
        chunk = pool.iloc[start:start + CHUNK]           # .iloc picks rows by position
        rows = [feat_one(st, mp) for st, mp in zip(chunk.structure, chunk.mp_id)]
        pd.DataFrame(rows).to_parquet(part, index=False)
        done = start + len(chunk)
        rate = (time.time() - t0) / len(chunk)
        print(f"  featurised {done}/{len(pool)}  {rate:.2f} s/crystal  "
              f"ETA {rate * (len(pool) - done) / 60:.0f} min", flush=True)

    parts = sorted(os.listdir(FEAT_CACHE))
    f = pd.concat([pd.read_parquet(os.path.join(FEAT_CACHE, x)) for x in parts], ignore_index=True)
    f = f[f.id.isin(pool.mp_id)]                         # ignore leftovers from a different pool
    bad = f.featurize_error.fillna("") != ""
    p = _s71.predict_baseline(f[~bad].set_index("id"))
    out = pd.DataFrame({"mp_id": f[~bad].id.values,
                        "K_newbase": 10 ** p["K"], "G_newbase": 10 ** p["G"],
                        "primitive_fallback": f[~bad].primitive_fallback.values})
    out.to_csv(os.path.join(RES, "87_newbase_moduli.csv"), index=False)
    print(f"\nwrote 87_newbase_moduli.csv: {len(out)} predicted, {bad.sum()} failed to featurise")


# =============================================================================
#  stage 3: score  (ml_env)
# =============================================================================
MODELS = {"ALIGNN": ("K_alignn", "G_alignn"),
          "CGCNN-ens": ("K_cgcnn", "G_cgcnn"),
          "newbase": ("K_newbase", "G_newbase")}


def stage_score():
    _s71 = import_module("71_screen_with_new_baseline")   # slack_physics, verify_physics
    _s82 = import_module("82_confusion_and_trust")        # confusion, wilson
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _s71.verify_physics(_s71.rank_four_models())    # the formula is the screen's, or stop

    pool = load_pool().drop(columns="structure")
    tm = pd.read_csv(os.path.join(RES, "87_torch_moduli.csv"))
    nb = pd.read_csv(os.path.join(RES, "87_newbase_moduli.csv"))
    d = pool.merge(tm, on="mp_id", how="left").merge(nb, on="mp_id", how="left")

    # ---- kappa per model, each with its OWN Poisson gamma ---------------------
    for name, (kc, gc) in MODELS.items():
        phys = _s71.slack_physics(d[kc].values, d[gc].values, d.volume_a3.values,
                                  d.density.values, d.mass_amu.values, d.n_atoms.values)
        d[name] = phys["kappa_cal"]                      # W/m/K at 300 K
        d[f"gamma_{name}"] = phys["gruneisen"]

    # the screen dropped crystals where ALIGNN's K or G came out ~0 (step 57),
    # so the same crystals are dropped here
    d["alignn_reliable"] = (d.K_alignn > MIN_MODULUS) & (d.G_alignn > MIN_MODULUS)
    names = list(MODELS)
    usable = d.alignn_reliable & d[names].notna().all(axis=1) & np.isfinite(d[names]).all(axis=1)
    print(f"\nscored by all three models: {usable.sum()} of {len(d)}  "
          f"(ALIGNN unreliable {(~d.alignn_reliable & d.K_alignn.notna()).sum()}, "
          f"a model failed {(d[names].isna().any(axis=1)).sum()})")
    s = d[usable].copy()
    s["kappa_max3"] = s[names].max(axis=1)               # the screen's pessimistic rule
    s["kappa_min3"] = s[names].min(axis=1)
    s["spread3"] = s.kappa_max3 / s.kappa_min3
    s["ratio_dft_over_max3"] = s.klat / s.kappa_max3     # > 1: DFT higher than the screen said

    keep = ["mp_id", "formula", "n_atoms", "klat", "kp", "kc",
            "K_alignn", "G_alignn", "K_cgcnn", "G_cgcnn", "K_newbase", "G_newbase",
            "gamma_ALIGNN", "gamma_CGCNN-ens", "gamma_newbase",
            "ALIGNN", "CGCNN-ens", "newbase", "kappa_max3", "kappa_min3", "spread3",
            "ratio_dft_over_max3"]
    s[keep].to_csv(os.path.join(RES, "87_phonix_poisson_predictions.csv"), index=False)

    # ---- confusion matrices ---------------------------------------------------
    rows = []
    for truth in ["klat", "kp"]:
        for thr in [LOW, VERY_LOW]:
            for name in names + ["kappa_max3"]:
                c = _s82.confusion(s[name], s[truth], thr)
                rho = spearmanr(np.log10(s[name]), np.log10(s[truth])).correlation
                rows.append({"truth": truth, "cutoff": thr, "model": name, "n": len(s),
                             "base_rate": float((s[truth] <= thr).mean()), **c,
                             "spearman_rho": rho,
                             "median_dft_over_pred": float(np.median(s[truth] / s[name]))})
    conf = pd.DataFrame(rows)
    conf.to_csv(os.path.join(RES, "87_phonix_poisson_confusion.csv"), index=False)

    show = conf[["truth", "cutoff", "model", "TP", "FP", "FN", "TN", "base_rate", "precision",
                 "precision_lo95", "precision_hi95", "recall", "accuracy", "always_no_accuracy",
                 "spearman_rho", "median_dft_over_pred"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- how far below 1 must the prediction be? --------------------------------
    # For each cutoff c: take the crystals the screen puts at or below c, and
    # ask what fraction of them DFT finds at or below 1.0.
    curve = []
    for name in names + ["kappa_max3"]:
        for c in np.round(np.geomspace(0.1, 3.0, 25), 4):   # 25 cutoffs, evenly spaced on a log axis
            sel = s[s[name] <= c]
            k = int((sel.klat <= LOW).sum())
            lo, hi = _s82.wilson(k, len(sel))
            curve.append({"model": name, "pred_cutoff": c, "n_selected": len(sel),
                          "n_dft_low": k, "frac_dft_low": k / len(sel) if len(sel) else np.nan,
                          "lo95": lo, "hi95": hi,
                          "frac_dft_below_2": float((sel.klat <= 2.0).mean()) if len(sel) else np.nan})
    curve = pd.DataFrame(curve)
    curve.to_csv(os.path.join(RES, "87_precision_vs_cutoff.csv"), index=False)
    m3 = curve[curve.model == "kappa_max3"]
    print("\nmax3: of the crystals predicted <= c, the fraction with DFT klat <= 1 (and <= 2)")
    print(m3[["pred_cutoff", "n_selected", "n_dft_low", "frac_dft_low", "lo95", "hi95",
              "frac_dft_below_2"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    called = s[s.kappa_max3 <= LOW]
    q = np.percentile(called.klat, [10, 50, 90])
    print(f"\npredicted low by max3: {len(called)}; their DFT klat: "
          f"10th pct {q[0]:.2f}, median {q[1]:.2f}, 90th pct {q[2]:.2f} W/m/K")
    r = np.percentile(s.ratio_dft_over_max3, [10, 50, 90])
    print(f"DFT klat / max3, all {len(s)}: 10th pct {r[0]:.2f}, median {r[1]:.2f}, 90th pct {r[2]:.2f}")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    tp = (s.kappa_max3 <= LOW) & (s.klat <= LOW)
    fp = (s.kappa_max3 <= LOW) & (s.klat > LOW)
    fn = (s.kappa_max3 > LOW) & (s.klat <= LOW)
    for mask, colour, label in [(~(tp | fp | fn), "#cccccc", "both high"),
                                (fn, "#c9803a", f"missed ({fn.sum()})"),
                                (fp, "#b03a3a", f"false low ({fp.sum()})"),
                                (tp, "#3b6ea8", f"confirmed low ({tp.sum()})")]:
        ax[0].scatter(s.kappa_max3[mask], s.klat[mask], s=6, color=colour, label=label)
    lim = [0.03, 3000]
    ax[0].plot(lim, lim, "k-", lw=0.6)
    ax[0].axvline(LOW, color="k", ls="--", lw=1)
    ax[0].axhline(LOW, color="k", ls="--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel="screen kappa_max3, Poisson gamma (W/m/K)",
              ylabel="PhoNIX DFT klat (W/m/K)",
              title=f"{len(s)} unseen crystals, <= {MAX_ATOMS} atoms")
    ax[0].legend(fontsize=7, loc="upper left", markerscale=2)

    colours = {"ALIGNN": "#7a9cc6", "CGCNN-ens": "#9bc67a", "newbase": "#c69b7a", "kappa_max3": "k"}
    for name, g in curve.groupby("model"):
        g = g[g.n_selected >= 10]                        # fewer than 10 picks is noise
        lw = 2.2 if name == "kappa_max3" else 1.0
        ax[1].plot(g.pred_cutoff, g.frac_dft_low, color=colours[name], lw=lw, label=name)
        if name == "kappa_max3":
            ax[1].fill_between(g.pred_cutoff, g.lo95, g.hi95, color="k", alpha=0.12)
    ax[1].axhline((s.klat <= LOW).mean(), color="grey", ls=":", label="blind pick (base rate)")
    ax[1].axvline(LOW, color="k", ls="--", lw=1)
    ax[1].set(xscale="log", ylim=(0, 1), xlabel="predict-low cutoff c (W/m/K)",
              ylabel=f"fraction with DFT klat <= {LOW}",
              title="of crystals predicted <= c, how many are really low")
    ax[1].legend(fontsize=7)

    bins = np.geomspace(0.03, 100, 40)
    ax[2].hist(called.klat, bins=bins, color="#3b6ea8")
    ax[2].axvline(LOW, color="k", ls="--", lw=1)
    ax[2].set(xscale="log", xlabel="PhoNIX DFT klat (W/m/K)", ylabel="crystals",
              title=f"DFT kappa of the {len(called)} the screen calls low")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "87_phonix_poisson_calibration.png"), dpi=150)
    print("\nwrote 87_phonix_poisson_predictions.csv, 87_phonix_poisson_confusion.csv, "
          "87_precision_vs_cutoff.csv, 87_phonix_poisson_calibration.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["torch", "newbase", "score"])
    stage = ap.parse_args().stage
    # a dict of functions: look the stage name up, then call what comes back
    {"torch": stage_torch, "newbase": stage_newbase, "score": stage_score}[stage]()
