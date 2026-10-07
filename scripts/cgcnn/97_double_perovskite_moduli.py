#!/usr/bin/env python
"""
Step 97 - double perovskites: our predicted moduli (K, G) against DFT moduli, and the kappa they give
=====================================================================================================

Four runs, two Python environments, in this order (each stage reads what the one before wrote):

    # 1. download from the Materials Project (needs internet + the MP key in ~/.pmgrc.yaml; seconds)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/97_double_perovskite_moduli.py --stage fetch
    # 2. ALIGNN + CGCNN-ens moduli (torch only works in infer_env; about a minute)
    ~/miniconda3/envs/infer_env/bin/python scripts/cgcnn/97_double_perovskite_moduli.py --stage torch
    # 3. newbase moduli (matminer + xgboost, ml_env; about a minute)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/97_double_perovskite_moduli.py --stage newbase
    # 4. compare, kappa, figure (ml_env; seconds)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/97_double_perovskite_moduli.py --stage score

WHY
---
2026-10-07: the DFT check on the final 15 will compute the ELASTIC MODULI, not
kappa_L (kappa_L costs too much; elastic constants are cheap). So for this
family the questions become:
  1. how close are our predicted K and G to DFT K and G?
  2. how much do DFT moduli themselves move with the DFT setup?
  3. what kappa does the physics chain give on each set of moduli?

The family is the A2BX6 vacancy-ordered double perovskites (Cs2SnI6 and
relatives), the parent family of 8 of the final 15. Step 96 compared kappa
only, and only where a held-out prediction was already stored. This step
runs the three screen models on every family member we have a structure for:
  26 in PhoNIX                  20 of them are in matbench, 6 are not
  5 of Cao's 18 PhoNIX lacks    cubic Fm-3m (No. 225, the structure Cao describes),
                                from the Materials Project; the sixth, Cs2ZrBr6,
                                is not in the Materials Project

WHICH STRUCTURE THE MODELS SEE - and why it matters
---------------------------------------------------
PhoNIX's cells for this family are about 11% SMALLER in volume per atom than
the matbench cells of the same crystals (median ratio 0.893; 0.952 over all
1,647 PhoNIX-matbench matches). Our models learned from matbench's cells
(Materials Project, PBE), so a compressed cell makes them predict a stiffer
crystal. Every model is therefore run twice:
  main          the matbench cell where the crystal is in matbench (the cell its
                DFT label was computed on), the Materials Project's PBE cell
                otherwise. All K, G and kappa columns without a suffix use it.
                (NOT MP's default cell: since MP moved to r2SCAN, that one is
                about 6% smaller in volume than the PBE cell. MP's PBE cell
                equals matbench's to 0.1%, checked in the score stage)
  PhoNIX cell   PhoNIX's cell, read exactly as step 87 read it. Columns end in
                "_phonix". Shows how much the cell alone moves the answer

WHAT IS COMPARED
----------------
predicted K, G   the screen's three models: ALIGNN (one model), CGCNN-ens (three
                 models averaged) and newbase (the 297-descriptor XGBoost)
DFT K, G         matbench   the labels our models were trained on (Materials
                            Project, PBE, the 2018-19 snapshot)
                 MP now     the Materials Project's elasticity data today
                 Bhumla     Bhumla, Jain, Sheoran, Bhattacharya, arXiv:2209.08559
                            (2022), Table 1, p.7: 4 Cs2BI6, VASP. The paper does
                            not say which functional gave the elastic constants;
                            its relaxations used PBE and optB86-vdW
kappa            each model's own chain kappa (Poisson gamma, Eq. 6 of the deck),
                 and the same chain on each set of DFT moduli

HELD OUT, OR NOT - every comparison carries this flag
  ALIGNN, CGCNN-ens  held out if not in matbench, or in matbench's test split
  newbase            the full model averages 5 fold models, so it is held out
                     only for crystals not in matbench. For matbench crystals
                     the OUT-OF-FOLD prediction (newbase_oof) is added: each
                     value comes from the one fold model that never saw it

WRITTEN BEFORE THE MODELS WERE RUN (but after the DFT reference values were seen)
-------------------------------------------------------------------------------
held-out K and G mostly within x1.25 of matbench's DFT (|log10 ratio| <= 0.1)
    -> the models do as well here as on the test set; a DFT run with the
       Materials Project's settings (PBE) should land near our numbers
clearly worse than that
    -> this family is hard for the models; the DFT moduli are a real test
Bhumla vs MP further apart than our models are from MP
    -> the DFT setup (functional, van der Waals) moves the moduli more than the
       models' error does; the supervisor's run must use MP-like settings, or
       a disagreement with our numbers cannot be pinned on the models

OUTPUTS
    results/cgcnn/97_mp_structures.json           stage 1: MP's PBE + r2SCAN cells (the 26 + 5 Cao)
    results/cgcnn/97_matbench_structures.json     stage 1: matbench's cells of the 20 in matbench
    results/cgcnn/97_mp_elasticity.csv            stage 1: MP's K, G today
    results/cgcnn/97_torch_moduli.csv             stage 2
    results/cgcnn/97_newbase_moduli.csv           stage 3
    results/cgcnn/97_moduli_vs_dft.csv            stage 4: one row per crystal x predictor x DFT source
    results/cgcnn/97_double_perovskite_kappa.csv  stage 4: one row per crystal: K, G and kappa from every source
    results/cgcnn/97_double_perovskite_moduli.png stage 4
"""
import os
import sys

# torch must be imported BEFORE numpy, or the two OpenMP runtimes clash (step 87
# does the same). sys.argv is the list of words typed after "python", e.g.
# ["97_double_perovskite_moduli.py", "--stage", "torch"].
if "torch" in " ".join(sys.argv[1:]):
    import torch  # noqa: F401
else:
    # xgboost crashes on this machine with more than one OpenMP thread
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse                       # reads "--stage torch" from the command line
import ast                            # turns PhoNIX's structure text back into a dict
import json
import warnings
from importlib import import_module   # imports a file whose name starts with a digit

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")     # pymatgen and matminer print a lot of noise
from pymatgen.core import Composition, Lattice, Structure  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path is where Python looks for modules; adding these folders lets this
# script reuse steps 04, 15, 71 and 87 instead of copying their code
for p in [ROOT, os.path.join(ROOT, "scripts", "cgcnn"), os.path.join(ROOT, "scripts", "alignn")]:
    sys.path.insert(0, p)

# =============================================================================
#  CONFIG
# =============================================================================
RES = os.path.join(ROOT, "results", "cgcnn")
STEP96 = os.path.join(RES, "96_double_perovskite_check.csv")      # the family + literature kappa
MATCH85 = os.path.join(RES, "85_phonix_matbench_match.csv")       # PhoNIX id -> matbench id
PHONIX = os.path.expanduser("~/Desktop/a new project/data/phonix/data_all.csv")
LABELS = os.path.join(ROOT, "data_full", "labels.csv")            # matbench DFT K_VRH, G_VRH
# newbase out-of-fold predictions; {} is filled with kvrh or gvrh by .format()
OOF = os.path.expanduser("~/Desktop/pink_reproduction/xgboost_oof/"
                         "matbench_log_{}_composition+structure+angular_oof.csv")

MP_STRUCT = os.path.join(RES, "97_mp_structures.json")
MB_STRUCT = os.path.join(RES, "97_matbench_structures.json")
# the stored single-ALIGNN test-set predictions (GPa), from training: a check on the matbench cells
ALIGNN_TEST = os.path.join(ROOT, "results", "alignn", "alignn_{}", "prediction_results_test_set.csv")
CHAIN82 = os.path.join(RES, "82_test_predictions.csv")            # CGCNN-ens on the matbench test cells
MP_ELAST = os.path.join(RES, "97_mp_elasticity.csv")
TORCH_OUT = os.path.join(RES, "97_torch_moduli.csv")
NEWBASE_OUT = os.path.join(RES, "97_newbase_moduli.csv")
OUT_COMPARE = os.path.join(RES, "97_moduli_vs_dft.csv")
OUT_KAPPA = os.path.join(RES, "97_double_perovskite_kappa.csv")
OUT_PNG = os.path.join(RES, "97_double_perovskite_moduli.png")

LOW = 1.0               # W/m/K, the screen's cutoff
MIN_MODULUS = 0.01      # GPa. Step 57's rule: an ALIGNN K or G below this is not a prediction
ALIGNN_TOL = 1e-2       # step 87: ALIGNN neighbour ties make it reproduce to ~0.7%, not 1e-4
TIE_JITTER = 1e-5       # Angstrom: how far each atom is nudged to expose ALIGNN's neighbour ties
N_JITTER = 8            # nudges per crystal
CUBIC = 225             # space group Fm-3m

# Bhumla et al., arXiv:2209.08559 (2022), Table 1, p.7, typed from the local PDF
# (dft/final_shortlist_15/papers/03_Bhumla2022_arXiv_Cs2BI6_thermoelectrics.pdf).
# All in GPa. B and G are the paper's Voigt-Reuss-Hill values; the score stage
# recomputes them from C11, C12, C44 (the paper's SI eqs. 7-10) as a typing check.
BHUMLA_REF = "Bhumla et al., arXiv:2209.08559 (2022), Table 1"
BHUMLA = {            # formula: (C11,  C12,   C44,   B,     G)
    "Cs2PtI6": (9.58, 4.51, 3.93, 6.20, 3.30),
    "Cs2PdI6": (16.64, 8.98, 7.36, 11.53, 5.66),
    "Cs2TeI6": (20.30, 10.55, 8.70, 13.80, 6.90),
    "Cs2SnI6": (14.36, 8.20, 6.65, 10.25, 4.88),
}

# the screen's three models: name -> (K column, G column)
MODELS = {"ALIGNN": ("K_alignn", "G_alignn"),
          "CGCNN-ens": ("K_cgcnn", "G_cgcnn"),
          "newbase": ("K_newbase", "G_newbase")}
# the DFT moduli: name -> (K column, G column)
DFT = {"matbench": ("K_matbench", "G_matbench"),
       "MP now": ("K_mp_now", "G_mp_now"),
       "Bhumla": ("K_bhumla", "G_bhumla")}


# =============================================================================
#  shared: every family member we have a structure for
# =============================================================================
def load_crystals(which="main"):
    """One row per crystal: formula, mp_id, where the structure came from, the structure.

    which = "main"    matbench's cell for the 20 crystals in matbench, the
                      Materials Project's PBE cell for the 6 PhoNIX crystals that are not
              "phonix"  PhoNIX's cell for all 26, read the way step 87 read them
                      (first copy per mp_id), so the 5 crystals step 87 already
                      scored must give step 87's numbers back
    The 5 Cao compounds PhoNIX lacks use the Materials Project's PBE cell either way.
    """
    fam = pd.read_csv(STEP96)
    ph = fam.dropna(subset=["mp_id"])              # the 26 PhoNIX rows
    mp_file = json.load(open(MP_STRUCT))

    rows = []
    if which == "phonix":
        raw = pd.read_csv(PHONIX, usecols=["mp_id", "structure"])
        raw = raw[raw.mp_id.isin(ph.mp_id)]
        # drop_duplicates keeps the FIRST row of each mp_id, as steps 85 and 87 did
        text_of = dict(raw.drop_duplicates("mp_id").set_index("mp_id").structure)
        for formula, mp in zip(ph.formula, ph.mp_id):  # zip walks the two columns side by side
            s = ast.literal_eval(text_of[mp])
            st = Structure(Lattice(s["cell"]), s["numbers"], s["positions"], coords_are_cartesian=True)
            rows.append({"formula": formula, "mp_id": mp, "structure_from": "PhoNIX", "structure": st})
    else:
        mb_cells = json.load(open(MB_STRUCT))       # mp_id -> {"mb_id": ..., "structure": ...}
        for formula, mp in zip(ph.formula, ph.mp_id):
            if mp in mb_cells:
                st, src = Structure.from_dict(mb_cells[mp]["structure"]), "matbench"
            else:
                st, src = Structure.from_dict(mp_file["pbe"][mp]), "Materials Project"
            rows.append({"formula": formula, "mp_id": mp, "structure_from": src, "structure": st})

    for p in mp_file["picked"]:
        rows.append({"formula": p["formula"], "mp_id": p["mp_id"], "structure_from": "Materials Project",
                     "structure": Structure.from_dict(mp_file["pbe"][p["mp_id"]])})

    d = pd.DataFrame(rows)
    # what the kappa formula needs, from the cell as stored (primitive in both sources)
    d["n_atoms"] = [len(s) for s in d.structure]
    d["volume_a3"] = [s.volume for s in d.structure]
    d["density"] = [s.density for s in d.structure]                       # g/cm^3
    d["mass_amu"] = [float(s.composition.weight) for s in d.structure]    # whole cell
    # every structure must have the formula its row claims
    same = [s.composition.reduced_formula == Composition(f).reduced_formula
            for s, f in zip(d.structure, d.formula)]
    if not all(same):
        raise SystemExit(f"structure/formula mismatch: {d.formula[~np.array(same)].tolist()}")
    if d.mp_id.duplicated().any():
        raise SystemExit("an mp_id appears twice")
    return d


def agree(a, b, tol):
    """True when every value in a matches b to within tol (relative)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.abs(a - b) <= tol * np.abs(b)))


# =============================================================================
#  stage 1: fetch from the Materials Project  (ml_env, internet)
# =============================================================================
def stage_fetch():
    from mp_api.client import MPRester

    fam = pd.read_csv(STEP96)
    missing = fam[fam.mp_id.isna()].formula.tolist()    # Cao's compounds PhoNIX lacks
    print(f"Cao compounds not in PhoNIX: {missing}")

    with MPRester() as mpr:                             # reads the key from ~/.pmgrc.yaml
        version = mpr.get_database_version()
        docs = mpr.materials.summary.search(
            formula=missing,
            fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull", "structure"])

        picked, not_found = [], []
        for f in missing:
            # the cubic polymorph only; if MP holds several, the most stable one
            cands = [x for x in docs if x.formula_pretty == f and x.symmetry.number == CUBIC]
            if not cands:
                not_found.append(f)
                continue
            best = min(cands, key=lambda x: x.energy_above_hull)   # lambda = a one-line function
            picked.append({"formula": f, "mp_id": str(best.material_id),
                           "energy_above_hull": best.energy_above_hull,
                           "structure": best.structure.as_dict()})
        print(f"found in MP (Fm-3m): {[p['formula'] for p in picked]};  not found: {not_found}")

        family_ids = fam.mp_id.dropna().tolist()       # the 26 PhoNIX crystals
        ids = family_ids + [p["mp_id"] for p in picked]

        # MP's DEFAULT cell today is an r2SCAN relaxation, about 6% smaller in volume
        # than the PBE cells matbench (and so our models) used. Kept for the record only.
        fdocs = mpr.materials.summary.search(material_ids=family_ids, fields=["material_id", "structure"])
        r2scan = {str(x.material_id): x.structure.as_dict() for x in fdocs}   # a dict built in one line

        # MP's PBE ("GGA") relaxed cell: the same kind of cell as matbench's. Used as the
        # main cell for every crystal that is not in matbench.
        pbe = {}
        for mid in ids:
            es = mpr.get_entries(mid, compatible_only=False,
                                 additional_criteria={"thermo_types": ["GGA_GGA+U"]})
            if len(es) != 1:
                raise SystemExit(f"{mid}: expected one PBE entry in MP, found {len(es)}")
            pbe[mid] = es[0].structure.as_dict()

        # MP's elasticity data today, for every family member that has an mp id
        el = mpr.materials.elasticity.search(material_ids=ids)

    with open(MP_STRUCT, "w") as fh:
        json.dump({"mp_database_version": version, "picked": picked, "not_found": not_found,
                   "pbe": pbe, "r2scan_default": r2scan}, fh)

    # ---- matbench's own cells (no internet: matminer's local copy) ----------------
    # torch runs in infer_env, which has no matminer, so the cells are saved here.
    # Row i of the dataset is "mb-{i:05d}", the naming step 01b used (step 85 checks it).
    from matminer.datasets import load_dataset
    mb = load_dataset("matbench_log_kvrh")
    m85 = pd.read_csv(MATCH85, usecols=["mp_id", "mb_id"])
    m85 = m85[m85.mp_id.isin(family_ids) & m85.mb_id.notna()]
    cells = {}
    for mp, mb_id in zip(m85.mp_id, m85.mb_id):
        st = mb.structure.iloc[int(mb_id.split("-")[1])]          # "mb-00042" -> row 42
        claimed = fam.loc[fam.mp_id == mp, "formula"].iloc[0]
        if st.composition.reduced_formula != Composition(claimed).reduced_formula:
            raise SystemExit(f"{mb_id} is {st.composition.reduced_formula}, expected {claimed}")
        cells[mp] = {"mb_id": mb_id, "structure": st.as_dict()}
    with open(MB_STRUCT, "w") as fh:
        json.dump(cells, fh)
    print(f"matbench cells for {len(cells)} crystals -> {MB_STRUCT}")

    rows = []
    for e in el:
        rows.append({"mp_id": str(e.material_id), "formula": e.formula_pretty,
                     "K_mp_now": e.bulk_modulus.vrh if e.bulk_modulus else np.nan,
                     "G_mp_now": e.shear_modulus.vrh if e.shear_modulus else np.nan,
                     "state": str(e.state), "mp_flags": "; ".join(e.warnings or []),
                     "mp_database_version": version})
    pd.DataFrame(rows).sort_values("mp_id").to_csv(MP_ELAST, index=False)
    print(f"MP database {version}: elasticity for {len(rows)} of {len(ids)} crystals")
    print(f"wrote {MP_STRUCT}\nwrote {MP_ELAST}")


# =============================================================================
#  stage 2: ALIGNN and CGCNN-ens  (infer_env)
# =============================================================================
def stage_torch():
    import torch
    from jarvis.core.atoms import pmg_to_atoms
    from cgcnn_scratch.data import AtomFeaturiser, GaussianDistance, structure_to_graph
    _al = import_module("15_alignn_predict_moduli")     # load_alignn_target, predict_pair
    _pred = import_module("04_predict_moduli")          # load_model, predict_ensemble

    # the same models, loaded the same way, as step 87 (the screen's models)
    dev = torch.device("cpu")
    k_al, cfg = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_bulk_modulus_kv"), dev)
    g_al, _ = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_shear_modulus_gv"), dev)
    k_cg = [_pred.load_model(t, RES)[:2] for t in ["K_VRH_full", "K_VRH_s1", "K_VRH_s2"]]
    g_cg = [_pred.load_model(t, RES)[:2] for t in ["G_VRH_full", "G_VRH_s1", "G_VRH_s2"]]
    ari = AtomFeaturiser(os.path.join(ROOT, "cgcnn_scratch", "atom_init.json"))
    gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)    # radius 8, step 0.2, as in steps 13 and 87

    def predict(d):
        """ALIGNN and CGCNN-ens K, G (GPa) for every structure in d, one row per mp_id."""
        rows, graphs = [], []
        for st, mp in zip(d.structure, d.mp_id):
            k, g = _al.predict_pair(k_al, g_al, pmg_to_atoms(st), cfg, dev)
            rows.append({"mp_id": mp, "K_alignn": k, "G_alignn": g})
            graphs.append(structure_to_graph(st, ari, gdf, 12, 8))   # 12 neighbours within 8 A
        out = pd.DataFrame(rows)
        ds = _pred.PredictionSet(graphs, d.mp_id.tolist())
        K, _ = _pred.predict_ensemble(k_cg, ds)         # dicts: mp_id -> GPa
        G, _ = _pred.predict_ensemble(g_cg, ds)
        out["K_cgcnn"] = out.mp_id.map(K)               # .map looks each id up in the dict
        out["G_cgcnn"] = out.mp_id.map(G)
        return out

    out = predict(load_crystals("main"))
    ph_cells = load_crystals("phonix")
    on_phonix = predict(ph_cells)

    # ---- check 1: on PhoNIX's cells, the crystals step 87 scored come back the same --
    old = pd.read_csv(os.path.join(RES, "87_torch_moduli.csv"))
    both = on_phonix.merge(old, on="mp_id", suffixes=("", "_87"))
    ok = (len(both) > 0
          and agree(both.K_cgcnn, both.K_cgcnn_87, 1e-4) and agree(both.G_cgcnn, both.G_cgcnn_87, 1e-4)
          and agree(both.K_alignn, both.K_alignn_87, ALIGNN_TOL)
          and agree(both.G_alignn, both.G_alignn_87, ALIGNN_TOL))
    print(both[["mp_id", "K_alignn", "K_alignn_87", "K_cgcnn", "K_cgcnn_87"]].to_string(index=False))
    if not ok:
        raise SystemExit("CHECK FAILED: step 87's stored ALIGNN/CGCNN moduli did not come back")
    print(f"check 1 passed: {len(both)} crystals match step 87 on PhoNIX cells "
          f"(CGCNN to 1e-4, ALIGNN to {ALIGNN_TOL:.0%})")

    # ---- check 2: on matbench's cells, CGCNN-ens gives step 82's test-split numbers back.
    # That proves the matbench cells are the ones the models were scored on.
    mb_ids = {mp: v["mb_id"] for mp, v in json.load(open(MB_STRUCT)).items()}
    out["mb_id"] = out.mp_id.map(mb_ids)
    s82 = pd.read_csv(CHAIN82, usecols=["mb_id", "K_CGCNN-ens", "G_CGCNN-ens"])
    test = out.merge(s82, on="mb_id")
    if not (len(test) > 0 and agree(test.K_cgcnn, test["K_CGCNN-ens"], 1e-4)
            and agree(test.G_cgcnn, test["G_CGCNN-ens"], 1e-4)):
        print(test.to_string(index=False))
        raise SystemExit("CHECK FAILED: CGCNN-ens on the matbench test cells did not give step 82's numbers")
    print(f"check 2 passed: {len(test)} test crystals, CGCNN-ens on matbench cells = step 82 to 1e-4")

    # ALIGNN is NOT checked against the test-set file its training run wrote, because
    # that file came from a different checkpoint: last_model.pt (epoch 150) reproduces
    # 2 of these 5 crystals exactly, best_model.pt (epoch 92 for K, 115 for G - the one
    # step 15 and the screen load) none. Shown for information only.
    for t, target in [("K", "bulk_modulus_kv"), ("G", "shear_modulus_gv")]:
        al = pd.read_csv(ALIGNN_TEST.format(target), skipinitialspace=True)   # id, target, prediction
        test[f"{t}_alignn_epoch150"] = test.mb_id.map(al.set_index("id").prediction)
    print("for information - ALIGNN best_model here vs the training run's epoch-150 file:")
    print(test[["mb_id", "K_alignn", "K_alignn_epoch150", "G_alignn", "G_alignn_epoch150"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- ALIGNN's neighbour-tie noise on the main cells ---------------------------
    # ALIGNN keeps each atom's 12 nearest neighbours. In these cubic cells many
    # neighbours sit at EXACTLY the same distance, so which ones make the cut is
    # decided by rounding. Nudging every atom by ~TIE_JITTER Angstrom (far below any
    # physical change) re-rolls that choice; the range of answers is the noise.
    rng = np.random.default_rng(0)                      # fixed seed: the same nudges every run
    main = load_crystals("main")
    tie = []
    for st, mp in zip(main.structure, main.mp_id):
        ks, gs = [], []
        for _ in range(N_JITTER):
            moved = Structure(st.lattice, st.species,
                              st.cart_coords + rng.normal(0, TIE_JITTER, (len(st), 3)),
                              coords_are_cartesian=True)
            k, g = _al.predict_pair(k_al, g_al, pmg_to_atoms(moved), cfg, dev)
            ks.append(k)
            gs.append(g)
        tie.append({"mp_id": mp, "K_alignn_tie_min": min(ks), "K_alignn_tie_max": max(ks),
                    "G_alignn_tie_min": min(gs), "G_alignn_tie_max": max(gs)})
    out = out.merge(pd.DataFrame(tie), on="mp_id")
    for t in "KG":
        # the unnudged prediction counts too: the range covers all N_JITTER + 1 answers
        lo = np.minimum(out[f"{t}_alignn_tie_min"], out[f"{t}_alignn"])
        hi = np.maximum(out[f"{t}_alignn_tie_max"], out[f"{t}_alignn"])
        out[f"{t}_alignn_tie_min"], out[f"{t}_alignn_tie_max"] = lo, hi
        spread = hi / lo
        print(f"ALIGNN {t} tie noise (max/min over {N_JITTER} nudges + the original): "
              f"median x{spread.median():.3f}, worst x{spread.max():.3f} "
              f"({out.mp_id[spread.idxmax()]})")

    # one file: the main predictions, then the PhoNIX-cell ones with a "_phonix" suffix
    # (only the 26 PhoNIX crystals: the 5 Cao compounds have one cell, MP's, either way)
    side = on_phonix[on_phonix.mp_id.isin(ph_cells[ph_cells.structure_from == "PhoNIX"].mp_id)]
    out = out.drop(columns="mb_id").merge(side.add_suffix("_phonix").rename(columns={"mp_id_phonix": "mp_id"}),
                                          on="mp_id", how="left")
    out.to_csv(TORCH_OUT, index=False)
    print(f"wrote {TORCH_OUT}: {len(out)} crystals")


# =============================================================================
#  stage 3: newbase  (ml_env)
# =============================================================================
def stage_newbase():
    _s87 = import_module("87_phonix_poisson_calibration")   # make_featuriser: the 297 descriptors
    _s71 = import_module("71_screen_with_new_baseline")     # predict_baseline: 5 xgboost folds
    feat_one = _s87.make_featuriser()

    def predict(d):
        """newbase K, G (GPa) for every structure in d, one row per mp_id."""
        f = pd.DataFrame([feat_one(st, mp) for st, mp in zip(d.structure, d.mp_id)])
        bad = f.featurize_error.fillna("") != ""
        if bad.any():
            raise SystemExit(f"failed to featurise: {f[bad].id.tolist()}")
        p = _s71.predict_baseline(f.set_index("id"))        # log10(GPa)
        return pd.DataFrame({"mp_id": f.id.values, "K_newbase": 10 ** p["K"], "G_newbase": 10 ** p["G"],
                             "primitive_fallback": f.primitive_fallback.values})

    out = predict(load_crystals("main"))
    ph_cells = load_crystals("phonix")
    on_phonix = predict(ph_cells[ph_cells.structure_from == "PhoNIX"])   # the 26 with a PhoNIX cell

    # ---- check: on PhoNIX's cells, the crystals step 87 scored come back the same --
    # (no stored newbase-full prediction exists on matbench's cells: step 82 used the
    # out-of-fold ones, which come from different fold models)
    old = pd.read_csv(os.path.join(RES, "87_newbase_moduli.csv"))
    both = on_phonix.merge(old, on="mp_id", suffixes=("", "_87"))
    if not (len(both) > 0 and agree(both.K_newbase, both.K_newbase_87, 1e-4)
            and agree(both.G_newbase, both.G_newbase_87, 1e-4)):
        print(both.to_string(index=False))
        raise SystemExit("CHECK FAILED: step 87's stored newbase moduli did not come back")
    print(f"check passed: {len(both)} crystals match step 87's newbase K and G to 1e-4 on PhoNIX cells")

    side = on_phonix.drop(columns="primitive_fallback").add_suffix("_phonix")
    out = out.merge(side.rename(columns={"mp_id_phonix": "mp_id"}), on="mp_id", how="left")
    out.to_csv(NEWBASE_OUT, index=False)
    print(f"wrote {NEWBASE_OUT}: {len(out)} crystals")


# =============================================================================
#  stage 4: compare and score  (ml_env)
# =============================================================================
def bhumla_table():
    """Bhumla's B and G, after checking them against their own elastic constants."""
    rows = []
    for formula, (c11, c12, c44, b, g) in BHUMLA.items():
        # Voigt and Reuss bounds for a cubic crystal (Bhumla SI eqs. 7-9), Hill = their mean (eq. 10)
        b_vrh = (c11 + 2 * c12) / 3
        g_v = (c11 - c12 + 3 * c44) / 5
        g_r = 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12))
        g_vrh = (g_v + g_r) / 2
        # the table rounds to 0.01 GPa, so recomputed values may differ by a few 0.001
        if abs(b_vrh - b) > 0.01 or abs(g_vrh - g) > 0.01:
            raise SystemExit(f"Bhumla {formula}: typed B, G ({b}, {g}) disagree with "
                             f"their own C11, C12, C44 ({b_vrh:.3f}, {g_vrh:.3f}) - check the typing")
        rows.append({"formula": formula, "K_bhumla": b, "G_bhumla": g})
    return pd.DataFrame(rows)


def stage_score():
    _s71 = import_module("71_screen_with_new_baseline")    # slack_physics, verify_physics
    import matplotlib
    matplotlib.use("Agg")                                  # draw to a file, not a window
    import matplotlib.pyplot as plt

    _s71.verify_physics(_s71.rank_four_models())   # the kappa formula is the screen's, or stop

    # ---- the crystals and what step 96 already knows about them ---------------
    d = load_crystals("main")
    mp_file = json.load(open(MP_STRUCT))

    # ---- the cells: how different are PhoNIX's and MP's today from the main one? ----
    ph_cells = load_crystals("phonix").set_index("mp_id")
    is_ph = ph_cells.structure_from == "PhoNIX"            # the 26 with a real PhoNIX cell
    per_atom = d.volume_a3 / d.n_atoms                     # A^3 per atom, main cell
    d["vol_ratio_phonix"] = d.mp_id.map((ph_cells.volume_a3 / ph_cells.n_atoms)[is_ph]) / per_atom
    for key, col in [("pbe", "vol_ratio_mp_pbe"), ("r2scan_default", "vol_ratio_mp_r2scan")]:
        vpa = {}                                           # mp_id -> A^3 per atom in that MP cell
        for m, s in mp_file[key].items():
            st = Structure.from_dict(s)
            vpa[m] = st.volume / len(st)
        d[col] = d.mp_id.map(vpa) / per_atom
        d.loc[d.structure_from != "matbench", col] = np.nan   # only meaningful against matbench's cell
    # the check that makes MP's PBE cells a fair stand-in for matbench's: the typical
    # cell is identical, and none is off by more than 2% (when written, 19 of 20 were
    # identical and Rb2TeBr6, re-relaxed by MP since 2018, was 0.75% larger - small next
    # to the ~6% r2SCAN and ~10% PhoNIX differences this step is about)
    r = d.vol_ratio_mp_pbe.dropna()
    if abs(r.median() - 1) > 1e-3 or (r - 1).abs().max() > 0.02:
        raise SystemExit("MP's PBE cells are not the same size as matbench's - check the source")
    # what the kappa formula needs, from PhoNIX's cell, for the side columns
    for c in ["n_atoms", "volume_a3", "density", "mass_amu"]:
        d[f"{c}_phonix"] = d.mp_id.map(ph_cells[c][is_ph])
    d = d.drop(columns="structure")
    fam = pd.read_csv(STEP96)
    # give the 5 MP-only compounds their MP id, so every row can be joined on mp_id
    new_id = {p["formula"]: p["mp_id"] for p in mp_file["picked"]}
    fam["mp_id"] = fam.mp_id.fillna(fam.formula.map(new_id))
    info = fam[["mp_id", "matbench_split", "phonix_klat", "in_cao_18",
                "lit_measured", "lit_DFT_cao", "lit_DFT_other"]]
    d = d.merge(info, on="mp_id", how="left")

    # the MP-only compounds have no matbench split yet: prove they are not in matbench
    labels = pd.read_csv(LABELS)
    mb_formulas = {Composition(f).reduced_formula for f in labels.formula}
    new = d.mp_id.isin(new_id.values())                   # the 5 Cao compounds PhoNIX lacks
    if any(Composition(f).reduced_formula in mb_formulas for f in d.formula[new]):
        raise SystemExit("an MP-only compound IS in matbench - its split must be looked up")
    d.loc[new, "matbench_split"] = "not in mb"

    # ---- DFT moduli: matbench labels, MP today, Bhumla ---------------------------
    m85 = pd.read_csv(MATCH85, usecols=["mp_id", "mb_id"])
    d = d.merge(m85, on="mp_id", how="left")
    lab = labels.set_index("mb_id")
    d["K_matbench"] = d.mb_id.map(lab.K_VRH)
    d["G_matbench"] = d.mb_id.map(lab.G_VRH)
    el = pd.read_csv(MP_ELAST)
    d = d.merge(el[["mp_id", "K_mp_now", "G_mp_now"]], on="mp_id", how="left")
    d = d.merge(bhumla_table(), on="formula", how="left")

    # ---- predictions --------------------------------------------------------------
    d = d.merge(pd.read_csv(TORCH_OUT), on="mp_id", how="left")
    d = d.merge(pd.read_csv(NEWBASE_OUT).drop(columns="primitive_fallback"), on="mp_id", how="left")
    for t, task in [("K", "kvrh"), ("G", "gvrh")]:
        oof = pd.read_csv(OOF.format(task))                       # row_index = matbench row
        oof["mb_id"] = [f"mb-{i:05d}" for i in oof.row_index]     # f"{i:05d}" pads to 5 digits
        oof = oof.set_index("mb_id")
        d[f"{t}_newbase_oof"] = 10 ** d.mb_id.map(oof.y_pred_oof)
        # the file's own truth must equal matbench's label - proves the ids line up
        truth = 10 ** d.mb_id.map(oof.y_true)
        has = d.mb_id.notna()
        if not agree(truth[has], d.loc[has, f"{t}_matbench"], 1e-4):
            raise SystemExit(f"newbase out-of-fold truth disagrees with matbench {t} - ids mismatched")
    if d[["K_alignn", "K_cgcnn", "K_newbase"]].isna().any().any():
        raise SystemExit("a crystal is missing a model prediction")

    # held out? (see the docstring)
    d["networks_held_out"] = d.matbench_split.isin(["not in mb", "test"])
    d["newbase_held_out"] = d.matbench_split == "not in mb"

    # ---- 1. moduli: every predictor against every DFT source ----------------------
    predictors = {"ALIGNN": ("K_alignn", "G_alignn", "networks_held_out"),
                  "CGCNN-ens": ("K_cgcnn", "G_cgcnn", "networks_held_out"),
                  "newbase": ("K_newbase", "G_newbase", "newbase_held_out"),
                  "newbase_oof": ("K_newbase_oof", "G_newbase_oof", None),   # None = always held out
                  "DFT: MP now": ("K_mp_now", "G_mp_now", None),             # DFT against DFT
                  # the same three models on PhoNIX's compressed cells: shows what the cell does
                  "ALIGNN, PhoNIX cell": ("K_alignn_phonix", "G_alignn_phonix", "networks_held_out"),
                  "CGCNN-ens, PhoNIX cell": ("K_cgcnn_phonix", "G_cgcnn_phonix", "networks_held_out"),
                  "newbase, PhoNIX cell": ("K_newbase_phonix", "G_newbase_phonix", "newbase_held_out")}
    long = []
    for pname, (pk, pg, flag) in predictors.items():
        for rname, (rk, rg) in DFT.items():
            if rname == "MP now" and pname == "DFT: MP now":
                continue                                   # a source against itself
            sub = d.dropna(subset=[pk, pg, rk, rg])
            for _, r in sub.iterrows():                    # iterrows gives one row at a time
                long.append({"formula": r.formula, "mp_id": r.mp_id, "matbench_split": r.matbench_split,
                             "predictor": pname, "dft_source": rname,
                             "held_out": True if flag is None else bool(r[flag]),
                             "K_pred": r[pk], "K_dft": r[rk], "G_pred": r[pg], "G_dft": r[rg],
                             "dlog10_K": np.log10(r[pk] / r[rk]), "dlog10_G": np.log10(r[pg] / r[rg])})
    long = pd.DataFrame(long)
    long.loc[long.predictor == "DFT: MP now", "held_out"] = np.nan   # not a model: no such flag
    long.to_csv(OUT_COMPARE, index=False)

    # summary: per DFT source x predictor x held out.
    # "within x1.25" = K and G BOTH within a factor 1.25 of the DFT value (|log10| <= 0.097)
    x125 = np.log10(1.25)
    summ = (long.assign(abs_K=long.dlog10_K.abs(), abs_G=long.dlog10_G.abs(),
                        both_within=(long.dlog10_K.abs() <= x125) & (long.dlog10_G.abs() <= x125))
            .groupby(["dft_source", "predictor", "held_out"], dropna=False)   # dropna=False keeps the DFT rows
            .agg(n=("formula", "size"),
                 MAE_log10_K=("abs_K", "mean"), MAE_log10_G=("abs_G", "mean"),
                 median_ratio_K=("dlog10_K", lambda x: 10 ** np.median(x)),
                 median_ratio_G=("dlog10_G", lambda x: 10 ** np.median(x)),
                 within_x125=("both_within", "mean"))
            .reset_index())
    print("\nPREDICTED (or MP-now DFT) moduli against each DFT source  "
          "(ratio = predicted / DFT;  within_x125 = K and G both within x1.25)")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # REFERENCE: the same "both within x1.25" rate on the whole matbench test split
    # (step 82's stored predictions). The pre-registration says "as well as on the
    # test set"; the soft subset (G <= 6 GPa, this family's range) was added AFTER the
    # result was seen, because every model is worse on soft crystals.
    # Note: step 82's ALIGNN is the 3-model ensemble, not the screen's single ALIGNN.
    t82 = pd.read_csv(CHAIN82)
    print("\nREFERENCE, matbench test split (step 82): K and G both within x1.25 of matbench DFT")
    for m in ["ALIGNN", "CGCNN-ens", "newbase"]:
        within = ((np.log10(t82[f"K_{m}"] / t82.K_VRH).abs() <= x125)
                  & (np.log10(t82[f"G_{m}"] / t82.G_VRH).abs() <= x125))
        soft = t82.G_VRH <= 6
        label = {"ALIGNN": "ALIGNN (3-ens)", "newbase": "newbase (oof)"}.get(m, m)   # .get(key, default)
        print(f"  {label:15s} all {len(t82):,}: {within.mean():.2f}    soft, G <= 6 GPa "
              f"(n={soft.sum()}): {within[soft].mean():.2f}")

    # ---- 2. kappa: the chain on every set of moduli -----------------------------
    sources = {**MODELS, "newbase_oof": ("K_newbase_oof", "G_newbase_oof"),
               **{f"DFT {k}": v for k, v in DFT.items()}}     # ** unpacks one dict into another
    for name, (kc, gc) in sources.items():
        phys = _s71.slack_physics(d[kc].values, d[gc].values, d.volume_a3.values,
                                  d.density.values, d.mass_amu.values, d.n_atoms.values)
        d[f"kappa_{name}"] = phys["kappa_cal"]               # W/m/K at 300 K; NaN where moduli are missing
        d[f"gamma_{name}"] = phys["gruneisen"]
    # step 57's rule: an ALIGNN K or G near zero is not a prediction
    d["alignn_reliable"] = (d.K_alignn > MIN_MODULUS) & (d.G_alignn > MIN_MODULUS)
    d.loc[~d.alignn_reliable, "kappa_ALIGNN"] = np.nan
    d["kappa_max3"] = d[[f"kappa_{m}" for m in MODELS]].max(axis=1)   # the screen's rule

    # the same chain on PhoNIX's cells: their moduli AND their volume, density, mass
    for name, (kc, gc) in MODELS.items():
        phys = _s71.slack_physics(d[f"{kc}_phonix"].values, d[f"{gc}_phonix"].values,
                                  d.volume_a3_phonix.values, d.density_phonix.values,
                                  d.mass_amu_phonix.values, d.n_atoms_phonix.values)
        d[f"kappa_{name}_phonix"] = phys["kappa_cal"]
    bad = (d.K_alignn_phonix <= MIN_MODULUS) | (d.G_alignn_phonix <= MIN_MODULUS)
    d.loc[bad, "kappa_ALIGNN_phonix"] = np.nan
    d["kappa_max3_phonix"] = d[[f"kappa_{m}_phonix" for m in MODELS]].max(axis=1)

    print("\nCELLS (volume per atom, over the main cell's):")
    for col, what in [("vol_ratio_phonix", "PhoNIX cell / main cell"),
                      ("vol_ratio_mp_pbe", "MP's PBE cell / matbench cell"),
                      ("vol_ratio_mp_r2scan", "MP's r2SCAN cell / matbench cell")]:
        v = d[col].dropna()
        print(f"  {what:34s} n={len(v):2d}  median {v.median():.3f}  range {v.min():.3f}-{v.max():.3f}")
    same_n = (d.n_atoms_phonix == d.n_atoms) | d.n_atoms_phonix.isna()
    print(f"  atoms per cell the same in PhoNIX's and the main cell: {same_n.sum()}/{len(d)}"
          + ("" if same_n.all() else f"  (differ: {d.formula[~same_n].tolist()})"))
    print("WHAT THE PHONIX CELL DOES (median ratio PhoNIX cell / main cell, 26 crystals):")
    for name, (kc, gc) in MODELS.items():
        r = d.dropna(subset=[f"{kc}_phonix"])
        print(f"  {name:10s} K x{np.median(r[f'{kc}_phonix'] / r[kc]):.3f}   "
              f"G x{np.median(r[f'{gc}_phonix'] / r[gc]):.3f}   "
              f"kappa x{np.nanmedian(r[f'kappa_{name}_phonix'] / r[f'kappa_{name}']):.3f}")

    # Cs2ZrBr6 has no structure anywhere; keep it as an empty row so the table is complete
    gone = fam[~fam.mp_id.isin(d.mp_id)][["formula", "in_cao_18"]].assign(
        structure_from="none: not in PhoNIX or the Materials Project")
    d = pd.concat([d, gone], ignore_index=True)
    d.to_csv(OUT_KAPPA, index=False)

    have = d.dropna(subset=["kappa_max3"])
    print(f"\nKAPPA (physics chain), {len(have)} crystals with a structure")
    for name in list(MODELS) + ["max3", "max3_phonix"]:   # max3_phonix = the same, on PhoNIX's cells
        col = f"kappa_{name}"
        cao = have[have.in_cao_18 == True]                 # noqa: E712 - the column holds True/False
        ph = have.dropna(subset=["phonix_klat"])
        agree_ph = ((ph[col] <= LOW) == (ph.phonix_klat <= LOW)).sum()
        print(f"  {name:10s} calls <= 1: {(have[col] <= LOW).sum()}/{have[col].notna().sum()};  "
              f"Cao's compounds: {(cao[col] <= LOW).sum()}/{len(cao)} (Cao: all below 1);  "
              f"same call as PhoNIX: {agree_ph}/{len(ph)}")
    for name in DFT:
        col = f"kappa_DFT {name}"
        sub = have.dropna(subset=[col])
        print(f"  chain on {name:8s} DFT moduli: {(sub[col] <= LOW).sum()}/{len(sub)} <= 1")

    # the two compounds with a literature kappa
    show = ["formula", "phonix_klat", "lit_measured", "lit_DFT_cao", "lit_DFT_other",
            "kappa_ALIGNN", "kappa_CGCNN-ens", "kappa_newbase", "kappa_newbase_oof", "kappa_max3_phonix",
            "kappa_DFT matbench", "kappa_DFT MP now", "kappa_DFT Bhumla"]
    lit = d[d.lit_DFT_cao.notna() | d.lit_measured.notna()]
    print("\nCompounds with a literature kappa (chain kappa on each set of moduli):")
    print(lit[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    mods = ["formula", "matbench_split", "K_matbench", "K_mp_now", "K_bhumla", "K_alignn", "K_cgcnn",
            "K_newbase_oof", "G_matbench", "G_mp_now", "G_bhumla", "G_alignn", "G_cgcnn", "G_newbase_oof"]
    print("\nModuli (GPa) for Bhumla's four:")
    print(d[d.K_bhumla.notna()][mods].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ---- figure -----------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])   # left: K over G; right: kappa
    colours = {"ALIGNN": "tab:blue", "CGCNN-ens": "tab:orange", "newbase_oof": "tab:green"}
    for row, t in enumerate(["K", "G"]):
        ax = fig.add_subplot(grid[row, 0])
        sub = long[long.dft_source == "matbench"]
        for pname, c in colours.items():
            s = sub[sub.predictor == pname]
            for held, face in [(True, c), (False, "none")]:   # filled = held out, hollow = seen
                x = s[s.held_out == held]
                ax.scatter(x[f"{t}_dft"], x[f"{t}_pred"], s=36, facecolors=face, edgecolors=c,
                           label=f"{pname}, {'held out' if held else 'seen in training'}"
                           if len(x) else None)
        # DFT against DFT: Bhumla (x) against the Materials Project today (y)
        b = d.dropna(subset=[f"{t}_bhumla", f"{t}_mp_now"])
        ax.scatter(b[f"{t}_bhumla"], b[f"{t}_mp_now"], marker="*", s=140, c="k",
                   label="MP DFT (y) vs Bhumla DFT (x)")
        for _, r in b.iterrows():
            ax.annotate(r.formula, (r[f"{t}_bhumla"], r[f"{t}_mp_now"]), fontsize=7,
                        xytext=(4, -9), textcoords="offset points")
        lims = np.array([2.0, 25.0])
        ax.plot(lims, lims, "k-", lw=0.8)
        ax.fill_between(lims, lims / 1.25, lims * 1.25, color="0.9", zorder=0, label="within x1.25")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"DFT {t} (GPa): matbench label, or Bhumla for the stars")
        ax.set_ylabel(f"predicted {t} (GPa), on the matbench cell")
        ax.set_title(f"{'AB'[row]}. {'Bulk' if t == 'K' else 'Shear'} modulus, A2BX6 family")
        if row == 0:
            ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(grid[:, 1])
    p = have.sort_values("kappa_max3").reset_index(drop=True)
    y = np.arange(len(p))
    ax.scatter(p.kappa_max3, y, c="tab:red", s=30, label="our screen (max of 3 models)", zorder=3)
    ax.scatter(p.kappa_max3_phonix, y, facecolors="none", edgecolors="tab:red", s=30,
               label="our screen, on PhoNIX's cell", zorder=3)
    ax.scatter(p["kappa_DFT MP now"], y, marker="s", facecolors="none", edgecolors="k", s=30,
               label="chain on MP-now DFT moduli")
    ax.scatter(p.phonix_klat, y, marker="D", c="tab:purple", s=26, label="PhoNIX phonon DFT")
    ax.scatter(p.lit_measured, y, marker="*", c="gold", edgecolors="k", s=180, label="measured (Bhui 2022)")
    ax.scatter(p.lit_DFT_cao, y, marker="^", c="tab:cyan", edgecolors="k", s=50, label="Cao 2026 DFT")
    ax.axvline(LOW, color="k", ls="--", lw=0.8)
    names = [f"{f}{' (Cao)' if c == True else ''}{' [MP]' if s == 'Materials Project' else ''}"  # noqa: E712
             for f, c, s in zip(p.formula, p.in_cao_18, p.structure_from)]
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("kappa_L at 300 K (W/m/K)")
    ax.set_title("C. kappa: our chain, the chain on DFT moduli, PhoNIX, literature\n"
                 "(Cao) = one of Cao's 18, all below 1 in Cao's DFT\n[MP] = cell from MP, not matbench")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {OUT_COMPARE}\nwrote {OUT_KAPPA}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "torch", "newbase", "score"])
    stage = ap.parse_args().stage
    # a dict of functions: look the stage name up, then call what comes back
    {"fetch": stage_fetch, "torch": stage_torch, "newbase": stage_newbase, "score": stage_score}[stage]()
