#!/usr/bin/env python3
"""
STEP 41 - Screen JARVIS-DFT's perovskites, since GNoME has none.
================================================================================

    python scripts/cgcnn/41_screen_jarvis_perovskites.py

WHY THIS SCRIPT EXISTS
------------------------
40_sort_by_family.py's own numbers made this necessary: GNoME's 33,118
filtered candidates contain ZERO ABO3-type perovskites (and zero rock-salt,
zero fluorite - GNoME's discovery method hunts NOVEL multi-cation
substitutions, so well-known simple structures are already in the reference
hull and never come back as "new" candidates). Perovskites exist only in
this project's TRAINING data (300 in matbench, 120 in AFLOW) - never
screened, only ever used to fit and validate the model.

JARVIS-DFT (`jarvis.db.figshare.data("dft_3d")`, already verified working
and documented in the materials-dft-databases memory) has 93,902 entries and
carries full structures, not just a summary table - 1,635 of them match the
ABO3 stoichiometric pattern (918 unique formulas, after the same
molecular-anion exclusion 40_sort_by_family.py applies - a raw formula match
alone would double-count carbonates/nitrates/tellurites the way the
perovskite count did earlier in this project's history). Of those, 1,467
also pass the structural check below - roughly 5x matbench's and 12x
AFLOW's own VERIFIED perovskite count, in one bulk download, with no AFLUX
paging or rate-limit failures to work around (see 33_fetch_aflow_gamma.py's own
docstring for what that cost the AFLOW route).

WHY THE ROUND-9 MODEL, NOT JARVIS'S OWN K/G VALUES
-------------------------------------------------------
This project already checked JARVIS-DFT's own elastic moduli against
matbench's and found the disagreement WORSE than AFLOW's (0.127/0.155
log10 K/G vs AFLOW's 0.094/0.109 - see materials-dft-databases memory).
This script does not use JARVIS's reported K/G/kappa at all - only its
STRUCTURES. Every predicted number here comes from this project's own
round-9 model (scripts/cgcnn/37_train_gamma.py, trained on AFLOW's real
gamma), self-consistently, exactly the way 39_screen_gnome_gamma.py scores
GNoME - reused directly (load_gamma_model / predict_gamma_model /
ensemble_and_score are imported from that script, not reimplemented).

WHAT "ALREADY KNOWN" MEANS IN THE OUTPUT
--------------------------------------------
A handful of JARVIS perovskites share a formula with something already in
matbench or AFLOW's training set (different DFT calculation, same
composition - not the same crystal). Those rows are flagged
`in_training_set=True`, not excluded - screening a composition we also have
a training label for is a free sanity check on the model's own prediction,
not a reason to skip it.

STOICHIOMETRY ALONE IS NOT ENOUGH - A SECOND, STRUCTURAL CHECK
--------------------------------------------------------------------
The first version of this script trusted the 1:1:3 ratio alone and its own
top-ranked "lowest kappa perovskite" was SrTeO3/CaTeO3 - tellurites, not
perovskites (Te's stereochemically active lone pair gives 3-coordinate
pyramidal TeO3 groups, confirmed via CrystalNN, not a 6-coordinate BO6
octahedron). Excluding Te from ANION_FORMERS fixed that, but a second,
sharper problem surfaced even after that fix: the SAME formula can be a
genuine perovskite in one JARVIS entry and a completely different structure
in another - KAsO3 (JVASP-117155, triclinic P-1, As 4-coordinate) is not a
perovskite, while NaAsO3 (JVASP-38495, cubic Pm-3m, As 6-coordinate) is,
despite identical stoichiometry. No formula-level exclusion list can catch
a per-polymorph difference like that. Every candidate is therefore checked
structurally, not just chemically: `verified_perovskite` is True only when
at least one of the two cations sits in 6-fold coordination (CrystalNN),
the one structural signature common to every perovskite distortion
(cubic, orthorhombic, rhombohedral, tetragonal Glazer tilt systems all
keep the BO6 octahedron - only its tilting changes). Ranking and reporting
should use the verified subset; the unverified rows are kept in the output,
not deleted, so the difference stays inspectable rather than asserted.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- source -----------------------------------------------------------
    "jarvis_dataset": "dft_3d",              # jarvis.db.figshare dataset name
    "results_dir": "results/cgcnn",           # where round-9 checkpoints live and this script's own output lands
    "families_dir": "results/families",       # mirrors the low-kappa/full tables into the sorted family directory too

    # ---- the round-9 gamma model (identical to 39_screen_gnome_gamma.py) -----
    "r9_tags": "r9_full_s42,r9_full_s1,r9_full_s2",
    "kappa_threshold": 1.0,   # W/m/K

    # ---- featurisation (MUST match what the round-9 model was trained on) ----
    "max_num_nbr": 12,
    "radius": 8,
    "step": 0.2,

    # ---- cross-reference against this project's own training data -----------
    "matbench_labels_csv": "data_full/labels.csv",
    "aflow_labels_csv": "data_full/gamma_labels.csv",
}
# =============================================================================

import os                 # path joining/creation
import sys                 # sys.path mutation
import time                 # wall-clock timing for progress printouts
import warnings              # silences pymatgen/jarvis warnings
from importlib import import_module   # loads numeric-prefixed sibling scripts by string name

import torch  # noqa: F401   # torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard)
import numpy as np                          # array math for the ensemble step (via the imported ensemble_and_score)
import pandas as pd                          # DataFrame construction, CSV I/O
from pymatgen.core import Composition        # formula parsing for the perovskite/anion-former check
from pymatgen.analysis.local_env import CrystalNN   # bond-graph coordination number, the structural perovskite check

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of caller's cwd

from cgcnn_scratch.data import AtomFeaturiser, GaussianDistance, structure_to_graph   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # makes sibling numeric-prefixed scripts importable by name
_predict = import_module("04_predict_moduli")           # PredictionSet - the (graph, dummy target, id) Dataset wrapper
_screen39 = import_module("39_screen_gnome_gamma")       # load_gamma_model / predict_gamma_model / ensemble_and_score

# Same exclusion set as 40_sort_by_family.py's classify_family(): elements
# that form a molecular anion (carbonate, nitrate, sulfate, selenite,
# tellurite, borate) or a second halide anion alongside oxygen, rather than
# a genuine second cation - without this, SrCO3/LiNO3-style false positives
# sneak in under the ABO3 ratio the same way they did earlier in this
# project. Te is here because this script's own first run flagged SrTeO3/
# CaTeO3 as the two best low-kappa "perovskites" - both are tellurites on
# inspection (Te coordination number 3, space groups P2_1/c and P1, not a
# perovskite's 6-coordinate BO6 octahedra in a cubic/orthorhombic/
# rhombohedral cell), caught before being reported, not after.
ANION_FORMERS = {"C", "N", "S", "P", "Se", "Te", "B", "F", "Cl", "Br", "I"}


def is_real_perovskite(formula):
    try:
        comp = Composition(formula)
    except Exception:
        return False
    elements = {str(el) for el in comp.elements}
    if "O" not in elements:
        return False
    amounts = comp.reduced_composition.get_el_amt_dict()
    o_amt = amounts.pop("O", 0)
    if set(amounts) & ANION_FORMERS:
        return False
    return len(amounts) == 2 and sorted(amounts.values()) == [1, 1] and o_amt == 3


_CNN = CrystalNN()   # one shared instance - construction has its own overhead, reuse it across every structure

# The classic large, high-coordinate A-site cations in ABO3 perovskites -
# alkali metals, alkaline earths, lanthanides, and a handful of heavy
# post-transition/actinide elements that commonly take the A-site instead
# (Bi, Pb, Tl, Ag, Y, Th, U). Requiring 6-fold coordination on WHICHEVER
# cation is not in this set is what actually distinguishes a perovskite -
# checking "either cation" is not enough: KAsO3 (JVASP-117155, triclinic,
# not a perovskite) has K sitting in an incidental 6-fold site while As
# itself is only 4-coordinate, and an "either" check wrongly passed it.
A_SITE_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Bi", "Pb", "Tl", "Ag", "Y", "Th", "U",
}


def has_sixfold_b_site(structure):
    """True only when the non-A-site cation (the real perovskite B-site) is
    itself 6-fold coordinated - the one structural feature every perovskite
    distortion (cubic, orthorhombic, rhombohedral, tetragonal Glazer tilt
    systems) keeps regardless of octahedral tilting. If BOTH cations happen
    to be outside A_SITE_ELEMENTS (rare for a real perovskite, e.g. a
    double transition-metal oxide), both are checked and either counts."""
    cations = {str(el) for el in structure.composition.elements if str(el) != "O"}
    b_candidates = cations - A_SITE_ELEMENTS or cations   # prefer non-A-site cations; fall back to checking both if that set is empty
    for element in b_candidates:
        site_index = next(i for i, s in enumerate(structure) if s.specie.symbol == element)   # first site of this element
        try:
            if len(_CNN.get_nn_info(structure, site_index)) == 6:
                return True
        except Exception:
            continue   # a pathological/disordered site fails silently here - treated as "not verified", not a crash
    return False


def main():
    results_dir = os.path.join(PROJECT_ROOT, CONFIG["results_dir"])
    families_dir = os.path.join(PROJECT_ROOT, CONFIG["families_dir"], "perovskite-like-abo3")
    os.makedirs(families_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 41 - screening JARVIS-DFT's perovskites with the round-9 model")
    print("=" * 78)

    # ---- pull JARVIS-DFT and filter to real perovskites ----------------------
    print(f"\nLoading JARVIS-DFT ({CONFIG['jarvis_dataset']})...")
    from jarvis.db.figshare import data as jarvis_data   # imported here, not at module top, to keep the import-order guard clean
    from jarvis.core.atoms import Atoms as JarvisAtoms
    entries = jarvis_data(CONFIG["jarvis_dataset"])
    print(f"  {len(entries)} total entries")

    perovskites = [e for e in entries if e.get("formula") and is_real_perovskite(e["formula"])]
    print(f"  {len(perovskites)} entries match the ABO3 pattern "
         f"({len({e['formula'] for e in perovskites})} unique formulas)")

    # ---- flag formulas already present in this project's own training data --
    mb_formulas = set(pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["matbench_labels_csv"]))["formula"])
    af_formulas = set(pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["aflow_labels_csv"]))["formula"])
    known_formulas = mb_formulas | af_formulas

    # ---- featurise every candidate --------------------------------------------
    print("\nFeaturising JARVIS structures...")
    ari = AtomFeaturiser(os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"))
    gdf = GaussianDistance(dmin=0, dmax=CONFIG["radius"], step=CONFIG["step"])

    graphs, ids, meta, failed = [], [], [], []
    start = time.time()
    for entry in perovskites:
        jid = entry["jid"]
        try:
            structure = JarvisAtoms.from_dict(entry["atoms"]).pymatgen_converter()   # JARVIS atoms dict -> pymatgen Structure
            graph = structure_to_graph(structure, ari, gdf, CONFIG["max_num_nbr"], CONFIG["radius"])
            primitive = structure.get_primitive_structure()   # smallest repeating cell, for volume/density/n_sites below
            verified = has_sixfold_b_site(primitive)          # the structural (not just stoichiometric) perovskite check
        except Exception as exc:
            failed.append((jid, str(exc)[:70]))
            continue
        graphs.append(graph)
        ids.append(jid)
        meta.append({
            "material_id": jid,
            "formula": entry["formula"],
            "Number of Atoms": len(primitive),
            "Volume (A3)": primitive.volume,
            "Density (g cm-3)": primitive.density,
            "in_training_set": entry["formula"] in known_formulas,
            "verified_perovskite": verified,
        })
        if (len(graphs)) % 400 == 0:   # coordination checks are the slow step here - show progress every 400
            print(f"  {len(graphs)}/{len(perovskites)} featurised+verified "
                 f"({len(graphs)/(time.time()-start):.0f}/s)")
    print(f"  featurised {len(graphs)}/{len(perovskites)} in {time.time() - start:.0f}s, "
         f"{sum(m['verified_perovskite'] for m in meta)} structurally verified")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")

    meta = pd.DataFrame(meta).set_index("material_id")
    dataset = _predict.PredictionSet(graphs, ids)

    # ---- run the round-9 ensemble, identical to 39_screen_gnome_gamma.py -----
    print(f"\nRunning the round-9 gamma-head ensemble ({CONFIG['r9_tags']})...")
    sample_shapes = (graphs[0][0].shape[-1], graphs[0][1].shape[-1])
    per_seed_preds = []
    for tag in CONFIG["r9_tags"].split(","):
        model, normalizer, _ = _screen39.load_gamma_model(tag.strip(), results_dir, sample_shapes)
        preds = _screen39.predict_gamma_model(model, normalizer, dataset)
        per_seed_preds.append(preds)
        print(f"  {tag.strip():<14} scored {len(preds)} crystals")

    print("\nCombining seeds into an ensemble point estimate + Monte Carlo band...")
    scored = _screen39.ensemble_and_score(per_seed_preds, meta)

    out = meta.join(scored, how="inner").reset_index()
    out = out.sort_values("Kappa_r9_p95").reset_index(drop=True)   # pessimistic-bound ranking, same convention as elsewhere

    all_path = os.path.join(results_dir, "41_jarvis_perovskite_screen_all.csv")
    out.to_csv(all_path, index=False)
    print(f"\nWrote {all_path} ({len(out)} scored perovskites)")

    candidates = out[out["Kappa_r9_gamma"] <= CONFIG["kappa_threshold"]].copy()
    cand_path = os.path.join(results_dir, "41_jarvis_perovskite_low_kappa.csv")
    candidates.to_csv(cand_path, index=False)
    print(f"kappa_L <= {CONFIG['kappa_threshold']} W/m/K: {len(candidates)} candidates "
         f"({candidates['verified_perovskite'].sum()} structurally verified)")
    print(f"Wrote {cand_path}")

    # ---- mirror into the sorted family directory ------------------------------
    out.to_csv(os.path.join(families_dir, "jarvis_screen_all.csv"), index=False)
    candidates.to_csv(os.path.join(families_dir, "jarvis_screen_low_kappa.csv"), index=False)
    print(f"Mirrored both into {os.path.relpath(families_dir, PROJECT_ROOT)}/")

    verified = out[out["verified_perovskite"]]
    print(f"\n{len(verified)}/{len(out)} candidates are structurally verified "
         f"(6-fold cation coordination) - ranking below is restricted to those.")

    print("\n" + "=" * 78)
    print("  TOP 15 VERIFIED PEROVSKITES, BY POINT-ESTIMATE KAPPA")
    print("=" * 78)
    print(verified.sort_values("Kappa_r9_gamma")[
        ["material_id", "formula", "Kappa_r9_gamma", "Kappa_r9_p05", "Kappa_r9_p95", "in_training_set"]
    ].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
