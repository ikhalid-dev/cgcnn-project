#!/usr/bin/env python
"""
Step 99 - literature low-kappa crystals: what does our screen predict for them?
================================================================================

Four runs, two Python environments, in this order (each stage reads what the one before wrote):

    # 1. Materials Project cells + elasticity, and the matbench match (internet + MP key; ~1 min)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/99_literature_low_kappa.py --stage fetch
    # 2. ALIGNN + CGCNN-ens moduli (torch only works in infer_env; a few minutes)
    ~/miniconda3/envs/infer_env/bin/python scripts/cgcnn/99_literature_low_kappa.py --stage torch
    # 3. newbase moduli (matminer + xgboost, ml_env; a few minutes - CrystalNN is slow on 84-atom cells)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/99_literature_low_kappa.py --stage newbase
    # 4. kappa, comparison with the papers, figure (ml_env; seconds)
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/99_literature_low_kappa.py --stage score

WHY
---
2026-10-07: the user asked to "take low thermal conductivity materials that are cited in
the literature and predict kL for it". The screen was built to FIND low-kappa crystals,
so the fair test is: given crystals the literature already knows are low, does the
screen call them low, and how close is its number? The source rule chosen by the
user: MEASURED kappa first; values that exist only as DFT go in a separate, flagged
column; every value carries a DOI and a verbatim quote; nothing unsourced.

THE LIST (LIT below) - how it was built
---------------------------------------
Every number was read from the paper's ABSTRACT, fetched by DOI from OpenAlex,
Semantic Scholar or arXiv on 2026-10-07, and the quote is copied into the table.
Nothing was read off a figure. A compound is predicted only if the Materials Project
has a cell of the polymorph that was measured (space group checked in stage 1).
  15 measured, with a structure   the scored set
   2 measured, no structure       kept as rows, no prediction (Bi4O4SeCl2, Cs3Bi2I6Cl3)
   3 DFT only                     predicted, compared with the DFT number, flagged
Left out on purpose:
  CsPbI3 (Lee 2017)   the abstract does not say which phase was measured; at room
                      temperature CsPbI3 is usually the yellow non-perovskite phase,
                      so the cell to use is a guess
  SnSe, AgCrSe2       the abstracts give kappa only at 973 K / 500 K
  BiCuSeO, Cu3SbSe3,  no single room-temperature lattice value in the abstract
  Cu12Sb4S13
  Ag8SnSe6 (2023)     "<0.5" is an upper bound; the 2016 paper's value is used instead

WHAT IS COMPARED, per compound
------------------------------
  measured kappa        kappa_meas (W/m/K), the value in the paper's abstract (anisotropic
                        crystals: the orientation average (k_par + 2 k_perp)/3)
  literature DFT        kappa_lit_dft, flagged, never mixed with the measured column
  our screen            the physics chain on each model's K and G (ALIGNN, CGCNN-ens,
                        newbase; newbase_oof where in matbench); kappa_max3 = the
                        screen's rule (largest of the three)
  chain on DFT moduli   the same chain on the Materials Project's DFT K and G (and the
                        matbench label where the crystal is in matbench). This is what
                        the supervisor's DFT-moduli route will produce, so it says
                        whether a better K and G would fix the number
  PhoNIX                PhoNIX's phonon-DFT kappa_L where the same mp_id is in PhoNIX

WHICH CELL THE MODELS SEE: matbench's cell if the crystal is in matbench (matched as
step 85 does: same reduced formula, StructureMatcher), else the Materials Project's
PBE cell - never MP's r2SCAN default, which is ~6% smaller (memory:
structure-cells-and-checkpoints).

HELD OUT, OR NOT (as step 97)
  ALIGNN, CGCNN-ens  held out if not in matbench, or in matbench's test split
  newbase            held out only if not in matbench; for matbench crystals the
                     out-of-fold prediction (newbase_oof) is the held-out one

WRITTEN BEFORE THE MODELS WERE RUN (the measured values were known, the predictions were not)
-------------------------------------------------------------------------------
kappa_max3 <= 1 for at least 12 of the 15 measured compounds
    -> the screen would have flagged the literature's known low-kappa crystals
median kappa_max3 / measured within x2 (0.5-2)
    -> the chain gets the size right, not only the side of 1
median above x2
    -> the chain over-predicts these crystals: the Poisson-ratio gamma does not see
       rattling or lone-pair anharmonicity, which is WHY they are low
the chain on DFT moduli about as far from measured as the models' chain
    -> the error is in the physics chain, not the ML; DFT moduli will not fix
       kappa for compounds like these (worth telling the supervisor)

RESULT (2026-10-07, written after the run)
------------------------------------------
                                 called <= 1   median pred/measured   within x2
    max3 (the screen's rule)       11/15            x1.20               11/15
    max3, networks held out         8/10            x1.05                8/10
    CGCNN-ens alone / newbase      12/15        x0.72 / x0.86       8/15 / 10/15
  1. ">= 12 of 15 called low": MISSED BY ONE (11/15). The four misses:
       Tl3VSe4  7.4 vs 0.30. matbench's DFT label for it is K 149, G 42 GPa; MP's current
                DFT is K 19.5, G 7.4 (x7.6, x5.7 lower). The networks learnt the bad label.
                The chain on MP's current moduli gives 0.32 - the measurement.
       InTe     1.26 vs 0.40. The chain on DFT moduli gives 1.63: the chain itself misses
       AgSbTe2  1.76 vs 0.70 (total). DFT moduli give 1.49: the chain misses (and MP's cell
                is an ordered model of a disordered crystal)
       TlInTe2  1.65 vs 0.50: ALIGNN only; CGCNN-ens 0.69 and newbase 0.86 call it low
  2. "median within x2": YES (x1.20; x1.05 held out). The size is right for most.
  3. "chain on DFT moduli as far off as the models": MIXED, 5 crystals with MP moduli.
       DFT moduli FIX it where the model was wrong (Tl3VSe4 x24.7 -> x1.06; Cs2SnI6
       x1.71 -> x0.82); they do NOT fix InTe (x4.1) or AgSbTe2 (x2.1), where the
       Slack/Poisson chain itself is too high - better K, G cannot help there.
  DFT-only rows: Cs2AgBiBr6 max3 0.95 vs the paper's DFT 0.21 (x4.5); the other two within x1.3.

OUTPUTS
    results/cgcnn/99_structures.json         stage 1: MP PBE cells, matbench cells + ids
    results/cgcnn/99_mp_elasticity.csv        stage 1: MP's DFT K, G (where MP has them)
    results/cgcnn/99_torch_moduli.csv         stage 2
    results/cgcnn/99_newbase_moduli.csv       stage 3
    results/cgcnn/99_literature_low_kappa.csv stage 4: one row per compound, every source
    results/cgcnn/99_literature_low_kappa.png stage 4
"""
import os
import sys

# torch must be imported BEFORE numpy, or the two OpenMP runtimes clash (steps 87, 97).
# sys.argv is the list of words typed after "python".
if "torch" in " ".join(sys.argv[1:]):
    import torch  # noqa: F401
else:
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # xgboost crashes here with >1 OpenMP thread

import argparse                       # reads "--stage torch" from the command line
import json
import warnings
from importlib import import_module   # imports a file whose name starts with a digit

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")     # pymatgen and matminer print a lot of noise
from pymatgen.core import Composition, Structure  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [ROOT, os.path.join(ROOT, "scripts", "cgcnn"), os.path.join(ROOT, "scripts", "alignn")]:
    sys.path.insert(0, p)              # so steps 04, 15, 71, 85, 87 can be imported

# =============================================================================
#  CONFIG
# =============================================================================
RES = os.path.join(ROOT, "results", "cgcnn")
LABELS = os.path.join(ROOT, "data_full", "labels.csv")            # matbench DFT K_VRH, G_VRH
CHAIN82 = os.path.join(RES, "82_test_predictions.csv")            # its mb_ids = the test split
PHONIX = os.path.expanduser("~/Desktop/a new project/data/phonix/data_all.csv")
OOF = os.path.expanduser("~/Desktop/pink_reproduction/xgboost_oof/"
                         "matbench_log_{}_composition+structure+angular_oof.csv")
STEP97_TORCH = os.path.join(RES, "97_torch_moduli.csv")           # Cs2SnI6 is in both steps:
STEP97_NEWBASE = os.path.join(RES, "97_newbase_moduli.csv")       # its numbers must come back
STEP97_MB = os.path.join(RES, "97_matbench_structures.json")

STRUCT = os.path.join(RES, "99_structures.json")
MP_ELAST = os.path.join(RES, "99_mp_elasticity.csv")
TORCH_OUT = os.path.join(RES, "99_torch_moduli.csv")
NEWBASE_OUT = os.path.join(RES, "99_newbase_moduli.csv")
OUT_CSV = os.path.join(RES, "99_literature_low_kappa.csv")
OUT_PNG = os.path.join(RES, "99_literature_low_kappa.png")

LOW = 1.0               # W/m/K, the screen's cutoff
MIN_MODULUS = 0.01      # GPa. Step 57's rule: an ALIGNN K or G below this is not a prediction
ALIGNN_TOL = 1e-2       # ALIGNN reproduces to ~1% (neighbour ties), CGCNN to 1e-4
TIE_JITTER = 1e-5       # Angstrom: nudge size that exposes ALIGNN's neighbour ties (step 97)
N_JITTER = 8

# The literature. One dict per compound; dict(a=1, b=2) is the same as {"a": 1, "b": 2}.
#   mp_id, sg     the Materials Project cell of the MEASURED polymorph and its space-group number
#   kind          "measured" or "DFT only"
#   kappa         the single number compared with the predictions (W/m/K)
#   lo, hi        the range the paper gives, where it gives one (drawn as a bar)
#   quantity      "lattice" if the paper says lattice; "total" if it measured total kappa
#                 (total >= lattice, so for a metal-like sample it is an upper bound)
#   quote         copied from the abstract, so the number can be checked without the paper
LIT = [
    dict(formula="Tl3VSe4", mp_id="mp-1025549", sg=217, kind="measured", kappa=0.30,
         T="300 K", quantity="total", form="crystal",
         ref="Science (2018)", doi="10.1126/science.aar8072",
         quote="a calculated phonon κ [0.16 Watts per meter-Kelvin (W/m-K)] one-half that of "
               "our measured κ (0.30 W/m-K) at 300 K",
         dft=0.16, dft_doi="10.1126/science.aar8072",
         caveat="the paper's DFT (3-phonon) is half the measured value"),
    dict(formula="CsSnI3", mp_id="mp-568570", sg=62, kind="measured", kappa=0.38, lo=0.34, hi=0.42,
         T="not stated in abstract", quantity="lattice", form="nanowire",
         ref="PNAS (2017)", doi="10.1073/pnas.1711744114",
         quote="ultralow lattice thermal conductivities of solution-synthesized, single-crystalline "
               "all-inorganic halide perovskite nanowires composed of CsPbI3 (0.45 ± 0.05 W·m−1·K−1), "
               "CsPbBr3 (0.42 ± 0.04 W·m−1·K−1), and CsSnI3 (0.38 ± 0.04 W·m−1·K−1)",
         perovskite=True,
         caveat="nanowires (surfaces can lower kappa below bulk); black perovskite phase "
                "(mp-568570), not MP's yellow 1D Pnma phase mp-27381"),
    dict(formula="CsPbBr3", mp_id="mp-567629", sg=62, kind="measured", kappa=0.42, lo=0.38, hi=0.46,
         T="not stated in abstract", quantity="lattice", form="nanowire",
         ref="PNAS (2017)", doi="10.1073/pnas.1711744114",
         quote="(same sentence as CsSnI3) ... CsPbBr3 (0.42 ± 0.04 W·m−1·K−1) ...",
         perovskite=True,
         caveat="nanowires; orthorhombic perovskite (mp-567629), not MP's non-perovskite Pnma mp-567681"),
    dict(formula="InTe", mp_id="mp-20320", sg=140, kind="measured", kappa=0.4,
         T="not stated in abstract", quantity="lattice", form="not stated in abstract",
         ref="Angew. Chem. Int. Ed. (2016)", doi="10.1002/anie.201511737",
         quote="an ultralow lattice thermal conductivity (ca. 0.4 W m(-1) K(-1) ) in mixed valent InTe",
         caveat="the ambient TlSe-type cell (I4/mcm); MP's lowest-energy InTe is rocksalt, "
                "a high-pressure phase"),
    dict(formula="TlInTe2", mp_id="mp-22791", sg=140, kind="measured", kappa=0.5,
         T="near room temperature", quantity="lattice", form="not stated in abstract",
         ref="J. Am. Chem. Soc. (2017)", doi="10.1021/jacs.7b01434",
         quote="exhibits lattice thermal conductivity as low as ca. 0.5 W/mK near room temperature"),
    dict(formula="TlSe", mp_id="mp-1836", sg=140, kind="measured", kappa=0.62,
         T="295 K (0.62 falls to 0.4 by 525 K)", quantity="lattice", form="not stated in abstract",
         ref="J. Am. Chem. Soc. (2019)", doi="10.1021/jacs.9b10551",
         quote="exhibits intrinsically ultralow lattice thermal conductivity (κ L ) of 0.62–0.4 W/mK "
               "in the range 295–525 K"),
    dict(formula="CsAg5Te3", mp_id="mp-9206", sg=136, kind="measured", kappa=0.18,
         T="not stated in abstract", quantity="lattice", form="not stated in abstract",
         ref="Angew. Chem. Int. Ed. (2016)", doi="10.1002/anie.201605015",
         quote="exhibits ultralow lattice thermal conductivity (ca. 0.18 Wm(-1) K(-1) ) and a high "
               "figure of merit of about 1.5 at 727 K"),
    dict(formula="Tl9BiTe6", mp_id="mp-34361", sg=87, kind="measured", kappa=0.39,
         T="300 K", quantity="total", form="not stated in abstract",
         ref="Phys. Rev. Lett. (2001)", doi="10.1103/PhysRevLett.86.4350",
         quote="the extremely low thermal conductivity of Tl9BiTe6 [0.39 W/(m K) at 300 K]",
         caveat="TOTAL kappa, so lattice kappa is lower; MP's cell is a computed ordered model "
                "(the real crystal shares one site between Tl and Bi)"),
    dict(formula="AgSbTe2", mp_id="mp-12360", sg=166, kind="measured", kappa=0.70,
         T="80-300 K", quantity="total", form="crystals",
         ref="Phys. Rev. Lett. (2008)", doi="10.1103/PhysRevLett.101.035901",
         quote="The thermal conductivity is temperature independent from 80 to 300 K at a value of "
               "approximately 0.70 W/mK. Heat conduction is dominated by the lattice term",
         caveat="the real crystal is cation-DISORDERED rocksalt; MP's R-3m is an ordered model"),
    dict(formula="Cs2SnI6", mp_id="mp-27636", sg=225, kind="measured", kappa=0.29, lo=0.22, hi=0.29,
         T="range not stated in abstract (0.29 = the high end)", quantity="lattice",
         form="not stated in abstract",
         ref="Chem. Mater. 34, 3301 (2022)", doi="10.1021/acs.chemmater.2c00084",
         quote="an ultralow lattice thermal conductivity (κ lat ∼0.29–0.22 W/m·K) in an air-stable "
               "vacancy-ordered double perovskite Cs2SnI6",
         caveat="the value step 96 used; also in steps 96-97, in matbench TRAINING"),
    dict(formula="Cs3Bi2I9", mp_id="mp-624214", sg=194, kind="measured", kappa=(0.3 + 2 * 0.17) / 3,
         lo=0.17, hi=0.3, T="room temperature", quantity="total", form="single crystal",
         ref="Natl. Sci. Open (2025)", doi="10.1360/nso/20250056",
         quote="Along the crystallographic c-axis direction, the particle-behaviordominated thermal "
               "conductivity (k) is approximately 0.3 W/m-K at room temperature ... the thermal "
               "transport perpendicular to the c-direction is primarily wave-like behavior, "
               "exhibiting a room-temperature k of 0.17 W/m-K",
         caveat="anisotropic: compared value = (0.3 + 2 x 0.17)/3, the orientation average"),
    dict(formula="CsAg2I3", mp_id="mp-23496", sg=62, kind="measured", kappa=0.155, lo=0.15, hi=0.16,
         T="170-400 K", quantity="not stated in abstract", form="not stated in abstract",
         ref="arXiv:2511.21172 (2025), PREPRINT", doi="10.48550/arXiv.2511.21172",
         quote="experimentally confirming record-low thermal conductivity values of 0.15-0.16 W/m/K "
               "from 170 to 400 K in the halide metal CsAg2I3",
         caveat="preprint, not yet peer reviewed"),
    dict(formula="Ag8SnSe6", mp_id="mp-17984", sg=31, kind="measured", kappa=0.2,
         T="'the entire temperature range'", quantity="lattice", form="not stated in abstract",
         ref="Adv. Sci. (2016)", doi="10.1002/advs.201600196",
         quote="the lattice thermal conductivity is found to be as low as 0.2 W m−1 K−1 in the "
               "entire temperature range",
         caveat="Ag-ion conductor (argyrodite); 'as low as' = the lowest value"),
    dict(formula="Re6Se8Te7", mp_id="mp-667286", sg=61, kind="measured", kappa=0.32, lo=0.30, hi=0.34,
         T="room temperature", quantity="total", form="single crystal",
         ref="Appl. Phys. Rev. (2026)", doi="10.1063/5.0309916",
         quote="At room temperature, κ is 0.32 ± 0.02 and 0.53 ± 0.02 W m−1 K−1 in Re6Se8Te7 and "
               "Re6Te15, respectively",
         caveat="84-atom cell; MP puts it 0.072 eV/atom above the hull"),
    dict(formula="Re6Te15", mp_id="mp-583070", sg=61, kind="measured", kappa=0.53, lo=0.51, hi=0.55,
         T="room temperature", quantity="total", form="single crystal",
         ref="Appl. Phys. Rev. (2026)", doi="10.1063/5.0309916",
         quote="(same sentence as Re6Se8Te7) ... 0.53 ± 0.02 ... in ... Re6Te15",
         caveat="84-atom cell (MP lists it as Re2Te5)"),
    # ---- measured, but no Materials Project structure: kept as rows, not predicted ----
    dict(formula="Bi4O4SeCl2", mp_id=None, sg=None, kind="measured", kappa=0.1,
         T="not stated in summary", quantity="not stated", form="crystal",
         ref="Science (2021)", doi="10.1126/science.abh1619",
         quote="push the conductivity down to 0.1 watts per kelvin per meter",
         caveat="quote from the journal's editor summary; not in the Materials Project"),
    dict(formula="Cs3Bi2I6Cl3", mp_id=None, sg=None, kind="measured", kappa=0.20,
         T="room temperature", quantity="lattice", form="single crystal",
         ref="Nat. Commun. (2022)", doi="10.1038/s41467-022-32773-4",
         quote="we demonstrate an ultralow (~0.20 W/m·K at room temperature) and glass-like temperature "
               "dependence (2–400 K) of κL in a single crystal of layered halide perovskite, Cs3Bi2I6Cl3",
         caveat="not in the Materials Project"),
    # ---- DFT only: no measurement found; flagged, compared with the DFT number ----
    dict(formula="Cs2AgBiBr6", mp_id="mp-1078250", sg=225, kind="DFT only", kappa=None,
         T="room temperature", quantity="lattice (computed)", form="-",
         ref="npj Comput. Mater. (2024)", doi="10.1038/s41524-024-01211-y",
         quote="An ultra-low thermal conductivity at room temperature (~0.21 Wm−1 K−1) is predicted",
         dft=0.21, dft_doi="10.1038/s41524-024-01211-y",
         caveat="DFT only (self-consistent phonons + wave-like term)"),
    dict(formula="Cs3Cu2I5", mp_id="mp-23481", sg=62, kind="DFT only", kappa=None,
         T="room temperature", quantity="lattice (computed)", form="-",
         ref="Appl. Phys. Lett. (2025)", doi="10.1063/5.0301860",
         quote="Cs3Cu2I5 possesses an ultralow lattice thermal conductivity (κL) of 0.126 W/(m K) at "
               "room temperature (RT)",
         dft=0.126, dft_doi="10.1063/5.0301860",
         caveat="DFT only; an earlier DFT paper (10.1038/s41524-021-00521-9) says <0.1"),
    dict(formula="CsCu2I3", mp_id="mp-23431", sg=63, kind="DFT only", kappa=None,
         T="300 K", quantity="lattice (computed)", form="-",
         ref="arXiv:2310.13680 (2023), PREPRINT", doi="10.48550/arXiv.2310.13680",
         quote="we predict an ultra-low thermal conductivity of 0.362 Wm^(-1) K^(-1) along the chain "
               "axis and 0.201 Wm^(-1) K^(-1) along cross chain direction in CsCu2I3 at 300 K",
         dft=(0.362 + 2 * 0.201) / 3, dft_doi="10.48550/arXiv.2310.13680",
         caveat="DFT only, preprint; anisotropic: (0.362 + 2 x 0.201)/3"),
]

MODELS = {"ALIGNN": ("K_alignn", "G_alignn"),
          "CGCNN-ens": ("K_cgcnn", "G_cgcnn"),
          "newbase": ("K_newbase", "G_newbase")}


def literature_table():
    """LIT as a table, one row per compound (keys a dict leaves out become blank)."""
    d = pd.DataFrame(LIT).rename(columns={"kappa": "kappa_meas", "lo": "kappa_meas_lo",
                                          "hi": "kappa_meas_hi", "dft": "kappa_lit_dft",
                                          "sg": "spacegroup_number"})
    if d.formula.duplicated().any() or d.mp_id.dropna().duplicated().any():
        raise SystemExit("a compound or mp_id is listed twice in LIT")
    # every row needs a source; a measured row needs a measured value, a DFT row a DFT value
    if d.doi.isna().any() or d.quote.isna().any():
        raise SystemExit("a LIT row has no DOI or no quote")
    if (d.kind == "measured").ne(d.kappa_meas.notna()).any():
        raise SystemExit("'measured' rows (and only those) must have a measured kappa")
    if ((d.kind == "DFT only") & d.kappa_lit_dft.isna()).any():
        raise SystemExit("a 'DFT only' row has no DFT kappa")
    return d


def load_crystals():
    """One row per compound with a structure: the cell the models see, and its geometry."""
    f = json.load(open(STRUCT))
    lit = literature_table()
    rows = []
    for r in lit[lit.mp_id.notna()].itertuples():      # itertuples: one row at a time, as r.formula etc.
        mb = f["matbench"].get(r.mp_id)                  # .get returns None when the key is absent
        if mb:
            st, src, mb_id = Structure.from_dict(mb["structure"]), "matbench", mb["mb_id"]
        else:
            st, src, mb_id = Structure.from_dict(f["pbe"][r.mp_id]), "Materials Project (PBE)", None
        rows.append({"formula": r.formula, "mp_id": r.mp_id, "mb_id": mb_id,
                     "structure_from": src, "structure": st})
    d = pd.DataFrame(rows)
    d["n_atoms"] = [len(s) for s in d.structure]
    d["volume_a3"] = [s.volume for s in d.structure]
    d["density"] = [s.density for s in d.structure]                       # g/cm^3
    d["mass_amu"] = [float(s.composition.weight) for s in d.structure]    # whole cell
    same = [s.composition.reduced_formula == Composition(x).reduced_formula
            for s, x in zip(d.structure, d.formula)]
    if not all(same):
        raise SystemExit(f"structure/formula mismatch: {d.formula[~np.array(same)].tolist()}")
    return d


def agree(a, b, tol):
    """True when every value in a matches b to within tol (relative)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(len(a) > 0 and np.all(np.abs(a - b) <= tol * np.abs(b)))


# =============================================================================
#  stage 1: fetch  (ml_env, internet)
# =============================================================================
def stage_fetch():
    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _s85 = import_module("85_match_phonix_matbench")    # load_matbench, match (StructureMatcher)

    lit = literature_table()
    have = lit[lit.mp_id.notna()]
    ids = have.mp_id.tolist()
    with MPRester() as mpr:                              # reads the key from ~/.pmgrc.yaml
        version = mpr.get_database_version()
        summ = mpr.materials.summary.search(material_ids=ids, fields=[
            "material_id", "energy_above_hull", "theoretical", "symmetry"])
        mp_sg = {str(x.material_id): x.symmetry.number for x in summ}   # MP's own space group
        # MP's PBE ("GGA") relaxed cell: the same kind of cell matbench used (step 97)
        pbe, sg_checked_by = {}, {}
        for r in have.itertuples():
            es = mpr.get_entries(r.mp_id, compatible_only=False,
                                 additional_criteria={"thermo_types": ["GGA_GGA+U"]})
            if len(es) != 1:
                raise SystemExit(f"{r.mp_id}: expected one PBE entry in MP, found {len(es)}")
            st = es[0].structure
            if st.composition.reduced_formula != Composition(r.formula).reduced_formula:
                raise SystemExit(f"{r.mp_id} is {st.composition.reduced_formula}, not {r.formula}")
            # the space group, two ways. (a) MP's record for this mp_id - always available.
            if mp_sg[r.mp_id] != r.spacegroup_number:
                raise SystemExit(f"{r.mp_id} ({r.formula}): MP says space group {mp_sg[r.mp_id]}, "
                                 f"expected {r.spacegroup_number}")
            # (b) our own spglib on the PBE cell. spglib 2.7.0 has a bug that fails on a few
            # cells at every tolerance (CsCu2I3, found 2026-10-07); then (a) has to do.
            # for ... else: the else runs only if the loop never reached "break"
            for tol in [0.1, 0.01, 1e-3]:
                try:
                    sga = SpacegroupAnalyzer(st, symprec=tol)
                    break
                except Exception:
                    continue
            else:
                sga = None
            if sga is not None and sga.get_space_group_number() != r.spacegroup_number:
                raise SystemExit(f"{r.mp_id} ({r.formula}): the PBE cell is space group "
                                 f"{sga.get_space_group_number()}, expected {r.spacegroup_number}")
            sg_checked_by[r.mp_id] = "MP record + spglib" if sga is not None else "MP record only (spglib bug)"
            # two Pnma polymorphs share the space group; the corner-sharing PEROVSKITE has
            # three long axes (~8.4-12.7 A), the 1D chain phase one short axis (~4.7 A)
            if r.perovskite == True:                     # noqa: E712 - the column also holds NaN
                if sga is None:
                    raise SystemExit(f"{r.mp_id}: cannot check the perovskite polymorph without spglib")
                short = min(sga.get_conventional_standard_structure().lattice.abc)
                if short < 7:
                    raise SystemExit(f"{r.mp_id} has a {short:.2f} A axis: not the perovskite polymorph")
            pbe[r.mp_id] = st.as_dict()
        el = mpr.materials.elasticity.search(material_ids=ids)
    print(f"MP database {version}: {len(pbe)} PBE cells, all of the expected formula and space group; "
          f"spglib could not type {[m for m, v in sg_checked_by.items() if 'only' in v]}")

    # ---- is the crystal in matbench? the same match step 85 used -----------------
    mb = _s85.load_matbench()
    cand = pd.DataFrame({"mp_id": list(pbe),
                         "formula": [Structure.from_dict(s).composition.reduced_formula for s in pbe.values()],
                         "structure": [Structure.from_dict(s) for s in pbe.values()]})
    cand = _s85.match(cand, mb)
    cells = {}
    for mp, mb_id, n in zip(cand.mp_id, cand.mb_id, cand.n_mb_hits):
        if mb_id:
            row = int(mb_id.split("-")[1])               # "mb-00042" -> row 42
            cells[mp] = {"mb_id": mb_id, "n_hits": int(n), "structure": mb.structure[row].as_dict()}
    print(f"in matbench: {len(cells)} of {len(cand)}: "
          f"{[(m, v['mb_id'], v['n_hits']) for m, v in cells.items()]}")
    # Cs2SnI6 is in step 97 too: both steps must pick the same matbench crystal
    old = json.load(open(STEP97_MB))
    for mp in set(cells) & set(old):                     # & = ids in both
        if cells[mp]["mb_id"] != old[mp]["mb_id"]:
            raise SystemExit(f"{mp}: matched {cells[mp]['mb_id']}, step 97 had {old[mp]['mb_id']}")
    with open(STRUCT, "w") as fh:
        json.dump({"mp_database_version": version, "pbe": pbe, "matbench": cells}, fh)

    hull = {str(x.material_id): (x.energy_above_hull, x.theoretical) for x in summ}
    rows = []
    for mp in ids:
        e = next((x for x in el if str(x.material_id) == mp), None)   # next(..., None): first match or None
        rows.append({"mp_id": mp, "mp_e_above_hull": hull[mp][0], "mp_theoretical": hull[mp][1],
                     "K_mp_now": e.bulk_modulus.vrh if e and e.bulk_modulus else np.nan,
                     "G_mp_now": e.shear_modulus.vrh if e and e.shear_modulus else np.nan,
                     "mp_elastic_flags": "; ".join(e.warnings or []) if e else "",
                     "spacegroup_checked_by": sg_checked_by[mp],
                     "mp_database_version": version})
    out = pd.DataFrame(rows)
    out.to_csv(MP_ELAST, index=False)
    print(f"MP elasticity for {out.K_mp_now.notna().sum()} of {len(out)}")
    print(f"wrote {STRUCT}\nwrote {MP_ELAST}")


# =============================================================================
#  stage 2: ALIGNN and CGCNN-ens  (infer_env)
# =============================================================================
def stage_torch():
    import torch
    from jarvis.core.atoms import pmg_to_atoms
    from cgcnn_scratch.data import AtomFeaturiser, GaussianDistance, structure_to_graph
    _al = import_module("15_alignn_predict_moduli")     # load_alignn_target, predict_pair
    _pred = import_module("04_predict_moduli")          # load_model, predict_ensemble

    # the screen's models, loaded the same way as steps 87 and 97
    dev = torch.device("cpu")
    k_al, cfg = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_bulk_modulus_kv"), dev)
    g_al, _ = _al.load_alignn_target(os.path.join(ROOT, "results/alignn/alignn_shear_modulus_gv"), dev)
    k_cg = [_pred.load_model(t, RES)[:2] for t in ["K_VRH_full", "K_VRH_s1", "K_VRH_s2"]]
    g_cg = [_pred.load_model(t, RES)[:2] for t in ["G_VRH_full", "G_VRH_s1", "G_VRH_s2"]]
    ari = AtomFeaturiser(os.path.join(ROOT, "cgcnn_scratch", "atom_init.json"))
    gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)

    d = load_crystals()
    rows, graphs = [], []
    for st, mp in zip(d.structure, d.mp_id):
        k, g = _al.predict_pair(k_al, g_al, pmg_to_atoms(st), cfg, dev)
        rows.append({"mp_id": mp, "K_alignn": k, "G_alignn": g})
        graphs.append(structure_to_graph(st, ari, gdf, 12, 8))   # 12 neighbours within 8 A
    out = pd.DataFrame(rows)
    ds = _pred.PredictionSet(graphs, d.mp_id.tolist())
    K, _ = _pred.predict_ensemble(k_cg, ds)             # dicts: mp_id -> GPa
    G, _ = _pred.predict_ensemble(g_cg, ds)
    out["K_cgcnn"], out["G_cgcnn"] = out.mp_id.map(K), out.mp_id.map(G)

    # ---- check 1: Cs2SnI6 (same matbench cell in step 97) gives step 97's numbers back
    both = out.merge(pd.read_csv(STEP97_TORCH), on="mp_id", suffixes=("", "_97"))
    ok = (agree(both.K_cgcnn, both.K_cgcnn_97, 1e-4) and agree(both.G_cgcnn, both.G_cgcnn_97, 1e-4)
          and agree(both.K_alignn, both.K_alignn_97, ALIGNN_TOL)
          and agree(both.G_alignn, both.G_alignn_97, ALIGNN_TOL))
    print(both[["mp_id", "K_alignn", "K_alignn_97", "K_cgcnn", "K_cgcnn_97"]].to_string(index=False))
    if not ok:
        raise SystemExit("CHECK FAILED: step 97's moduli for the shared crystal did not come back")
    print(f"check 1 passed: {len(both)} crystal(s) shared with step 97 reproduce it")

    # ---- check 2: crystals in matbench's TEST split give step 82's CGCNN-ens back
    s82 = pd.read_csv(CHAIN82, usecols=["mb_id", "K_CGCNN-ens", "G_CGCNN-ens"])
    test = out.assign(mb_id=d.mb_id.values).merge(s82, on="mb_id")
    if len(test):
        if not (agree(test.K_cgcnn, test["K_CGCNN-ens"], 1e-4) and agree(test.G_cgcnn, test["G_CGCNN-ens"], 1e-4)):
            raise SystemExit("CHECK FAILED: CGCNN-ens on the matbench test cells != step 82")
        print(f"check 2 passed: {len(test)} test-split crystal(s) match step 82 to 1e-4")
    else:
        print("check 2 skipped: none of these crystals is in matbench's test split")

    # ---- ALIGNN's neighbour-tie noise (step 97): nudge every atom ~1e-5 A, 8 times
    rng = np.random.default_rng(0)                      # fixed seed: the same nudges every run
    tie = []
    for st, mp in zip(d.structure, d.mp_id):
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
        lo = np.minimum(out[f"{t}_alignn_tie_min"], out[f"{t}_alignn"])   # include the unnudged answer
        hi = np.maximum(out[f"{t}_alignn_tie_max"], out[f"{t}_alignn"])
        out[f"{t}_alignn_tie_min"], out[f"{t}_alignn_tie_max"] = lo, hi
        spread = hi / lo
        print(f"ALIGNN {t} tie noise: median x{spread.median():.3f}, worst x{spread.max():.3f} "
              f"({d.formula[spread.idxmax()]})")
    out.to_csv(TORCH_OUT, index=False)
    print(f"wrote {TORCH_OUT}: {len(out)} crystals")


# =============================================================================
#  stage 3: newbase  (ml_env)
# =============================================================================
def stage_newbase():
    _s87 = import_module("87_phonix_poisson_calibration")   # make_featuriser: the 297 descriptors
    _s71 = import_module("71_screen_with_new_baseline")     # predict_baseline: 5 xgboost folds
    feat_one = _s87.make_featuriser()

    d = load_crystals()
    pbe = json.load(open(STRUCT))["pbe"]
    rows = []
    for st, mp in zip(d.structure, d.mp_id):
        r = feat_one(st, mp)
        r["newbase_cell"] = "same as ALIGNN, CGCNN"
        if r["featurize_error"]:
            # matminer's Voronoi search fails on a few exactly-symmetric matbench cells (atoms at
            # coordinate -0.0 after standardising; found on Tl3VSe4). MP's PBE cell of the SAME
            # crystal goes through - used only if its volume agrees to 1e-4
            alt = Structure.from_dict(pbe[mp])
            if abs(alt.volume / st.volume - 1) > 1e-4:
                raise SystemExit(f"{mp}: featurisation failed and the PBE cell is a different volume")
            print(f"{mp}: {r['featurize_error']} -> retried on MP's PBE cell")
            r = feat_one(alt, mp)
            r["newbase_cell"] = "MP PBE (matbench cell failed featurisation)"
        rows.append(r)
    f = pd.DataFrame(rows)
    bad = f.featurize_error.fillna("") != ""
    if bad.any():
        raise SystemExit(f"failed to featurise: {f[bad].id.tolist()}")
    p = _s71.predict_baseline(f.set_index("id"))            # log10(GPa)
    out = pd.DataFrame({"mp_id": f.id.values, "K_newbase": 10 ** p["K"], "G_newbase": 10 ** p["G"],
                        "primitive_fallback": f.primitive_fallback.values, "newbase_cell": f.newbase_cell.values})

    both = out.merge(pd.read_csv(STEP97_NEWBASE), on="mp_id", suffixes=("", "_97"))
    if not (agree(both.K_newbase, both.K_newbase_97, 1e-4) and agree(both.G_newbase, both.G_newbase_97, 1e-4)):
        print(both.to_string(index=False))
        raise SystemExit("CHECK FAILED: step 97's newbase moduli for the shared crystal did not come back")
    print(f"check passed: {len(both)} crystal(s) shared with step 97 reproduce its newbase K, G to 1e-4")
    out.to_csv(NEWBASE_OUT, index=False)
    print(f"wrote {NEWBASE_OUT}: {len(out)} crystals")


# =============================================================================
#  stage 4: score  (ml_env)
# =============================================================================
def stage_score():
    _s71 = import_module("71_screen_with_new_baseline")    # slack_physics, verify_physics
    import matplotlib
    matplotlib.use("Agg")                                  # draw to a file, not a window
    import matplotlib.pyplot as plt

    _s71.verify_physics(_s71.rank_four_models())   # the kappa formula is the screen's, or stop

    lit = literature_table()
    d = load_crystals().drop(columns="structure")
    d = lit.merge(d, on=["formula", "mp_id"], how="left")      # left: keep the 2 rows with no structure

    # ---- matbench split and DFT labels ------------------------------------------
    test_ids = set(pd.read_csv(CHAIN82).mb_id)
    has_cell = d.structure_from.notna()
    d["matbench_split"] = np.where(~has_cell, None,
                          np.where(d.mb_id.isna(), "not in mb",
                          np.where(d.mb_id.isin(test_ids), "test", "train/val")))
    lab = pd.read_csv(LABELS).set_index("mb_id")
    d["K_matbench"], d["G_matbench"] = d.mb_id.map(lab.K_VRH), d.mb_id.map(lab.G_VRH)
    d = d.merge(pd.read_csv(MP_ELAST), on="mp_id", how="left")
    # MP attaches warnings to tensors it does not trust (CsCu2I3: "negative eigenvalue(s) ...
    # mechanically unstable", "unphysical modulus"). Such a K, G is not a DFT modulus to compare with.
    flagged = d.mp_elastic_flags.fillna("") != ""
    print(f"MP elasticity dropped as flagged by MP itself: {d.formula[flagged].tolist()}")
    d.loc[flagged, ["K_mp_now", "G_mp_now"]] = np.nan

    # ---- predictions --------------------------------------------------------------
    d = d.merge(pd.read_csv(TORCH_OUT), on="mp_id", how="left")
    d = d.merge(pd.read_csv(NEWBASE_OUT), on="mp_id", how="left")
    for t, task in [("K", "kvrh"), ("G", "gvrh")]:
        oof = pd.read_csv(OOF.format(task))
        oof["mb_id"] = [f"mb-{i:05d}" for i in oof.row_index]     # f"{i:05d}" pads to 5 digits
        oof = oof.set_index("mb_id")
        d[f"{t}_newbase_oof"] = 10 ** d.mb_id.map(oof.y_pred_oof)
        has = d.mb_id.notna()
        if not agree(10 ** d.mb_id.map(oof.y_true)[has], d.loc[has, f"{t}_matbench"], 1e-4):
            raise SystemExit(f"newbase out-of-fold truth disagrees with matbench {t} - ids mismatched")
    if d.loc[has_cell, ["K_alignn", "K_cgcnn", "K_newbase"]].isna().any().any():
        raise SystemExit("a crystal with a structure is missing a model prediction")
    d["networks_held_out"] = d.matbench_split.isin(["not in mb", "test"])
    d["newbase_held_out"] = d.matbench_split == "not in mb"

    # ---- kappa: the chain on every set of moduli ----------------------------------
    sources = {**MODELS, "newbase_oof": ("K_newbase_oof", "G_newbase_oof"),
               "DFT MP now": ("K_mp_now", "G_mp_now"), "DFT matbench": ("K_matbench", "G_matbench")}
    for name, (kc, gc) in sources.items():
        phys = _s71.slack_physics(d[kc].values, d[gc].values, d.volume_a3.values,
                                  d.density.values, d.mass_amu.values, d.n_atoms.values)
        d[f"kappa_{name}"] = phys["kappa_cal"]               # W/m/K at 300 K; NaN where moduli are missing
        d[f"gamma_{name}"] = phys["gruneisen"]
    d["alignn_reliable"] = (d.K_alignn > MIN_MODULUS) & (d.G_alignn > MIN_MODULUS)
    d.loc[has_cell & ~d.alignn_reliable, "kappa_ALIGNN"] = np.nan   # step 57's rule
    d["kappa_max3"] = d[[f"kappa_{m}" for m in MODELS]].max(axis=1)

    # PhoNIX phonon-DFT kappa_L, where the same mp_id was computed (median of repeats, as step 85)
    ph = pd.read_csv(PHONIX, usecols=["mp_id", "klat"]).groupby("mp_id").klat.median()
    d["phonix_klat"] = d.mp_id.map(ph)

    # matbench's frozen DFT label against MP's current DFT for the same crystal. The networks
    # learnt matbench's number, so a wrong label there becomes a wrong prediction here
    for t in "KG":
        d[f"{t}_matbench_over_mp_now"] = d[f"{t}_matbench"] / d[f"{t}_mp_now"]
    both = d.dropna(subset=["K_matbench_over_mp_now"])
    print("\nmatbench label / MP's current DFT, same crystal (matbench K, G are whole GPa):")
    print(both[["formula", "K_matbench", "K_mp_now", "K_matbench_over_mp_now", "G_matbench", "G_mp_now",
                "G_matbench_over_mp_now"]].round(2).to_string(index=False))

    # the literature value each prediction is compared with: measured, else the DFT one
    d["kappa_reference"] = d.kappa_meas.fillna(d.kappa_lit_dft)
    for name in list(MODELS) + ["max3", "newbase_oof", "DFT MP now", "DFT matbench"]:
        d[f"ratio_{name}"] = d[f"kappa_{name}"] / d.kappa_reference
    d["ratio_phonix"] = d.phonix_klat / d.kappa_reference

    first = ["formula", "kind", "kappa_meas", "kappa_meas_lo", "kappa_meas_hi", "kappa_lit_dft",
             "T", "quantity", "form", "ref", "doi", "quote", "dft_doi", "caveat",
             "mp_id", "spacegroup_number", "structure_from", "mb_id", "matbench_split",
             "networks_held_out", "newbase_held_out", "kappa_max3", "kappa_ALIGNN", "kappa_CGCNN-ens",
             "kappa_newbase", "kappa_newbase_oof", "kappa_DFT MP now", "kappa_DFT matbench", "phonix_klat"]
    d = d[first + [c for c in d.columns if c not in first and c != "perovskite"]]
    d.to_csv(OUT_CSV, index=False)

    # ---- summary ------------------------------------------------------------------
    def line(sub, col, label, ref="kappa_meas"):
        """One printed line: how many are called low, median ratio, how many within x2."""
        s = sub.dropna(subset=[col, ref])
        if not len(s):
            print(f"  {label:34s} n= 0")
            return
        r = s[col] / s[ref]
        print(f"  {label:34s} n={len(s):2d}  called <= 1: {(s[col] <= LOW).sum():2d}/{len(s):2d}   "
              f"median pred/lit x{np.median(r):.2f}   within x2: {((r >= 0.5) & (r <= 2)).sum():2d}/{len(s)}")

    m = d[(d.kind == "measured") & has_cell]
    print(f"\nMEASURED low-kappa compounds with a structure: {len(m)}  "
          f"(all measured <= {m.kappa_meas.max():.2f} W/m/K)")
    print(f"  matbench split: {m.matbench_split.value_counts().to_dict()}")
    for name in list(MODELS) + ["max3"]:
        line(m, f"kappa_{name}", f"chain on {name} K, G")
    line(m[m.networks_held_out], "kappa_max3", "chain on max3, networks held out")
    line(m, "kappa_newbase_oof", "chain on newbase_oof K, G")
    line(m, "kappa_DFT MP now", "chain on MP's DFT K, G")
    line(m, "kappa_DFT matbench", "chain on matbench's DFT K, G")
    line(m, "phonix_klat", "PhoNIX phonon DFT")
    line(m, "kappa_lit_dft", "the paper's own DFT")

    print("\nper compound (W/m/K; ratio = max3 / measured):")
    show = ["formula", "kappa_meas", "quantity", "matbench_split", "kappa_ALIGNN", "kappa_CGCNN-ens",
            "kappa_newbase", "kappa_max3", "ratio_max3", "kappa_DFT MP now", "phonix_klat", "kappa_lit_dft"]
    print(m.sort_values("kappa_meas")[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    dft = d[d.kind == "DFT only"]
    print("\nDFT-ONLY compounds (flagged: no measurement found), compared with the paper's DFT:")
    for name in ["max3", "DFT MP now"]:
        line(dft, f"kappa_{name}", f"chain on {name}", ref="kappa_lit_dft")
    print(dft[["formula", "kappa_lit_dft", "kappa_max3", "ratio_max3", "kappa_DFT MP now", "phonix_klat"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    nope = d[~has_cell]
    print(f"\nno structure, not predicted: {nope.formula.tolist()}")

    # ---- figure -----------------------------------------------------------------
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.25, 1]})
    p = pd.concat([m.sort_values("kappa_meas"), dft]).reset_index(drop=True)
    y = np.arange(len(p))
    # the three models: a grey bar from the lowest to the highest, each model a small mark
    lo3 = p[[f"kappa_{x}" for x in MODELS]].min(axis=1)
    ax.hlines(y, lo3, p.kappa_max3, color="0.75", lw=3, zorder=1)
    for name, c, mk in [("ALIGNN", "tab:blue", "o"), ("CGCNN-ens", "tab:orange", "s"), ("newbase", "tab:green", "^")]:
        ax.scatter(p[f"kappa_{name}"], y, c=c, marker=mk, s=28, zorder=3, label=f"chain on {name} K, G")
    ax.scatter(p["kappa_DFT MP now"], y, marker="s", facecolors="none", edgecolors="k", s=60, lw=1.3,
               zorder=4, label="chain on MP's DFT K, G")
    ax.scatter(p.phonix_klat, y, marker="D", c="tab:purple", s=30, zorder=4, label="PhoNIX phonon DFT")
    ax.scatter(p.kappa_lit_dft, y, marker="v", c="tab:cyan", edgecolors="k", s=55, zorder=4,
               label="paper's DFT (flagged)")
    rng_ = p.dropna(subset=["kappa_meas_lo"])
    ax.hlines(rng_.index, rng_.kappa_meas_lo, rng_.kappa_meas_hi, color="goldenrod", lw=2, zorder=4)
    ax.scatter(p.kappa_meas, y, marker="*", c="gold", edgecolors="k", s=190, zorder=5, label="measured")
    ax.axvline(LOW, color="k", ls="--", lw=0.8)
    if len(dft):
        ax.axhline(len(m) - 0.5, color="0.4", lw=0.8)
        # get_yaxis_transform(): x in fractions of the panel width, y in data units (row numbers)
        ax.text(0.01, len(m) - 0.35, "above this line: DFT only, no measurement (triangle = the paper's DFT)",
                fontsize=7, color="0.3", transform=ax.get_yaxis_transform())
    tag = {"train/val": " [in training]", "test": " [matbench test]", "not in mb": ""}
    names = [f"{f}{tag.get(s, '')}{' (total k)' if q == 'total' else ''}"
             for f, s, q in zip(p.formula, p.matbench_split, p.quantity)]
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("kappa_L at ~300 K (W/m/K)")
    ax.set_title("A. Literature low-kappa crystals: measured vs our chain\n"
                 "grey bar = range over our three models; its right end is the screen's max3")
    ax.legend(fontsize=7, loc="lower right")

    for col, mk, c, lab in [("kappa_max3", "o", "tab:red", "our screen (max3)"),
                            ("kappa_DFT MP now", "s", "k", "chain on MP's DFT K, G"),
                            ("phonix_klat", "D", "tab:purple", "PhoNIX phonon DFT")]:
        s = m.dropna(subset=[col])
        face = "none" if col == "kappa_DFT MP now" else c
        bx.scatter(s.kappa_meas, s[col], marker=mk, facecolors=face, edgecolors=c, s=45, label=lab)
    for r in m.itertuples():
        bx.annotate(r.formula, (r.kappa_meas, r.kappa_max3), fontsize=6.5, xytext=(4, 2),
                    textcoords="offset points", color="tab:red")
    lims = np.array([0.08, 10.0])                          # up to 10: Tl3VSe4's max3 is 7.4
    bx.plot(lims, lims, "k-", lw=0.8)
    bx.fill_between(lims, lims / 2, lims * 2, color="0.92", zorder=0, label="within x2")
    bx.axhline(LOW, color="k", ls="--", lw=0.8)
    bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlim(lims); bx.set_ylim(lims)
    bx.set_xlabel("measured kappa (W/m/K)"); bx.set_ylabel("predicted kappa_L (W/m/K)")
    bx.set_title("B. Predicted vs measured (15 compounds)\nabove the dashed line = the screen would call it NOT low")
    bx.legend(fontsize=7, loc="lower right")               # the upper left holds Tl3VSe4
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "torch", "newbase", "score"])
    stage = ap.parse_args().stage
    {"fetch": stage_fetch, "torch": stage_torch, "newbase": stage_newbase, "score": stage_score}[stage]()
