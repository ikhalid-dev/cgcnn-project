#!/usr/bin/env python3
"""
STEP 61 - Choose the DFT validation set: the experiment that closes the gap.
================================================================================

    python scripts/cgcnn/61_dft_validation_set.py

WHY THIS EXISTS
-----------------
Every result in this project is model-vs-model or model-vs-database. No
candidate has ever been checked against a calculation this project ran itself.
That is the single missing piece: a reviewer's first question about any
discovery claim here is "did you verify any of them?", and the answer is
currently no.

This picks the materials to check. It is a designed experiment, not a top-N
list, because "the ten lowest predicted kappa" is the WRONG set to compute.
Step 16 already established the winner's-curse problem: argmin over 33k noisy
predictions selects for noise as much as for merit. And a confirmation-only set
cannot resolve the disagreement this project actually has.

THE THREE ARMS
----------------
  A. CONSENSUS LOW  - all four models call it low-kappa. Tests whether the
     pipeline is right when everything agrees. Ranked by the MOST PESSIMISTIC
     of the four models rather than by the lowest prediction, so a candidate
     qualifies only if even the least enthusiastic model likes it. This is the
     step-16 lesson applied without needing a calibrated interval.

  B. ADJUDICATION   - the three matbench/composition models call it low and
     round 9 calls it high. This is the arm that pays for itself: steps 46, 57
     and 58 established that round 9 disagrees with every other model in this
     project and that the disagreement is in the MODULI, but GNoME has no
     ground truth so none of them could say who is right. A DFT elastic tensor
     settles it. Ranked by the size of the disagreement.

  C. NEGATIVE CONTROL - all four models call it high-kappa. Without this arm a
     confirmed arm A means little: a pipeline that calls everything soft would
     pass arm A and fail here. Cheap, and it is the difference between "our
     predictions were right" and "our predictions were right about the thing we
     selected for".

WHAT TO ACTUALLY COMPUTE, AND WHY IT IS CHEAPER THAN IT LOOKS
--------------------------------------------------------------
Do NOT compute kappa directly. Compute the ELASTIC TENSOR.

Full anharmonic lattice thermal conductivity needs third-order force constants
on supercells - expensive, and it is not the disputed quantity. Every diagnostic
in this project points at the moduli instead:

    step 46   the entire baseline-vs-round-9 screen gap is in the moduli
    step 58   moduli alone reproduce the full ALIGNN-vs-round-9 rank gap
              (rho 0.725 vs the 0.723 headline); gamma alone leaves it at 0.977

So K and G are what the models disagree about, and an elastic tensor from finite
strain is a handful of static calculations per material on cells this small.
Quasi-harmonic Debye kappa (what AFLOW-AGL itself reports) then follows almost
for free from that tensor, giving a second, independent comparison point at
near-zero extra cost.

SELECTION CONSTRAINTS
-----------------------
  * <= `max_atoms` atoms per cell - DFT cost scales steeply, and step 19 already
    built this project's small-cell pool for exactly this reason.
  * ALIGNN's reliability flag must be true. Step 28 caught ALIGNN emitting its
    ~1e-3 GPa regression floor, which reads as a spectacular discovery and is
    a degenerate output. Those are excluded, not ranked first.
  * Space group and crystal system are carried through so the person running
    the calculations can see the symmetry, which drives the real cost.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs (all pre-existing; no model is run) -------------------------
    "alignn_screen": "results/alignn/57_gnome_screen_alignn.csv",
    "cgcnn_screen":  "results/cgcnn/39_gnome_screen_all_gamma.csv",
    "spacegroups":   "results/cgcnn/gnome_screen_spacegroups.csv",

    # ---- outputs ------------------------------------------------------------
    # The full record, with every column the selection used.
    "out_csv": "results/cgcnn/61_dft_validation_set.csv",
    # A trimmed, run-ready sheet for whoever actually sets up the calculations:
    # renamed columns, units in the names, ordered by what to run first, and
    # carrying the GNoME Composition key that the CIF archive is indexed by.
    "run_csv": "dft/dft_materials.csv",
    # supplies material_id -> Composition, the archive's own filename key
    "gnome_summary": "gnome_data/stable_materials_summary.csv",

    # ---- selection ----------------------------------------------------------
    "max_atoms": 10,           # DFT cost; matches step 19's small-cell pool
    "low_threshold": 1.0,      # W/m/K - the call this project screens on
    "high_threshold": 3.0,     # arm C: every model must exceed this
    "adjudication_r9_min": 2.0,   # arm B: round 9 must be clearly above the call
    "n_arm_a": 8,
    "n_arm_b": 6,
    "n_arm_c": 4,
}
# =============================================================================

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The four independent predictions this project can put on one crystal. Two are
# matbench-trained with a derived gamma, one is AFLOW-trained with a trained
# gamma head, one is composition-only. They do not all count as independent
# evidence in equal measure - see the note printed at the end.
MODELS = {
    "ALIGNN":    "Kappa_alignn",
    "CGCNN-ens": "Kappa_cal_derived_matbench",
    "round9":    "Kappa_r9_gamma",
    "tree":      "Kappa_baseline",
}
KEEP = ["material_id", "formula", "Number of Atoms", "space_group", "crystal_system",
        "Volume (A3)", "Density (g cm-3)", "K_alignn", "G_alignn",
        "K_r9_pred", "G_r9_pred"] + list(MODELS)


def p(*a):
    print(*a, flush=True)


def main():
    cfg = CONFIG
    a = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["alignn_screen"]), dtype={"material_id": str})
    s = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["cgcnn_screen"]), dtype={"material_id": str})
    sg = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["spacegroups"]), dtype={"material_id": str},
                     usecols=["material_id", "space_group", "space_group_number", "crystal_system"])

    d = a.merge(s, on="material_id", suffixes=("", "_dup")).merge(sg, on="material_id", how="left")
    # ALIGNN's regression floor is a degenerate output, not a discovery (step 28)
    d = d[d["alignn_prediction_reliable"]].copy()
    for name, col in MODELS.items():
        d[name] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(MODELS))

    lo, hi = cfg["low_threshold"], cfg["high_threshold"]
    d["kappa_max"] = d[list(MODELS)].max(axis=1)      # the most pessimistic model
    d["kappa_min"] = d[list(MODELS)].min(axis=1)
    d["n_models_low"] = (d[list(MODELS)] <= lo).sum(axis=1)
    # the three non-round-9 models, which agree with each other across the board
    d["others_low"] = ((d["ALIGNN"] <= lo) & (d["CGCNN-ens"] <= lo) & (d["tree"] <= lo))

    small = d[d["Number of Atoms"] <= cfg["max_atoms"]].copy()

    p("=" * 78)
    p("  STEP 61 - the DFT validation set")
    p("=" * 78)
    p()
    p(f"  reliable candidates: {len(d)}   with <= {cfg['max_atoms']} atoms: {len(small)}")
    p(f"    arm A pool (all 4 call low):            {int((small.n_models_low == 4).sum())}")
    p(f"    arm B pool (others low, round 9 > {cfg['adjudication_r9_min']}):   "
      f"{int((small.others_low & (small.round9 > cfg['adjudication_r9_min'])).sum())}")
    p(f"    arm C pool (all 4 above {hi}):            "
      f"{int((small.kappa_min > hi).sum())}")

    # ---- arm A: ranked by the most pessimistic model, not the lowest --------
    A = small[small.n_models_low == 4].nsmallest(cfg["n_arm_a"], "kappa_max").copy()
    A["arm"] = "A: consensus low"
    A["why"] = "all four models agree; ranked by the most pessimistic of them"

    # ---- arm B: ranked by how badly round 9 disagrees -----------------------
    Bpool = small[small.others_low & (small.round9 > cfg["adjudication_r9_min"])].copy()
    Bpool["disagreement"] = Bpool["round9"] / Bpool["ALIGNN"]
    B = Bpool.nlargest(cfg["n_arm_b"], "disagreement").copy()
    B["arm"] = "B: adjudication"
    B["why"] = "matbench+tree say low, round 9 says high - DFT settles which"

    # ---- arm C: the negative control ---------------------------------------
    C = small[small.kappa_min > hi].nlargest(cfg["n_arm_c"], "kappa_min").copy()
    C["arm"] = "C: negative control"
    C["why"] = "all four models call it stiff; guards against a pipeline that calls everything soft"

    out = pd.concat([A, B, C], ignore_index=True)
    cols = ["arm", "why"] + KEEP + ["kappa_max", "kappa_min", "n_models_low"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    out.to_csv(os.path.join(PROJECT_ROOT, cfg["out_csv"]), index=False)

    # ---- the run-ready sheet ------------------------------------------------
    # Arm B first: it is the arm that resolves this project's open question, so
    # if the calculations stop early they should stop having answered something.
    order = {"B: adjudication": 0, "A: consensus low": 1, "C: negative control": 2}
    run = out.copy()
    run["_o"] = run["arm"].map(order)
    run = run.sort_values(["_o", "Number of Atoms"]).drop(columns="_o").reset_index(drop=True)

    # The CIF archive is keyed by Composition, not by material_id, so carry it:
    # without this every user of the sheet has to redo the same 176 MB join.
    summ_path = os.path.join(PROJECT_ROOT, cfg["gnome_summary"])
    if os.path.exists(summ_path):
        summ = pd.read_csv(summ_path, usecols=["MaterialId", "Composition"],
                           dtype={"MaterialId": str})
        run = run.merge(summ, left_on="material_id", right_on="MaterialId", how="left")
        run = run.drop(columns=["MaterialId"])
    else:
        run["Composition"] = np.nan
        p("  NOTE: gnome_data/stable_materials_summary.csv absent - the CIF lookup")
        p("        key (Composition) is blank in the run sheet.")

    run.insert(0, "run_order", np.arange(1, len(run) + 1))
    run = run.rename(columns={
        "arm": "arm", "why": "what_this_tests", "formula": "formula",
        "material_id": "gnome_material_id", "Composition": "cif_key_in_archive",
        "Number of Atoms": "n_atoms", "space_group": "space_group",
        "crystal_system": "crystal_system", "Volume (A3)": "volume_A3",
        "Density (g cm-3)": "density_g_cm3",
        "K_alignn": "K_pred_alignn_GPa", "G_alignn": "G_pred_alignn_GPa",
        "K_r9_pred": "K_pred_round9_GPa", "G_r9_pred": "G_pred_round9_GPa",
        "ALIGNN": "kappa_alignn", "CGCNN-ens": "kappa_cgcnn_ens",
        "round9": "kappa_round9", "tree": "kappa_tree",
    })
    run_cols = ["run_order", "arm", "formula", "gnome_material_id", "cif_key_in_archive",
                "n_atoms", "space_group", "crystal_system", "volume_A3", "density_g_cm3",
                "K_pred_alignn_GPa", "G_pred_alignn_GPa",
                "K_pred_round9_GPa", "G_pred_round9_GPa",
                "kappa_alignn", "kappa_cgcnn_ens", "kappa_round9", "kappa_tree",
                "what_this_tests"]
    run = run[[c for c in run_cols if c in run.columns]]
    run_path = os.path.join(PROJECT_ROOT, cfg["run_csv"])
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    run.to_csv(run_path, index=False)

    for arm, grp in out.groupby("arm", sort=True):
        p()
        p("  " + "-" * 74)
        p(f"  {arm.upper()}   ({len(grp)} materials)")
        p("  " + "-" * 74)
        p(f"  {'formula':14s} {'at':>3} {'space grp':11s} {'ALIGNN':>8} {'CGCNN':>8} "
          f"{'round9':>8} {'tree':>8}")
        for _, r in grp.iterrows():
            p(f"  {str(r['formula']):14s} {int(r['Number of Atoms']):>3} "
              f"{str(r['space_group']):11s} {r['ALIGNN']:>8.3f} {r['CGCNN-ens']:>8.3f} "
              f"{r['round9']:>8.3f} {r['tree']:>8.3f}")

    p()
    p("  " + "-" * 74)
    p("  WHAT TO COMPUTE")
    p("  " + "-" * 74)
    p("    The ELASTIC TENSOR, not kappa. Steps 46 and 58 both localise every")
    p("    model disagreement in this project to the moduli, so K and G are the")
    p("    disputed quantity; a finite-strain elastic tensor on cells this small")
    p("    is a handful of static calculations each. Quasi-harmonic Debye kappa")
    p("    then follows from that tensor at near-zero extra cost, which is what")
    p("    AFLOW-AGL itself reports - giving a second comparison point free.")
    p()
    p("  TWO THINGS THIS SET CANNOT DO")
    p("    * The four models are not four independent votes. ALIGNN and the CGCNN")
    p("      ensemble share matbench training data and both derive gamma the same")
    p("      way, so arm A's 'all four agree' is closer to two-and-a-half votes")
    p("      than four. Arm B is the arm that carries real information.")
    p("    * DFT is not experiment. A confirmed arm A means the pipeline agrees")
    p("      with a different calculation, not with reality - and step 59 measured")
    p("      that two DFT workflows already disagree by 0.054 log10 in G.")
    p()
    p(f"  wrote: {cfg['out_csv']}  ({len(out)} materials, full record)")
    p(f"         {cfg['run_csv']}  (run-ready sheet, arm B first)")


if __name__ == "__main__":
    main()
