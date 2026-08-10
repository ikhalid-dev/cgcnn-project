#!/usr/bin/env python3
"""
STEP 7 - Stage 2 of PINK: turn our predicted moduli into kappa_L.
===================================================================

    python scripts/07_predict_kappa.py

WHERE THIS FITS
----------------
Stage 1 (scripts/01b-06) trained a CGCNN from scratch and produced
`results/pink_moduli_predictions.csv` - bulk and shear modulus for all 1,213
PINK crystals. That is only half of PINK. The other half - unchanged from the
paper, and NOT machine-learned - is a closed-form physics formula (a
simplified Slack model) that turns (K, G, volume, mass, atom count) into a
lattice thermal conductivity kappa_L.

`pink_predict.py` already implements that physics (function `add_physics`),
but it drives it with the PAPER's own pre-trained CGCNN. This script is the
missing piece: run the SAME physics on OUR OWN moduli, so the whole pipeline
- from CIF to kappa_L - runs end to end on our from-scratch model.

WHY THE PHYSICS IS RE-DERIVED HERE RATHER THAN RE-IMPORTED
------------------------------------------------------------
`add_physics()` in pink_predict.py operates on a pandas DataFrame of SCALAR
columns - one K, one G per crystal. That is fine for a point estimate, but it
cannot express "K and G are each a distribution", which is exactly what our
ensemble gives us and the paper's single model cannot. So `slack_physics()`
below reimplements the identical formulas in plain numpy, shape-agnostic via
broadcasting: called with scalars it reproduces add_physics() exactly (the
--self-test flag checks this directly against pink_predict.py), and called
with (n_crystals, n_samples) arrays it becomes a vectorised Monte Carlo.

UNCERTAINTY PROPAGATION - THE THING THE ORIGINAL PIPELINE CANNOT DO
--------------------------------------------------------------------
Our ensemble already gives every crystal a K and G spread (std across 3
independently-seeded models, in log10 space) - a free per-crystal confidence
signal that a single pre-trained model has no equivalent of. This script
propagates it: for each crystal, draw N samples of log10(K) and log10(G) from
Normal(mean, spread), push every sample through the SAME physics, and report
the resulting spread in kappa_L. A crystal where the ensemble disagrees a lot
about K or G should - and does - come out with a wide kappa_L interval; one
where all three models agree gets a tight one. K and G are sampled
independently because they come from separately-seeded ensembles with no
shared randomness to correlate.

OUTPUT
------
    results/pink_kappa_predictions.csv
        material_id, formula, provenance, structural quantities,
        Poisson ratio, Gruneisen parameter,
        Kappa_Slack (W/m/K), Kappa_cal (W/m/K)   <- point estimates (Eq. 2 of the paper)
        Kappa_cal_p05 / p50 / p95                 <- Monte Carlo interval from ensemble spread
    Sorted ascending by Kappa_cal - PINK screens for ultralow kappa_L, so the
    most interesting candidates are the first rows.
"""

import argparse
import math
import os
import sys
import warnings

# torch first, even though this script never trains anything: --self-test
# imports pink_predict.py, which imports torch, and in this conda env MKL
# loads its own OpenMP runtime first - the duplicate libiomp5 aborts the
# process if torch is imported after numpy/pandas. Same rule as every other
# script here; see cgcnn_scratch/data.py for the full explanation.
import torch  # noqa: F401

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.constants import h, k

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def slack_physics(k_gpa, g_gpa, volume_a3, density, mass_amu, n_atoms):
    """The Slack-model physics, shape-agnostic via numpy broadcasting.

    Identical formulas to pink_predict.py's add_physics(), rewritten so that
    K and G can be arrays (a Monte Carlo sample per crystal) rather than only
    scalars. Broadcasting rules do the rest: pass matching-shape arrays for a
    point estimate, or (n_crystals, 1) structural quantities against
    (n_crystals, n_samples) moduli for a vectorised Monte Carlo.

    Returns a dict of arrays, same shape as k_gpa: v_long, v_trans, v_sound,
    debye_temp, poisson, gruneisen, kappa_slack, kappa_cal.
    """
    v_long = ((k_gpa + 4 * g_gpa / 3) / density) ** 0.5 * 1000
    v_trans = (g_gpa / density) ** 0.5 * 1000
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

    return {"v_long": v_long, "v_trans": v_trans, "v_sound": v_sound,
           "debye_temp": debye_temp, "poisson": poisson,
           "gruneisen": gruneisen, "kappa_slack": kappa_slack,
           "kappa_cal": kappa_cal}


def self_test():
    """Prove slack_physics() matches pink_predict.py's add_physics() exactly.

    The two are independent implementations of the same formulas - one built
    for scalar/pandas use, one for numpy broadcasting - so agreement is
    checked, not assumed. Run with --self-test.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from pink_predict import add_physics

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Bulk modulus (GPa)": rng.uniform(5, 400, n),
        "Shear modulus (GPa)": rng.uniform(2, 200, n),
        "Number of Atoms": rng.randint(2, 40, n),
        "Volume (A3)": rng.uniform(20, 800, n),
        "Density (g cm-3)": rng.uniform(1, 12, n),
        "Atomic mass (amu)": rng.uniform(10, 300, n),
    })
    reference = add_physics(df.copy())

    ours = slack_physics(
        df["Bulk modulus (GPa)"].values, df["Shear modulus (GPa)"].values,
        df["Volume (A3)"].values, df["Density (g cm-3)"].values,
        df["Atomic mass (amu)"].values, df["Number of Atoms"].values)

    checks = [
        ("Poisson ratio", reference["Poisson ratio"].values, ours["poisson"]),
        ("Gruneisen parameter", reference["Gruneisen parameter"].values, ours["gruneisen"]),
        ("Kappa_Slack (W m-1 K-1)", reference["Kappa_Slack (W m-1 K-1)"].values, ours["kappa_slack"]),
        ("Kappa_cal (W m-1 K-1)", reference["Kappa_cal (W m-1 K-1)"].values, ours["kappa_cal"]),
    ]
    for name, ref_vals, our_vals in checks:
        if not np.allclose(ref_vals, our_vals, rtol=1e-9):
            max_diff = np.max(np.abs(ref_vals - our_vals))
            sys.exit(f"SELF-TEST FAILED on {name}: max abs diff {max_diff:.3e}")
        print(f"  OK  {name} matches pink_predict.add_physics() "
             f"(n={n}, rtol=1e-9)")
    print("Self-test passed: slack_physics() is verified identical to "
         "pink_predict.py's add_physics().")


def structure_quantities(cif_dir, material_ids):
    """Number of atoms, volume, density, atomic mass - from the CIF itself.

    Uses get_primitive_structure(), matching pink_predict.py exactly, so a
    crystal's structural quantities here are directly comparable to its
    quantities in the paper's own reference run - neither is inflated by a
    supercell the CIF author happened to write down.
    """
    rows, failed = [], []
    for material_id in material_ids:
        path = os.path.join(cif_dir, f"{material_id}.cif")
        try:
            struct = Structure.from_file(path).get_primitive_structure()
        except Exception as exc:
            failed.append((material_id, str(exc)[:70]))
            continue
        rows.append({
            "material_id": material_id,
            "Number of Atoms": len(struct),
            "Volume (A3)": struct.volume,
            "Density (g cm-3)": struct.density,
            "Atomic mass (amu)": float(struct.composition.weight),
        })
    if failed:
        print(f"  {len(failed)} CIFs failed to parse, e.g. {failed[:3]}")
    return pd.DataFrame(rows)


def monte_carlo_kappa(df, n_samples, seed):
    """Propagate ensemble (K, G) uncertainty through to a kappa_L interval.

    log10(K) and log10(G) are each drawn independently from
    Normal(prediction, ensemble spread) - independent because the K-ensemble
    and G-ensemble are separately seeded 3-model runs with nothing shared
    between them. Every draw goes through the identical slack_physics() used
    for the point estimate, so the interval is consistent with the reported
    central value by construction rather than by a separate approximation.
    """
    rng = np.random.RandomState(seed)
    n = len(df)

    z_k = rng.standard_normal((n, n_samples))
    z_g = rng.standard_normal((n, n_samples))
    log_k = np.log10(df["K_VRH_pred"].values)[:, None] + \
        df["K_VRH_spread_log10"].values[:, None] * z_k
    log_g = np.log10(df["G_VRH_pred"].values)[:, None] + \
        df["G_VRH_spread_log10"].values[:, None] * z_g
    k_samples = 10 ** log_k
    g_samples = 10 ** log_g

    # Structural quantities do not vary in the Monte Carlo - only the
    # ML-predicted moduli are uncertain - so broadcast them as a column.
    col = lambda name: df[name].values[:, None]
    result = slack_physics(k_samples, g_samples, col("Volume (A3)"),
                           col("Density (g cm-3)"), col("Atomic mass (amu)"),
                           col("Number of Atoms"))

    kappa = result["kappa_cal"]
    return {
        "Kappa_cal_p05": np.nanpercentile(kappa, 5, axis=1),
        "Kappa_cal_p50": np.nanpercentile(kappa, 50, axis=1),
        "Kappa_cal_p95": np.nanpercentile(kappa, 95, axis=1),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions",
                        default=os.path.join(PROJECT_ROOT, "results", "cgcnn",
                                             "pink_moduli_predictions.csv"))
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "cgcnn",
                                                      "pink_kappa_predictions.csv"))
    parser.add_argument("--mc-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true",
                        help="verify slack_physics() against pink_predict.py "
                             "and exit, without processing any CIFs")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    print("=== Stage 2: our moduli -> kappa_L (Slack model) ===\n")
    moduli = pd.read_csv(args.predictions)
    print(f"Loaded {len(moduli)} crystals from {args.predictions}")

    print("Computing structural quantities from CIFs "
         "(primitive cell, matching pink_predict.py)...")
    structure = structure_quantities(args.cif_dir, moduli.material_id)
    df = moduli.merge(structure, on="material_id", how="inner")
    dropped = len(moduli) - len(df)
    if dropped:
        print(f"  {dropped} crystals dropped: CIF failed to parse")

    # --- Point estimate: physics run once, on the ensemble MEAN modulus ----
    point = slack_physics(
        df["K_VRH_pred"].values, df["G_VRH_pred"].values,
        df["Volume (A3)"].values, df["Density (g cm-3)"].values,
        df["Atomic mass (amu)"].values, df["Number of Atoms"].values)
    df["Poisson ratio"] = point["poisson"]
    df["Gruneisen parameter"] = point["gruneisen"]
    df["Kappa_Slack (W m-1 K-1)"] = point["kappa_slack"]
    df["Kappa_cal (W m-1 K-1)"] = point["kappa_cal"]

    # --- Uncertainty: physics run n_samples times per crystal --------------
    print(f"Propagating ensemble uncertainty ({args.mc_samples} Monte Carlo "
         f"draws per crystal)...")
    mc = monte_carlo_kappa(df, args.mc_samples, args.seed)
    for name, values in mc.items():
        df[name] = values

    df = df.sort_values("Kappa_cal (W m-1 K-1)").reset_index(drop=True)

    keep = ["material_id", "formula", "provenance", "Number of Atoms",
           "Volume (A3)", "Density (g cm-3)", "Atomic mass (amu)",
           "K_VRH_pred", "G_VRH_pred", "K_VRH_spread_log10", "G_VRH_spread_log10",
           "Poisson ratio", "Gruneisen parameter",
           "Kappa_Slack (W m-1 K-1)", "Kappa_cal (W m-1 K-1)",
           "Kappa_cal_p05", "Kappa_cal_p50", "Kappa_cal_p95"]
    df[keep].round(5).to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} rows to {args.out}")
    print("\nLowest predicted kappa_L (our from-scratch model):")
    print(df.head(10)[["material_id", "formula", "K_VRH_pred", "G_VRH_pred",
                       "Gruneisen parameter", "Kappa_cal (W m-1 K-1)",
                       "Kappa_cal_p05", "Kappa_cal_p95"]]
          .round(3).to_string(index=False))

    width = (df["Kappa_cal_p95"] / df["Kappa_cal_p05"])
    print(f"\nMonte Carlo interval width (p95/p05 ratio): "
         f"median {width.median():.2f}x, "
         f"widest {width.max():.1f}x ({df.loc[width.idxmax(), 'material_id']})")


if __name__ == "__main__":
    main()
