#!/usr/bin/env python3
"""Standalone PINK pipeline: CIF files -> CGCNN (bulk/shear modulus) -> lattice thermal conductivity.

Reimplements the AI4Kappa Streamlit app as a batch CLI, so predictions can run
headless over a directory of CIFs. Physics mirrors
external/AI4Kappa/streamlit_scripts/calculate_K.py (Liu et al., J. Mater. Inf. 2025, 5, 12).

Usage:
    python pink_predict.py <cif_dir> [-o out.csv] [-n LIMIT]
"""
import argparse
import csv
import glob
import math
import os
import shutil
import sys
import tempfile

# torch must be imported before numpy/pandas/pymatgen: on macOS conda envs, MKL
# loads its own OpenMP runtime first and the duplicate libiomp5 segfaults torch.
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.constants import h, k

AI4KAPPA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "AI4Kappa")
sys.path.insert(0, AI4KAPPA)

from cgcnn.data import CIFData, collate_pool  # noqa: E402
from cgcnn.model import CrystalGraphConvNet  # noqa: E402

MODELS = {
    "Bulk modulus (GPa)": "Bulk modulus (GPa)-pre-trained.pth.tar",
    "Shear modulus (GPa)": "Shear modulus (GPa)-pre-trained.pth.tar",
}


class Normalizer:
    """Denormalizes model output using the mean/std stored in the checkpoint."""

    def load_state_dict(self, state):
        self.mean, self.std = state["mean"], state["std"]

    def denorm(self, tensor):
        return tensor * self.std + self.mean


def stage_inputs(cif_dir, work_dir, limit=None):
    """Copy CIFs plus atom_init.json into work_dir and write the id_prop.csv CIFData expects.

    This fork of CIFData skips a header row and uses cif_id as the full filename,
    so id_prop.csv needs a header and the '.cif' suffix retained.
    """
    cifs = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not cifs:
        raise SystemExit(f"No CIF files found in {cif_dir}")
    if limit:
        cifs = cifs[:limit]

    shutil.copy(os.path.join(AI4KAPPA, "root_dir", "atom_init.json"), work_dir)
    for path in cifs:
        shutil.copy(path, work_dir)

    with open(os.path.join(work_dir, "id_prop.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cif_id", "target"])
        writer.writerows([[os.path.basename(p), 0] for p in cifs])
    return len(cifs)


def predict(dataset, loader, model_file):
    """Run one pre-trained CGCNN model over the dataset, returning (cif_ids, values)."""
    ckpt = torch.load(os.path.join(AI4KAPPA, "model", model_file), map_location="cpu",
                      weights_only=False)
    margs = ckpt["args"]
    structures, _, _ = dataset[0]
    model = CrystalGraphConvNet(
        structures[0].shape[-1], structures[1].shape[-1],
        atom_fea_len=margs["atom_fea_len"], n_conv=margs["n_conv"],
        h_fea_len=margs["h_fea_len"], n_h=margs["n_h"], classification=False,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    normalizer = Normalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    values, cif_ids = [], []
    with torch.no_grad():
        for inputs, _, batch_ids in loader:
            out = model(inputs[0], inputs[1], inputs[2], inputs[3])
            values.append(normalizer.denorm(out).view(-1))
            cif_ids += list(batch_ids)
    # models are trained on log10 of the modulus
    return cif_ids, 10 ** torch.cat(values).numpy()


def add_physics(df):
    """Sound velocities, Debye temperature, Gruneisen parameter, and kappa_L."""
    v_long = ((df["Bulk modulus (GPa)"] + 4 * df["Shear modulus (GPa)"] / 3) / df["Density (g cm-3)"]) ** 0.5 * 1000
    v_trans = (df["Shear modulus (GPa)"] / df["Density (g cm-3)"]) ** 0.5 * 1000
    v_sound = ((1 / v_long ** 3 + 2 / v_trans ** 3) / 3) ** (-1 / 3)

    df["Sound velocity longitudinal (m s-1)"] = v_long
    df["Sound velocity transverse (m s-1)"] = v_trans
    df["Speed of sound (m s-1)"] = v_sound
    df["Acoustic Debye Temperature (K)"] = (
        h / k * np.power(3 / (4 * math.pi * df["Volume (A3)"]), 1 / 3) * v_sound * 1e10
    )

    ratio = v_long / v_trans
    df["Poisson ratio"] = (ratio ** 2 - 2) / (2 * ratio ** 2 - 2)
    df["Gruneisen parameter"] = 3 * (1 + df["Poisson ratio"]) / (2 * (2 - 3 * df["Poisson ratio"]))

    gamma = df["Gruneisen parameter"]
    coeff = 2.43e-8 / (1 - 0.514 / gamma + 0.228 / gamma ** 2)
    df["Kappa_Slack (W m-1 K-1)"] = (
        coeff * df["Atomic mass (amu)"] * df["Volume (A3)"] ** (1 / 3)
        * df["Acoustic Debye Temperature (K)"] ** 3
        / (gamma ** 2 * 300 * df["Number of Atoms"]) * 100
    )
    # Equation (2) of the PINK paper, three-phonon scattering (delta = 1)
    df["Kappa_cal (W m-1 K-1)"] = (
        df["Shear modulus (GPa)"] * 1e9 * df["Speed of sound (m s-1)"]
        * (df["Volume (A3)"] * 1e-30) ** (1 / 3)
        / (df["Number of Atoms"] * 300) * np.exp(-gamma)
    )
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cif_dir", help="directory containing .cif files")
    ap.add_argument("-o", "--output", default="pink_results.csv")
    ap.add_argument("-n", "--limit", type=int, help="only process the first N CIFs")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="pink_") as work:
        count = stage_inputs(args.cif_dir, work, args.limit)
        print(f"Staged {count} CIFs")

        dataset = CIFData(work)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pool)

        moduli = {}
        for label, model_file in MODELS.items():
            cif_ids, values = predict(dataset, loader, model_file)
            moduli[label] = dict(zip(cif_ids, values))
            print(f"Predicted {label}")

        rows = []
        for cif_id in moduli["Bulk modulus (GPa)"]:
            struct = Structure.from_file(os.path.join(work, cif_id)).get_primitive_structure()
            rows.append({
                "Material": os.path.splitext(cif_id)[0],
                "Bulk modulus (GPa)": moduli["Bulk modulus (GPa)"][cif_id],
                "Shear modulus (GPa)": moduli["Shear modulus (GPa)"][cif_id],
                "Number of Atoms": len(struct),
                "Volume (A3)": struct.volume,
                "Density (g cm-3)": struct.density,
                "Atomic mass (amu)": float(struct.composition.weight),
            })

    df = add_physics(pd.DataFrame(rows)).sort_values("Kappa_cal (W m-1 K-1)")
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print("\nLowest predicted kappa_L:")
    print(df.head(10)[["Material", "Bulk modulus (GPa)", "Shear modulus (GPa)",
                       "Gruneisen parameter", "Kappa_Slack (W m-1 K-1)",
                       "Kappa_cal (W m-1 K-1)"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
