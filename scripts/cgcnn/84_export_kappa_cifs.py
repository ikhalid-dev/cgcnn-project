#!/usr/bin/env python
"""
Step 84 - CIF files for the lattice-thermal-conductivity DFT run, small cells only.
================================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/84_export_kappa_cifs.py
    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/84_export_kappa_cifs.py --max-atoms 15

WHY
---
The DFT run computes lattice thermal conductivity directly (third-order force
constants + Boltzmann transport), and its cost climbs steeply with the number
of atoms in the cell. The plan is to run cells of at most 10-20 atoms. This
step takes step 83's queue, keeps the crystals under that cap, and writes one
CIF per crystal plus an index sheet in queue order.

Files are named so the order survives being copied anywhere:

    g1_03__Cs4NbCrI12__d7e7d897af.cif
    |   |   |          |
    |   |   formula    GNoME material_id
    |   position inside its group, after the atom-count cut
    group (1 real candidate, 2 low/models disagree, 3 near cutoff, C control)

WHAT IS CHECKED, NOT TRUSTED
----------------------------
* The GNoME archive is keyed by COMPOSITION, not material_id (see step 62).
  That is only safe if no two GNoME entries share a composition - checked.
* Every CIF is parsed with pymatgen before it is written, and its atom count
  and reduced formula must match the queue's, or the script stops.
* Checked once by hand (2026-10-05), in mlip_env: all 57 cells of the <=20
  export are already PRIMITIVE (so the atom count is the real DFT cost) and
  all 57 space groups match GNoME's. Do that check in mlip_env, not ml_env -
  ml_env's numpy 2.3 stack fails to find the symmetry of 5 centred cells
  (Ama2, Cmcm, C2/c), with the same spglib 2.7.0. It is the environment, not
  the files.

WHAT THE INDEX DOES NOT CARRY
-----------------------------
Step 83's "expected DFT range" columns are deliberately left OUT. They were
calibrated against DFT ELASTIC MODULI (step 82), and this DFT run computes
kappa_L, which also tests the phonon gamma and the Slack formula itself. A
range for kappa_L has not been measured yet; printing the moduli one here would
invite exactly the wrong comparison.

OUTPUTS
    dft/kappa_L_cifs/*.cif
    dft/kappa_L_cifs/index.csv
"""
import argparse
import os
import shutil                  # deleting the previous export so no stale file survives
import zipfile                 # reading the GNoME archive without unpacking all 450 MB
import warnings

import pandas as pd

warnings.filterwarnings("ignore")                  # pymatgen is loud about CIF rounding
from pymatgen.core import Composition, Structure   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
SUMMARY = os.path.join(ROOT, "gnome_data", "stable_materials_summary.csv")
ARCHIVE = os.path.join(ROOT, "gnome_data", "by_composition.zip")
OUT = os.path.join(ROOT, "dft", "kappa_L_cifs")

# "1: real candidate" -> "g1", "C: negative control" -> "gC"
def group_code(label):
    return "g" + label.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-atoms", type=int, default=20, help="largest cell to keep")
    args = ap.parse_args()

    q = pd.read_csv(os.path.join(RES, "83_trusted_dft_queue.csv"), dtype={"material_id": str})
    summ = pd.read_csv(SUMMARY, usecols=["MaterialId", "Composition", "NSites", "Space Group"],
                       dtype={"MaterialId": str})

    # the archive key must be unambiguous
    if summ.Composition.duplicated().any():
        raise SystemExit("two GNoME entries share a composition - the archive key is ambiguous")

    # how="left" keeps every queue row even if its id were missing from the summary
    q = q.merge(summ, left_on="material_id", right_on="MaterialId", how="left")
    if q.Composition.isna().any():
        raise SystemExit(f"ids not in the GNoME summary: {q[q.Composition.isna()].material_id.tolist()}")

    # NSites = atoms in GNoME's cell, which is the cell the CIF holds. For the
    # controls the queue has no atom count, so NSites is the only source.
    q["n_atoms"] = q.NSites.astype(int)
    print(f"queue: {len(q)} rows (221 candidates + 4 controls)")
    print("atoms per cell across the queue:")
    bins = pd.cut(q.n_atoms, [0, 10, 15, 20, 10_000], labels=["<=10", "11-15", "16-20", ">20"])
    print(pd.crosstab(q.group_label, bins, margins=True).to_string())

    keep = q[q.n_atoms <= args.max_atoms].copy()
    print(f"\nkept with <= {args.max_atoms} atoms: {len(keep)}")

    # order: group, then the screen's own ranking inside the group
    keep["group"] = keep.group_label.map(group_code)
    keep = keep.sort_values(["group", "kappa_max3"]).reset_index(drop=True)
    keep["pos_in_group"] = keep.groupby("group").cumcount() + 1

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)                          # start clean - no leftover files from a wider cut
    os.makedirs(OUT)

    # zipfile.ZipFile opens the archive like a folder; .read(name) pulls ONE
    # file out of it as raw bytes, and .decode("utf-8") turns bytes into text.
    archive = zipfile.ZipFile(ARCHIVE)
    files = []                                      # an empty list; .append adds one name per crystal
    # .itertuples() walks the table one row at a time; r.formula is that row's
    # "formula" cell. (Columns with spaces, like "Space Group", are not
    # reachable this way, which is why the loop never touches them.)
    for r in keep.itertuples():
        text = archive.read(f"by_composition/{r.Composition}.CIF").decode("utf-8")
        st = Structure.from_str(text, fmt="cif")    # parse BEFORE writing
        # len(st) = number of atoms pymatgen actually found in the file
        if len(st) != r.n_atoms:
            raise SystemExit(f"{r.formula}: CIF has {len(st)} atoms, GNoME says {r.n_atoms}")
        # reduced_formula writes any formula the same way (Cs4I4 -> CsI), so the
        # two spellings can be compared fairly
        if st.composition.reduced_formula != Composition(r.formula).reduced_formula:
            raise SystemExit(f"{r.formula}: CIF holds {st.composition.reduced_formula}")
        # keep letters and digits, turn brackets etc. into "_" so the name is a safe filename
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.formula)
        # {r.pos_in_group:02d} prints 3 as "03", so files sort correctly by name
        name = f"{r.group}_{r.pos_in_group:02d}__{safe}__{r.material_id}.cif"
        # "with open(...) as fh" opens the file and closes it automatically afterwards
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(text)                          # the original GNoME text, unmodified
        files.append(name)
    keep["file"] = files                            # one name per row, same order as the loop

    index = keep[["file", "group_label", "pos_in_group", "formula", "material_id", "n_atoms",
                  "Space Group", "gamma_mlip", "ALIGNN", "CGCNN-ens", "newbase", "kappa_max3",
                  "spread3", "kappa_cahill", "below_floor", "old_tier_81"]].rename(columns={
        "Space Group": "space_group",
        "gamma_mlip": "gamma_phonon_mlip",
        "kappa_max3": "kappa_pred_slack_300K",
        "kappa_cahill": "kappa_amorphous_floor"})
    index.to_csv(os.path.join(OUT, "index.csv"), index=False)

    print(f"\nwrote {len(index)} CIFs + index.csv to {os.path.relpath(OUT, ROOT)}/")
    print(index[["file", "n_atoms", "space_group", "kappa_pred_slack_300K", "spread3",
                 "kappa_amorphous_floor"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
