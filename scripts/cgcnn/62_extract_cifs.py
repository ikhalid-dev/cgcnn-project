#!/usr/bin/env python3
"""
STEP 62 - Pull real CIF files out of the GNoME archive for any candidate list.
================================================================================

    # the DFT validation set
    python scripts/cgcnn/62_extract_cifs.py --csv dft/dft_materials.csv --out dft/cifs

    # every 5-element low-kappa oxide
    python scripts/cgcnn/62_extract_cifs.py \\
        --csv results/cgcnn/gnome_oxide_low_kappa_candidates.csv \\
        --elements 5 --out cifs/oxides_5el

    # the 20 lowest-kappa of some list
    python scripts/cgcnn/62_extract_cifs.py --csv <any screen csv> --top 20 --out cifs/look

WHY THIS EXISTS
-----------------
Every candidate table in this project carries a `material_id` and a formula, and
neither of those is something you can open and look at. The structures live only
inside `gnome_data/by_composition.zip` (gitignored, ~450 MB), and they are NOT
indexed by material_id - the archive is keyed by GNoME's own Composition string,
so MoBr5Cl is stored as `Br5Cl1Mo1.CIF`. Recovering that key needs a join
against the 176 MB summary file, which is why nobody does it by hand twice.

TWO ID FACTS THAT CAUSE REAL CONFUSION
----------------------------------------
1. GNoME ids are NOT Materials Project ids. They are 10-character hex-ish
   strings (`a0e05c85e7`, `000006a8c4`); MP ids look like `mp-1234`. Not one of
   the 554,054 GNoME entries carries an mp- id, so looking a GNoME id up on the
   Materials Project will always fail. These are different databases: GNoME
   structures are DeepMind's own predictions, most of which have never been in
   MP at all. To find a GNoME material on MP you have to search by COMPOSITION
   and then check whether MP has that phase - often it will not.

2. Ids must be read as strings. 34,046 GNoME ids begin with a leading zero and
   4,632 are all digits, so a read without `dtype=str` silently turns
   `000006a8c4`-style ids into numbers and loses the padding. Every read in this
   script pins the dtype. The GNoME summary itself also ships 4,543 malformed
   rows (`nan`, truncated 7-digit ids) - the source of this project's
   `id_unverifiable` flag - so a few ids legitimately resolve to nothing, and
   this script reports them rather than failing.

WHAT IT WRITES
----------------
One `.cif` per material, named `<formula>__<material_id>.cif` so the file is
identifiable from the filename alone, plus a `manifest.csv` mapping every file
back to its row and recording anything that failed. Each structure is parsed
with pymatgen before being written, so a file that lands is a file that opens.
"""

# =============================================================================
#  CONFIG - defaults; every one is overridable from the command line
# =============================================================================
CONFIG = {
    "gnome_summary": "gnome_data/stable_materials_summary.csv",
    "gnome_zip":     "gnome_data/by_composition.zip",
    "zip_prefix":    "by_composition",
    "out_dir":       "cifs/extracted",
    "kappa_cols":    ["Kappa_alignn", "kappa_alignn", "Kappa_cal (W m-1 K-1)",
                      "Kappa_cal_derived_matbench", "Kappa_r9_gamma"],
}
# =============================================================================

import argparse
import os
import sys
import warnings
import zipfile

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)   # pymatgen noble-gas EN warnings
from pymatgen.core import Composition, Structure   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p(*a):
    print(*a, flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", required=True,
                    help="any candidate table carrying a material_id column")
    ap.add_argument("--out", default=None, help="output directory for the .cif files")
    ap.add_argument("--elements", type=int, default=None,
                    help="keep only rows with exactly this many DISTINCT elements")
    ap.add_argument("--contains", default=None,
                    help="keep only rows containing this element (e.g. O). Uses a real "
                         "composition parse, so O does not match Os.")
    ap.add_argument("--max-kappa", type=float, default=None,
                    help="keep only rows at or below this kappa (first kappa column found)")
    ap.add_argument("--max-atoms", type=int, default=None, help="keep only cells this small")
    ap.add_argument("--top", type=int, default=None,
                    help="after filtering, keep the N lowest-kappa rows")
    ap.add_argument("--limit", type=int, default=0, help="hard cap on files written (0 = no cap)")
    return ap.parse_args()


def n_elements(formula):
    try:
        return len(Composition(str(formula)).elements)
    except Exception:
        return -1


def has_element(formula, sym):
    """Real composition parse - a substring test would match Os when asked for O."""
    try:
        return any(e.symbol == sym for e in Composition(str(formula)).elements)
    except Exception:
        return False


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    out_dir = os.path.join(PROJECT_ROOT, args.out or cfg["out_dir"])

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(PROJECT_ROOT, args.csv)
    # dtype pinned: see the id note in this module's docstring
    d = pd.read_csv(csv_path, dtype={"material_id": str, "gnome_material_id": str})
    if "material_id" not in d.columns and "gnome_material_id" in d.columns:
        d = d.rename(columns={"gnome_material_id": "material_id"})
    if "material_id" not in d.columns:
        sys.exit(f"ERROR: {args.csv} has no material_id column")

    p("=" * 78)
    p("  STEP 62 - extracting CIFs")
    p("=" * 78)
    p(f"  source: {args.csv}  ({len(d)} rows)")

    # ---- filters, each reported so the final count is never a surprise ------
    if args.elements is not None:
        before = len(d)
        d = d[d["formula"].map(n_elements) == args.elements]
        p(f"  filter: exactly {args.elements} distinct elements   {before} -> {len(d)}")
    if args.contains:
        before = len(d)
        d = d[d["formula"].map(lambda f: has_element(f, args.contains))]
        p(f"  filter: contains {args.contains}   {before} -> {len(d)}")
    if args.max_atoms is not None:
        col = next((c for c in ["Number of Atoms", "n_atoms", "n_sites"] if c in d.columns), None)
        if col:
            before = len(d)
            d = d[pd.to_numeric(d[col], errors="coerce") <= args.max_atoms]
            p(f"  filter: <= {args.max_atoms} atoms   {before} -> {len(d)}")

    kcol = next((c for c in cfg["kappa_cols"] if c in d.columns), None)
    if kcol and args.max_kappa is not None:
        before = len(d)
        d = d[pd.to_numeric(d[kcol], errors="coerce") <= args.max_kappa]
        p(f"  filter: {kcol} <= {args.max_kappa}   {before} -> {len(d)}")
    if kcol and args.top:
        d = d.assign(_k=pd.to_numeric(d[kcol], errors="coerce")).nsmallest(args.top, "_k")
        d = d.drop(columns="_k")
        p(f"  kept the {args.top} lowest by {kcol}")
    if args.limit:
        d = d.head(args.limit)

    if d.empty:
        sys.exit("ERROR: no rows left after filtering - nothing to extract")

    # ---- the archive is keyed by Composition, not material_id --------------
    summ = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["gnome_summary"]),
                       usecols=["MaterialId", "Composition"], dtype={"MaterialId": str})
    d = d.merge(summ, left_on="material_id", right_on="MaterialId", how="left")

    unresolved = d["Composition"].isna().sum()
    if unresolved:
        p(f"  WARNING: {unresolved} of {len(d)} ids do not appear in the GNoME summary and")
        p(f"           cannot be looked up. See the id note in this script's docstring -")
        p(f"           older candidate files predate an id-hygiene fix and carry a few.")

    os.makedirs(out_dir, exist_ok=True)
    archive = zipfile.ZipFile(os.path.join(PROJECT_ROOT, cfg["gnome_zip"]))

    rows, n_ok = [], 0
    for _, r in d.iterrows():
        mid, formula, comp = str(r["material_id"]), str(r.get("formula", "?")), r.get("Composition")
        rec = {"material_id": mid, "formula": formula, "cif_key": comp,
               "file": "", "status": ""}
        if not isinstance(comp, str):
            rec["status"] = "id not in GNoME summary"
            rows.append(rec)
            continue
        try:
            text = archive.read(f"{cfg['zip_prefix']}/{comp}.CIF").decode("utf-8")
            # parse before writing: a file that lands is a file that opens
            st = Structure.from_str(text, fmt="cif")
            safe = "".join(ch if ch.isalnum() else "_" for ch in formula)
            fname = f"{safe}__{mid}.cif"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(text)
            # The space group is a convenience column, never a reason to fail an
            # extraction: spglib cannot determine symmetry for some of these
            # cells (it is loud about it on stderr), and an earlier version of
            # this script let that exception mark an already-written file as
            # failed - including MoBr5Cl, the top DFT candidate. Prefer the
            # value the input table already carries; only compute as a
            # fallback, and swallow the failure.
            sg = r.get("space_group")
            if not isinstance(sg, str) or not sg:
                try:
                    sg = st.get_space_group_info()[0]
                except Exception:
                    sg = ""
            rec.update(file=fname, status="ok", n_sites=len(st), spacegroup=sg)
            n_ok += 1
        except KeyError:
            rec["status"] = f"{comp}.CIF not in archive"
        except Exception as exc:
            rec["status"] = f"{exc.__class__.__name__}: {str(exc)[:60]}"
        rows.append(rec)

    man = pd.DataFrame(rows)
    man_path = os.path.join(out_dir, "manifest.csv")
    man.to_csv(man_path, index=False)

    p()
    p(f"  wrote {n_ok} CIF files to {os.path.relpath(out_dir, PROJECT_ROOT)}/")
    bad = man[man["status"] != "ok"]
    if len(bad):
        p(f"  {len(bad)} could not be extracted:")
        for _, b in bad.head(10).iterrows():
            p(f"      {b['formula']:18s} {b['material_id']:12s} {b['status']}")
    p(f"  manifest: {os.path.relpath(man_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
