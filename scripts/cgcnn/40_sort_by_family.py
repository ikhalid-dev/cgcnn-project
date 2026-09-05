#!/usr/bin/env python3
"""
STEP 40 - Sort every result into per-structural-family directories.
================================================================================

    python scripts/cgcnn/40_sort_by_family.py

WHY THIS SCRIPT EXISTS
------------------------
Two things this project has produced are both organised by SOURCE (which
script wrote them) rather than by CHEMISTRY: the 33,118-row GNoME screen
(scripts/cgcnn/39_screen_gnome_gamma.py's output) and the two training sets
(matbench's 10,987 crystals, AFLOW's 5,563). Someone doing chemistry-specific
work - "how good is the model on perovskites", "what spinels did GNoME find"
- has to re-filter one of those big tables by hand every time. This script
does that filtering once, for every named family it can recognise from the
formula alone, and writes the result into results/families/<family>/.

WHY THE FAMILY LIST IS BIGGER THAN "perovskite, spinel, pyrochlore"
-----------------------------------------------------------------------
15_filter_oxides.py's classify_oxide() already tags three named oxide
families (ABO3 perovskite, AB2O4 spinel, A2B2O7 pyrochlore) but lumps EVERY
single-cation oxide into one undifferentiated "binary oxide (A-O)" bucket
regardless of its O:cation ratio, and everything else into "ternary oxide,
other ratio" or "complex oxide (N elements)". Both buckets hide real,
well-known mineral families that differ only by that ratio - splitting them
out is the actual point of this script, not a cosmetic relabelling:

    ratio (cations : O)     name                  example
    1 : 1                   rock-salt             MgO, NiO
    1 : 2                   fluorite/rutile-type  ZrO2, TiO2, CeO2
    2 : 3                   corundum/bixbyite     Al2O3, Fe2O3, In2O3
    1 : 1 : 2               delafossite           CuAlO2, AgFeO2
    1 : 1 : 3               perovskite            SrTiO3, BaTiO3
    1 : 1 : 4               scheelite/zircon-type CaWO4, YVO4
    1 : 2 : 4               spinel                MgAl2O4
    1 : 2 : 6               trirutile             ZnSb2O6, MgTa2O6
    2 : 2 : 7               pyrochlore            Gd2Zr2O7

Fluorite and rutile share a stoichiometry but not a structure (8- vs
6-coordinate cation) and cannot be told apart from the formula alone - so,
like classify_oxide()'s own perovskite/ilmenite caveat, this is a
stoichiometry-only first pass, not a verified space-group match.

WHY HALOGENS, C, N, S, P, Se AND B ARE EXCLUDED FROM "CATIONS"
-------------------------------------------------------------------
A formula containing O AND one of these can look like a clean ratio while
actually being a carbonate/nitrate/sulfate/selenite/borate/oxyhalide - a
molecular anion group substituting for a cation, not a second cation. The
earlier ABO3 count in this project's own history caught exactly this
(SrCO3, LiNO3, PdSeO3 all matched the 1:1:3 ratio but are not perovskites);
this script applies the same exclusion to every ratio-based family below,
not just ABO3.

WHAT ELSE IS SORTED, BESIDES THE GNOME SCREEN
--------------------------------------------------
The two training sets (data_full/labels.csv for matbench, data_full/
gamma_labels.csv for AFLOW) are sorted too, WITH their train/val/test split
label attached (reconstructed with the exact split_indices() function and
split_seed=42 every training script in this project uses) - so "how good is
the model on family X" can be answered directly from
results/families/<family>/training_set.csv without re-deriving the split
from scratch each time, the way earlier ad-hoc checks in this project did.

EXTRA_CANDIDATE_TABLES below are the other per-material result files this
project has produced that are NOT already fully represented by the GNoME
screen's own family column - the ALIGNN-agreement DFT shortlists and the
recalibrated rankings each carry columns (ALIGNN's own predictions,
agreement flags, the post-calibration confidence tier) that do not exist
anywhere else, so they are classified and re-filed the same way, one file
per family per source, rather than pointed back at the base screen.

WHAT IS DELIBERATELY *NOT* RE-SORTED, AND WHY
--------------------------------------------------
Several result files in results/cgcnn/ are strictly superseded by something
already sorted above, and re-sorting a stale copy would add confusion
instead of removing it:
  - gnome_screen_all.csv / gnome_screen_candidates.csv / gnome_oxide_
    candidates.csv / gnome_oxide_low_kappa_candidates.csv - the pre-rename
    2026-08-09 GNoME snapshot. 13_gnome_screen_all.csv / 15_gnome_oxide_
    *.csv are the same pipeline re-run on a later snapshot with the id-
    verification fix; 39_gnome_screen_all_gamma.csv is the current,
    complete table and already carries the pre-rename pipeline's kappa as
    a joined-in column for comparison.
  - 13_gnome_screen_*.csv / 15_gnome_oxide_*.csv themselves - superseded by
    39_gnome_screen_all_gamma.csv for the same reason once round 9 existed.
  - gnome_small_cell_dft_ranking.csv / gnome_small_cell_oxides_for_dft.csv /
    gnome_screen_small_cells.csv / gnome_screen_small_cells_low_kappa.csv -
    pre-recalibration / pre-ALIGNN-filter versions; the _recalibrated and
    _alignn variants below are what should actually be used.
  - gnome_*_by_stoichiometry.csv / *_spacegroups.csv / *_elements.csv /
    *_feature_importance.csv / *_best_by_*.csv / *_kappa_vs_atoms*.csv -
    these are already GROUPED summary tables (one row per spacegroup,
    element, or stoichiometry bucket), not per-material candidate lists -
    "sort by family" does not apply to a table that already IS a sort by
    something else. A family-stratified version of one of these would be a
    new analysis, not a re-filing of an existing one.

GOING FORWARD
-----------------
This is the one script that maintains results/families/ - when a future
screening or shortlisting script produces a new per-material candidate
table worth its own family view, add it to EXTRA_CANDIDATE_TABLES below
rather than leaving it to sort by hand.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs ---------------------------------------------------------------
    "gnome_screen_csv": "results/cgcnn/39_gnome_screen_all_gamma.csv",   # the most complete GNoME table (old + new kappa)
    "matbench_labels_csv": "data_full/labels.csv",
    "matbench_graphs_pt": "data_full/graphs.pt",
    "aflow_labels_csv": "data_full/gamma_labels.csv",
    "aflow_graphs_pt": "data_full/gamma_graphs.pt",

    # ---- split reconstruction (must match every training script exactly) -----
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,

    # ---- output -----------------------------------------------------------
    "out_dir": "results/families",

    # ---- other per-material candidate tables worth their own family view -----
    # (output filename stem, source csv). Each becomes
    # results/families/<family>/<stem>.csv, written only for families that
    # actually have rows in that table - see the module docstring for which
    # existing result files are deliberately left OUT of this list because
    # they are superseded by one of these, or are already a grouped summary
    # rather than a per-material table.
    "extra_candidate_tables": [
        ("dft_agreement_lt1", "results/cgcnn/gnome_dft_candidates_agreement_lt1.csv"),
        ("dft_agreement_lt5", "results/cgcnn/gnome_dft_candidates_agreement_lt5.csv"),
        ("dft_ranking_recalibrated", "results/cgcnn/gnome_small_cell_dft_ranking_recalibrated.csv"),
        ("dft_oxides_recalibrated", "results/cgcnn/gnome_small_cell_oxides_for_dft_recalibrated.csv"),
        ("alignn_small_cells", "results/cgcnn/gnome_small_cells_alignn.csv"),
        ("alignn_small_cells_low_kappa", "results/cgcnn/gnome_small_cells_alignn_low_kappa.csv"),
        ("alignn_small_cells_oxides", "results/cgcnn/gnome_small_cells_alignn_oxides.csv"),
    ],
}
# =============================================================================

import os                 # path joining/creation
import sys                 # sys.path mutation
import warnings              # silences pymatgen's formula-parsing warnings

import torch              # torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard)
import numpy as np          # the split-reconstruction PRNG
import pandas as pd          # every table this script reads and writes
from pymatgen.core import Composition   # parses a formula into its element:amount dict

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/cgcnn/ -> scripts/ -> root
sys.path.insert(0, PROJECT_ROOT)

# Elements that form a molecular anion (carbonate, nitrate, sulfate,
# selenite, tellurite, borate) or a second halide anion alongside oxygen,
# rather than a second true cation - excluded from every ratio-based family
# check below. Te added after JARVIS's "SrTeO3/CaTeO3 perovskites" turned
# out to be tellurites on inspection: Te(IV)'s stereochemically active lone
# pair gives 3-coordinate pyramidal TeO3 groups (confirmed directly via
# CrystalNN, coordination number 3) in low-symmetry space groups (P2_1/c,
# P1), not the 6-coordinate BO6 octahedra and cubic/orthorhombic/
# rhombohedral space groups a real ABO3 perovskite has.
ANION_FORMERS = {"C", "N", "S", "P", "Se", "Te", "B", "F", "Cl", "Br", "I"}


def classify_family(formula):
    """One family label per formula, mutually exclusive, stoichiometry-only.

    Returns a human-readable family name, or "non-oxide" / "other oxide"
    when nothing below matches. See the module docstring's table for the
    exact ratio -> name mapping and why each exclusion exists.
    """
    try:
        comp = Composition(formula)
    except Exception:
        return "unparseable"
    elements = {str(el) for el in comp.elements}
    if "O" not in elements:
        return "non-oxide"

    amounts = comp.reduced_composition.get_el_amt_dict()   # smallest-integer element:amount dict
    o_amt = amounts.pop("O", 0)                              # oxygen's own reduced amount
    cations = amounts                                        # everything left, before the anion-former check

    if set(cations) & ANION_FORMERS:
        # A real second cation might still coexist with e.g. a halide
        # (oxyhalide), but stoichiometry alone cannot tell that apart from
        # a carbonate/nitrate-style molecular anion - safer to bucket these
        # separately than to silently mislabel them as a clean oxide family.
        return "other oxide (mixed-anion / molecular-ion)"

    n_cat = len(cations)
    ratio = tuple(sorted(cations.values()))   # e.g. (1, 1), (1, 2), (2, 2) - order-independent pattern key

    if n_cat == 0:
        return "elemental oxygen (no cation)"   # e.g. solid O2/O4 - a real DFT phase, not a formula-parsing failure

    if n_cat == 1:
        if ratio == (1,) and o_amt == 1:
            return "rock-salt-like (AO)"
        if ratio == (1,) and o_amt == 2:
            return "fluorite/rutile-like (AO2)"
        if ratio == (2,) and o_amt == 3:
            return "corundum/bixbyite-like (A2O3)"
        # Every OTHER binary ratio collapses into one bucket rather than one
        # directory per exact ratio - most of these are one-off, uncommon
        # stoichiometries (5:12, 15:16, ...) with a handful of crystals each,
        # not named mineral families worth their own directory.
        return "other binary oxide (uncommon ratio)"

    if n_cat == 2:
        if ratio == (1, 1) and o_amt == 2:
            return "delafossite-like (ABO2)"
        if ratio == (1, 1) and o_amt == 3:
            return "perovskite-like (ABO3)"
        if ratio == (1, 1) and o_amt == 4:
            return "scheelite/zircon-like (ABO4)"
        if ratio == (1, 2) and o_amt == 4:
            return "spinel-like (AB2O4)"
        if ratio == (1, 2) and o_amt == 6:
            return "trirutile-like (AB2O6)"
        if ratio == (2, 2) and o_amt == 7:
            return "pyrochlore-like (A2B2O7)"
        return "ternary oxide (other ratio)"

    return f"complex oxide ({n_cat + 1} elements)"   # +1 to count oxygen back into the element total


def slugify(name):
    """Family name -> filesystem-safe directory name."""
    keep = "".join(c if c.isalnum() else "-" for c in name.lower())
    while "--" in keep:                # collapse repeated hyphens left by stripped punctuation
        keep = keep.replace("--", "-")
    return keep.strip("-")


def split_indices(n_total, train_ratio, val_ratio, seed):
    """Identical to every training script's own split_indices() - reproduced
    here rather than imported so this script has no import-order dependency
    on any specific trainer, since it reads from BOTH matbench and AFLOW
    trainers' data."""
    rng = np.random.RandomState(seed)              # seeded PRNG, deterministic given `seed`
    idx = rng.permutation(n_total)                  # shuffled array of indices 0..n_total-1
    n_tr = int(round(train_ratio * n_total))         # crystals assigned to train
    n_va = int(round(val_ratio * n_total))            # crystals assigned to val
    return idx[:n_tr].tolist(), idx[n_tr:n_tr + n_va].tolist(), idx[n_tr + n_va:].tolist()


def training_set_table(labels_csv, graphs_pt, id_col, source_name):
    """One row per crystal: id, formula, split, family, and whatever physical
    columns that labels file happens to carry (K_VRH/G_VRH always; AFLOW's
    file additionally has gamma/kappa_agl, matbench's does not - both are
    kept as-is rather than forced into one shared schema)."""
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, labels_csv))
    blob = torch.load(os.path.join(PROJECT_ROOT, graphs_pt), weights_only=False)   # {"ids": [...], "graphs": [...]}
    labelled = set(labels[id_col])                                                   # ids that actually have a label row
    ordered_ids = [i for i in blob["ids"] if i in labelled]                           # cache order, filtered to labelled ids

    tr, va, te = split_indices(len(ordered_ids), CONFIG["train_ratio"],
                               CONFIG["val_ratio"], CONFIG["split_seed"])
    split_of = {}                                          # id -> "train"/"val"/"test"
    for i in tr: split_of[ordered_ids[i]] = "train"        # noqa: E701 - deliberately compact, three near-identical lines
    for i in va: split_of[ordered_ids[i]] = "val"          # noqa: E701
    for i in te: split_of[ordered_ids[i]] = "test"         # noqa: E701

    table = labels[labels[id_col].isin(ordered_ids)].copy()   # only the rows that made it into the dataset
    table["split"] = table[id_col].map(split_of)               # attach each row's split
    table["family"] = table["formula"].apply(classify_family)  # attach each row's structural family
    table.insert(0, "source", source_name)                      # "matbench" or "aflow", first column
    return table


def main():
    out_root = os.path.join(PROJECT_ROOT, CONFIG["out_dir"])
    os.makedirs(out_root, exist_ok=True)

    print("=" * 78)
    print("  STEP 40 - sorting results and training data by structural family")
    print("=" * 78)

    # ---- classify the GNoME screen -------------------------------------------
    print(f"\nLoading {CONFIG['gnome_screen_csv']}...")
    gnome = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["gnome_screen_csv"]))
    gnome["family"] = gnome["formula"].apply(classify_family)   # one family label per candidate
    print(f"  {len(gnome)} candidates classified")

    # ---- classify both training sets -----------------------------------------
    print("\nReconstructing the matbench train/val/test split and classifying...")
    matbench = training_set_table(CONFIG["matbench_labels_csv"], CONFIG["matbench_graphs_pt"], "mb_id", "matbench")
    print(f"  {len(matbench)} crystals classified")

    print("\nReconstructing the AFLOW train/val/test split and classifying...")
    aflow = training_set_table(CONFIG["aflow_labels_csv"], CONFIG["aflow_graphs_pt"], "gid", "aflow")
    print(f"  {len(aflow)} crystals classified")

    training = pd.concat([matbench, aflow], ignore_index=True, sort=False)   # combined, ragged columns kept as-is

    # ---- classify the extra per-material candidate tables ---------------------
    # Each is independent of the base GNoME screen - own columns (ALIGNN's own
    # predictions, agreement flags, recalibrated confidence tiers), own row
    # set (a shortlist, not all 33,118 candidates) - so each gets its own
    # family split rather than being folded into `gnome`.
    extra_tables = {}   # stem -> classified DataFrame
    for stem, rel_path in CONFIG["extra_candidate_tables"]:
        path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(path):
            print(f"\n  (skipping {rel_path} - not found)")
            continue
        print(f"\nLoading {rel_path}...")
        df = pd.read_csv(path)
        df["family"] = df["formula"].apply(classify_family)
        print(f"  {len(df)} rows classified")
        extra_tables[stem] = df

    # ---- write one directory per family ---------------------------------------
    extra_families = set().union(*(set(df["family"]) for df in extra_tables.values())) if extra_tables else set()
    families = sorted(set(gnome["family"]) | set(training["family"]) | extra_families)   # every family seen in any source
    print(f"\nWriting {len(families)} family directories under {out_root}/...")
    summary_rows = []
    for family in families:
        family_dir = os.path.join(out_root, slugify(family))
        os.makedirs(family_dir, exist_ok=True)

        g_sub = gnome[gnome["family"] == family]
        t_sub = training[training["family"] == family]

        g_sub.to_csv(os.path.join(family_dir, "gnome_candidates.csv"), index=False)
        t_sub.to_csv(os.path.join(family_dir, "training_set.csv"), index=False)

        g_low = g_sub[g_sub.get("Kappa_r9_gamma", pd.Series(dtype=float)) <= 1.0] if "Kappa_r9_gamma" in g_sub.columns else g_sub.iloc[0:0]
        g_low.to_csv(os.path.join(family_dir, "gnome_low_kappa.csv"), index=False)   # previously computed for the summary only, never written out

        extra_written = 0
        for stem, df in extra_tables.items():
            sub = df[df["family"] == family]
            if len(sub):   # skip writing an empty file for a family this particular shortlist has nothing in
                sub.to_csv(os.path.join(family_dir, f"{stem}.csv"), index=False)
                extra_written += len(sub)

        summary_rows.append({
            "family": family, "gnome_candidates": len(g_sub), "gnome_low_kappa": len(g_low),
            "matbench_train": len(t_sub[(t_sub.source == "matbench") & (t_sub.split == "train")]),
            "matbench_val": len(t_sub[(t_sub.source == "matbench") & (t_sub.split == "val")]),
            "matbench_test": len(t_sub[(t_sub.source == "matbench") & (t_sub.split == "test")]),
            "aflow_train": len(t_sub[(t_sub.source == "aflow") & (t_sub.split == "train")]),
            "aflow_val": len(t_sub[(t_sub.source == "aflow") & (t_sub.split == "val")]),
            "aflow_test": len(t_sub[(t_sub.source == "aflow") & (t_sub.split == "test")]),
            "extra_shortlist_rows": extra_written,
        })

    summary = pd.DataFrame(summary_rows).sort_values("gnome_candidates", ascending=False)
    summary_path = os.path.join(out_root, "40_family_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Per-family data in {out_root}/<family-slug>/*.csv "
         f"(gnome_candidates, gnome_low_kappa, training_set, plus any shortlist tables that had rows)")


if __name__ == "__main__":
    main()
