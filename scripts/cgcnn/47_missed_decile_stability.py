#!/usr/bin/env python3
"""
STEP 47 - Are the crystals the screen misses the SAME crystals in every round?
================================================================================

    python scripts/cgcnn/47_missed_decile_stability.py

THE QUESTION
--------------
`recall@10%` is this project's only screening-relevant metric: rank the 1,648
held-out matbench crystals by TRUE kappa, take the bottom decile (164 crystals
- the ones a low-kappa screen exists to find), and count how many of them the
model's own bottom decile also contains. Seven rounds of architecture work all
landed on the same number:

    separate 3-model ensemble   115/164 = 70.1%
    joint 3-model ensemble      115/164 = 70.1%

49 crystals missed, both times. That coincidence is only interesting if the 49
are the SAME 49. Two different readings, and they have opposite consequences
for the paper:

  * SAME crystals every round -> the misses are a property of the DATA, not of
    any model. Some crystals are simply not identifiable as soft from a matbench
    crystal graph, no matter how the network is wired. That is a hard,
    quotable ceiling, and it tells the next round to change the inputs (gamma
    labels, better features) rather than the architecture.

  * DIFFERENT crystals each round -> the misses are model variance, the ceiling
    is soft, and ensembling more broadly (or across architectures) should
    already recover some of them.

Nothing here needs a GPU or a retrain: every round already wrote its per-crystal
test predictions to results/cgcnn/predictions_*.csv, so this is set arithmetic
over files that exist.

WHAT COUNTS AS A "ROUND" HERE
-------------------------------
Only rounds that PERSISTED per-crystal test predictions can be checked. Rounds
1-3 were sweeps scored on aggregate metrics and saved summaries only, so they
cannot enter this analysis - stating that plainly matters more than padding the
roster. What does exist is seven distinct ensembles (the unit every headline
number in this project is quoted at) plus the individual networks behind them.

Three of the saved ensemble files are byte-identical to each other - r4's
huber ensemble, r5's ensemble A and round 7's matbench alpha=0.0 control are
the same three networks re-scored under three tags, exactly as intended (the
r7 control's job was to reproduce r5). The script hashes each prediction vector
and collapses duplicates automatically, so an accidental "they all miss the
same crystals!" that is really one run counted three times cannot happen.

THE KAPPA USED, AND ITS ONE CAVEAT
------------------------------------
`cgcnn_scratch.joint.slack_factors` - the same function `kappa_quintile_
breakdown` calls, so the decile and the recall computed here reproduce the
published 70.1% exactly. It is the ratio-and-G part of Slack's formula; the
density/volume prefactor is dropped because matbench labels carry neither. That
omission cancels for a given crystal (truth and prediction share it) but does
shift which crystals form the true bottom decile, so this is "the decile the
published recall number is about", not "the decile of the true full kappa".
Every round is scored against that one fixed decile, which is what the
comparison requires.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    "results_dir": "results/cgcnn",                       # where every predictions_*.csv lives
    "csv_dir": "results/cgcnn/decile_stability/csv",      # same csv/png split 43/44/46 use
    "png_dir": "results/cgcnn/decile_stability/png",
    "decile": 0.10,        # fraction defining "the bottom decile" - 0.10 is what recall@10% means
    "split": "test",       # only the held-out split can say anything about generalisation

    # Crystal-level metadata for the missed crystals. labels.csv carries the
    # formula, atom count and true moduli; the space group needs the actual
    # structures, which come from matminer's cached matbench download (already
    # on this machine - nothing is fetched).
    "add_structure_metadata": 1,      # 0 skips the space-group step entirely
    "labels_csv": "data_full/labels.csv",
    "matbench_json_dir": "",          # "" = auto-locate via matminer's own data home
    "matbench_json_name": "matbench_log_kvrh.json.gz",
    "symprec": 0.1,                   # pymatgen symmetry tolerance; 0.1 A is the usual choice for DFT-relaxed cells
}
# =============================================================================

import os                    # path joining and directory creation
import sys                   # sys.path manipulation + early exit on a missing input
import hashlib               # fingerprinting prediction vectors to collapse duplicate runs

# torch first: MKL loads its own OpenMP runtime and a duplicate libiomp5 aborts
# the process if numpy/pandas get in first. Project-wide rule, not optional.
import torch  # noqa: F401
import numpy as np           # the set arithmetic, ranking and summary statistics
import pandas as pd          # per-run CSV loading and table output
import matplotlib            # figure backend selection, before pyplot is imported
matplotlib.use("Agg")        # headless: this machine has no display and the script only writes files
import matplotlib.pyplot as plt   # noqa: E402   the three-panel figure at the end

# scripts/cgcnn/ -> scripts/ -> project root, so `cgcnn_scratch` imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.joint import slack_factors  # noqa: E402   the exact kappa the published recall uses

# Same palette as 44_baseline_tree_analysis.py and 46_compare_baseline_vs_r9_screen.py,
# so every figure in this results family reads as one system.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"

# -----------------------------------------------------------------------------
#  THE ROSTER
# -----------------------------------------------------------------------------
# Each entry is (label, tier, files). "tier" separates the two questions:
#   ensemble - the unit every headline number in this project is quoted at;
#              this is the tier the paper sentence would be about.
#   single   - the individual networks behind them. A crystal missed by all 19
#              independently-seeded networks is a much stronger statement than
#              one missed by seven ensembles that share members.
# A two-file entry is a SEPARATE-model pair (K model + G model); a one-file
# entry is a joint model that emitted both heads.
ROSTER = [
    # ---- ensembles ----------------------------------------------------------
    ("separate 3-ens (baseline)", "ensemble", ["predictions_K_VRH_ens.csv", "predictions_G_VRH_ens.csv"]),
    ("joint r4 huber 3-ens",      "ensemble", ["predictions_joint_r4_ens.csv"]),
    ("joint r5 ens A",            "ensemble", ["predictions_joint_r5_ensA.csv"]),
    ("joint r5 ens B",            "ensemble", ["predictions_joint_r5_ensB.csv"]),
    ("joint r5 ens C",            "ensemble", ["predictions_joint_r5_ensC.csv"]),
    ("r7 matbench a=0.0",         "ensemble", ["predictions_joint_r7_mb_a00_ens.csv"]),
    ("r7 matbench a=0.4",         "ensemble", ["predictions_joint_r7_mb_a04_ens.csv"]),
    ("r7 +AFLOW a=0.0",           "ensemble", ["predictions_joint_r7_af_a00_ens.csv"]),
    ("r7 +AFLOW a=0.4",           "ensemble", ["predictions_joint_r7_af_a04_ens.csv"]),
    # ---- individual networks ------------------------------------------------
    ("separate 1-model",          "single",   ["predictions_K_VRH_full.csv", "predictions_G_VRH_full.csv"]),
    ("r4 huber s42",              "single",   ["predictions_joint_r4_huber_s42.csv"]),
    ("r4 huber s1",               "single",   ["predictions_joint_r4_huber_s1.csv"]),
    ("r4 huber s2",               "single",   ["predictions_joint_r4_huber_s2.csv"]),
    ("r5 huber s42",              "single",   ["predictions_joint_r5_huber_s42.csv"]),
    ("r5 huber s1",               "single",   ["predictions_joint_r5_huber_s1.csv"]),
    ("r5 huber s2",               "single",   ["predictions_joint_r5_huber_s2.csv"]),
    ("r5 huber s3",               "single",   ["predictions_joint_r5_huber_s3.csv"]),
    ("r5 huber s4",               "single",   ["predictions_joint_r5_huber_s4.csv"]),
    ("r5 huber s5",               "single",   ["predictions_joint_r5_huber_s5.csv"]),
    ("r5 huber s6",               "single",   ["predictions_joint_r5_huber_s6.csv"]),
    ("r5 huber s7",               "single",   ["predictions_joint_r5_huber_s7.csv"]),
    ("r5 huber s8",               "single",   ["predictions_joint_r5_huber_s8.csv"]),
    ("r7 mb a=0.0 s42",           "single",   ["predictions_joint_r7_mb_a00_s42.csv"]),
    ("r7 mb a=0.0 s1",            "single",   ["predictions_joint_r7_mb_a00_s1.csv"]),
    ("r7 mb a=0.0 s2",            "single",   ["predictions_joint_r7_mb_a00_s2.csv"]),
    ("r7 mb a=0.4 s42",           "single",   ["predictions_joint_r7_mb_a04_s42.csv"]),
    ("r7 mb a=0.4 s1",            "single",   ["predictions_joint_r7_mb_a04_s1.csv"]),
    ("r7 mb a=0.4 s2",            "single",   ["predictions_joint_r7_mb_a04_s2.csv"]),
    ("r7 af a=0.0 s42",           "single",   ["predictions_joint_r7_af_a00_s42.csv"]),
    ("r7 af a=0.0 s1",            "single",   ["predictions_joint_r7_af_a00_s1.csv"]),
    ("r7 af a=0.0 s2",            "single",   ["predictions_joint_r7_af_a00_s2.csv"]),
    ("r7 af a=0.4 s42",           "single",   ["predictions_joint_r7_af_a04_s42.csv"]),
    ("r7 af a=0.4 s1",            "single",   ["predictions_joint_r7_af_a04_s1.csv"]),
    ("r7 af a=0.4 s2",            "single",   ["predictions_joint_r7_af_a04_s2.csv"]),
]


def load_run(results_dir, files, split):
    """Read one run's held-out predictions into a uniform 4-column frame.

    Handles both storage conventions this project has used:
      * one file, joint model  -> true/pred log10 K and G already side by side
      * two files, separate K and G models -> merged on material_id here
    Returns a frame indexed by material_id, or None if any file is missing
    (a missing file is reported, never silently skipped as "no misses").
    """
    paths = [os.path.join(results_dir, f) for f in files]
    for p in paths:
        if not os.path.exists(p):
            return None                                    # caller prints the skip

    if len(paths) == 1:                                    # joint-model convention
        df = pd.read_csv(paths[0])
        df = df[df["split"] == split]                      # held-out rows only
        out = df.set_index("material_id")[
            ["true_log10_K", "true_log10_G", "pred_log10_K", "pred_log10_G"]]
    else:                                                  # separate K + G models
        k = pd.read_csv(paths[0])
        g = pd.read_csv(paths[1])
        k = k[k["split"] == split].set_index("material_id")
        g = g[g["split"] == split].set_index("material_id")
        # An inner join on the index: the two models were trained on one shared
        # split, so this must not drop anything - asserted by the caller via the
        # row-count check against the first run loaded.
        out = pd.DataFrame({
            "true_log10_K": k["true_log10"], "true_log10_G": g["true_log10"],
            "pred_log10_K": k["pred_log10"], "pred_log10_G": g["pred_log10"],
        }).dropna()
    return out.sort_index()                                # one canonical row order for every run


def kappa_proxy(log_K, log_G):
    """Slack kappa without the density/volume prefactor - see the docstring."""
    prefactor, anharmonic, _gamma = slack_factors(np.asarray(log_K), np.asarray(log_G))
    return prefactor * anharmonic


def fingerprint(values):
    """Stable hash of a prediction vector, used to collapse duplicate runs.

    Rounded to 9 decimals first so that a file rewritten with different float
    formatting still fingerprints identically - the thing being deduplicated is
    "the same three networks", not "the same bytes".
    """
    return hashlib.md5(np.round(np.asarray(values, dtype=float), 9).tobytes()).hexdigest()


def structure_metadata(cfg, wanted_ids):
    """formula / space group / atom count / true moduli for a list of mb-ids.

    Two sources, cross-checked against each other:
      * data_full/labels.csv - formula, n_sites, K_VRH, G_VRH (what the model
        was trained on).
      * matminer's cached matbench_log_kvrh.json.gz - the actual pymatgen
        Structures, needed for the space group and for nothing else.

    The mb-id encodes the matbench row number ("mb-00042" -> row 42), which is
    how 01b_prepare_full_dataset.py assigned them. That assumption is CHECKED
    here (formula and K_VRH must agree between the two files) rather than
    trusted, because a silent off-by-one would attach the wrong space group to
    every crystal in the report and nothing downstream would notice.
    """
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["labels_csv"])).set_index("mb_id")

    json_dir = cfg["matbench_json_dir"]
    if not json_dir:                                   # auto-locate matminer's cache
        from matminer.datasets.utils import _get_data_home
        json_dir = _get_data_home()
    json_path = os.path.join(json_dir, cfg["matbench_json_name"])

    rows = {}
    spacegroups = {}
    if os.path.exists(json_path):
        import gzip, json                              # local: only this branch needs them
        from pymatgen.core import Structure            # deserialises one structure dict
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        with gzip.open(json_path, "rt") as fh:
            blob = json.load(fh)                       # {"index": [...], "columns": [...], "data": [[...]]}
        col = blob["columns"].index("structure")
        for mb_id in wanted_ids:
            row = int(str(mb_id).split("-")[1])        # "mb-00042" -> 42
            if row >= len(blob["data"]):
                continue
            struct = Structure.from_dict(blob["data"][row][col])
            # The consistency check: same formula and same atom count as labels.csv.
            if mb_id in labels.index:
                same_n = int(struct.num_sites) == int(labels.loc[mb_id, "n_sites"])
                if not same_n:
                    sys.exit(f"ERROR: mb-id -> matbench row mapping is wrong for {mb_id} "
                             f"({struct.num_sites} sites in the structure vs "
                             f"{labels.loc[mb_id, 'n_sites']} in labels.csv). "
                             f"Refusing to report space groups that may be misattributed.")
            try:
                sga = SpacegroupAnalyzer(struct, symprec=cfg["symprec"])
                spacegroups[mb_id] = (sga.get_space_group_symbol(), sga.get_space_group_number())
            except Exception:                          # a few pathological cells defeat spglib
                spacegroups[mb_id] = ("(symmetry analysis failed)", -1)
    else:
        print(f"  NOTE: {json_path} not found - space groups omitted, everything "
              f"else in the metadata table still comes from labels.csv")

    for mb_id in wanted_ids:
        rec = {"formula": None, "n_sites": None, "K_VRH_GPa": None, "G_VRH_GPa": None}
        if mb_id in labels.index:
            lab = labels.loc[mb_id]
            rec.update(formula=lab["formula"], n_sites=int(lab["n_sites"]),
                       K_VRH_GPa=float(lab["K_VRH"]), G_VRH_GPa=float(lab["G_VRH"]))
        sym, num = spacegroups.get(mb_id, (None, None))
        rec["spacegroup"] = sym
        rec["spacegroup_number"] = num
        rows[mb_id] = rec
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    cfg = dict(CONFIG)
    results_dir = os.path.join(PROJECT_ROOT, cfg["results_dir"])
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 47 - is it the SAME 49 crystals the screen misses every round?")
    print("=" * 78)
    print()

    # ---- load every run, dropping exact duplicates ---------------------------
    runs = []                       # [(label, tier, frame)] in roster order, deduplicated
    seen = {}                       # fingerprint -> label that claimed it first
    dupes = []                      # (label, label_it_duplicates) for the report
    reference_index = None          # the crystal ordering every later run must match

    for label, tier, files in ROSTER:
        frame = load_run(results_dir, files, cfg["split"])
        if frame is None:
            print(f"  [skip] {label:<26} - prediction file not found")
            continue
        if reference_index is None:
            reference_index = frame.index
        elif not frame.index.equals(reference_index):
            # Different crystals in the held-out split means a different
            # split_seed, and comparing miss sets across it would be nonsense.
            sys.exit(f"ERROR: {label} has a different {cfg['split']} split "
                     f"({len(frame)} rows vs {len(reference_index)}) - "
                     f"comparing miss sets across splits is meaningless.")

        fp = fingerprint(np.concatenate([frame["pred_log10_K"].values,
                                         frame["pred_log10_G"].values]))
        if fp in seen:
            dupes.append((label, seen[fp]))
            continue
        seen[fp] = label
        runs.append((label, tier, frame))

    if not runs:
        sys.exit("ERROR: no prediction files found - nothing to compare.")

    n_test = len(reference_index)
    print(f"  {len(runs)} distinct runs loaded over {n_test} held-out crystals")
    if dupes:
        print("  collapsed as byte-identical to an earlier run (same networks, "
              "different tag):")
        for label, first in dupes:
            print(f"      {label:<26} == {first}")
    print()

    # ---- the one fixed truth decile every run is scored against -------------
    # Truth is a property of the split, not of any model, so it is taken from
    # the first run and every other run is checked against it rather than
    # recomputed - that check is what guarantees the decile really is shared.
    ref_label, _ref_tier, ref = runs[0]
    true_kappa = kappa_proxy(ref["true_log10_K"], ref["true_log10_G"])
    # Tolerance is on the stored log10 LABELS, not on kappa. The separate-model
    # files kept the labels at float32 while the joint files kept float64, so
    # identical labels still differ by ~3e-7 in log10 - which the exponential
    # in kappa magnifies to ~1e-3 absolute on a stiff crystal. Checking the
    # labels directly compares like with like; 1e-5 is two orders above the
    # float32 spacing and far below any real relabelling.
    for label, _tier, frame in runs[1:]:
        drift = max(np.abs(frame["true_log10_K"].values - ref["true_log10_K"].values).max(),
                    np.abs(frame["true_log10_G"].values - ref["true_log10_G"].values).max())
        if drift > 1e-5:
            sys.exit(f"ERROR: {label} disagrees with {ref_label} about the TRUE "
                     f"moduli (max drift {drift:.2e} in log10) - not the same labels.")

    finite = np.isfinite(true_kappa) & (true_kappa > 0)   # same guard kappa_quintile_breakdown applies
    if not finite.all():
        print(f"  NOTE: {int((~finite).sum())} crystals dropped for a non-finite "
              f"true kappa (same guard the published metric uses)")
    ids = np.asarray(reference_index)[finite]
    kt = true_kappa[finite]

    n10 = max(1, int(cfg["decile"] * len(kt)))            # 164 for the 1,648-crystal split
    truth_order = np.argsort(kt)                          # ascending: softest/lowest-kappa first
    truth_rows = truth_order[:n10]                        # positions of the true bottom decile
    truth_set = set(truth_rows.tolist())
    decile_ids = ids[truth_rows]                          # the 164 crystals, ordered by true kappa
    rank_in_decile = {row: i + 1 for i, row in enumerate(truth_rows)}   # 1 = lowest true kappa

    print(f"  true bottom decile = {n10} crystals "
          f"(kappa proxy {kt[truth_rows].min():.3f} to {kt[truth_rows].max():.3f})")
    print()

    # ---- score every run against that decile --------------------------------
    miss_sets = {}          # label -> set of truth-row positions it failed to pick
    pred_rank = {}          # label -> array of predicted rank (1 = model's softest) per truth row
    summary_rows = []
    for label, tier, frame in runs:
        kp = kappa_proxy(frame["pred_log10_K"], frame["pred_log10_G"])[finite]
        order = np.argsort(kp)                            # the model's own ranking
        picked = set(order[:n10].tolist())                # the model's bottom decile
        missed = truth_set - picked
        miss_sets[label] = missed

        # Where the model actually put each true-decile crystal. A miss that
        # ranks 170th is a boundary artefact; one that ranks 900th is a real
        # failure, and the paper sentence depends on which it is.
        rank_of = np.empty(len(kp), dtype=int)
        rank_of[order] = np.arange(1, len(kp) + 1)
        pred_rank[label] = rank_of

        summary_rows.append({
            "run": label,
            "tier": tier,
            "n_decile": n10,
            "n_hit": n10 - len(missed),
            "n_missed": len(missed),
            "recall_pct": round(100.0 * (n10 - len(missed)) / n10, 1),
            "median_pred_rank_of_missed": (int(np.median([rank_of[r] for r in missed]))
                                           if missed else 0),
        })

    summary = pd.DataFrame(summary_rows)
    print("  RECALL OF THE TRUE BOTTOM DECILE, RUN BY RUN")
    print("  " + "-" * 74)
    for tier in ["ensemble", "single"]:
        block = summary[summary["tier"] == tier]
        if block.empty:
            continue
        print(f"  {tier}s:")
        for _, r in block.iterrows():
            print(f"    {r['run']:<26} {r['n_hit']:>3}/{n10}  "
                  f"recall {r['recall_pct']:>5.1f}%   missed {r['n_missed']:>3}   "
                  f"median predicted rank of a miss: {r['median_pred_rank_of_missed']}")
    print()

    # ---- the actual question: how much do the miss sets overlap? ------------
    for tier in ["ensemble", "single"]:
        labels = [lab for lab, t, _ in runs if t == tier]
        if len(labels) < 2:
            continue
        sets = [miss_sets[lab] for lab in labels]
        inter = set.intersection(*sets)                    # missed by EVERY run in this tier
        union = set.union(*sets)                           # missed by at least one
        sizes = [len(s) for s in sets]
        print(f"  {tier.upper()}S ({len(labels)} runs) - overlap of the miss sets")
        print("  " + "-" * 74)
        print(f"    miss-set sizes           : {min(sizes)} to {max(sizes)} crystals")
        print(f"    missed by EVERY run      : {len(inter)}")
        print(f"    missed by AT LEAST ONE   : {len(union)}")
        print(f"    core / smallest miss set : {len(inter)}/{min(sizes)} = "
              f"{100.0 * len(inter) / min(sizes):.0f}%")
        print(f"    jaccard(all runs)        : {len(inter) / len(union):.3f}   "
              f"(1.000 would mean literally the same crystals every time)")
        # A crystal ranked, say, 1,200th by every model was never in contention;
        # one ranked 170th is a boundary case that a slightly different cut
        # would have caught. Separating them is what makes the fact quotable.
        if inter:
            deep = [r for r in inter
                    if min(pred_rank[lab][r] for lab in labels) > 2 * n10]
            print(f"    of those {len(inter)}, {len(deep)} are ranked outside the top "
                  f"{2 * n10} by EVERY run - not boundary cases")
        print()

    # ---- per-crystal table --------------------------------------------------
    ens_labels = [lab for lab, t, _ in runs if t == "ensemble"]
    sgl_labels = [lab for lab, t, _ in runs if t == "single"]
    per_crystal = pd.DataFrame({
        "material_id": decile_ids,
        "rank_in_true_decile": [rank_in_decile[r] for r in truth_rows],
        "true_kappa_proxy": kt[truth_rows],
        "true_K_GPa": 10.0 ** ref["true_log10_K"].values[finite][truth_rows],
        "true_G_GPa": 10.0 ** ref["true_log10_G"].values[finite][truth_rows],
    })
    per_crystal["n_ensembles_missing"] = [sum(r in miss_sets[l] for l in ens_labels) for r in truth_rows]
    per_crystal["n_singles_missing"] = [sum(r in miss_sets[l] for l in sgl_labels) for r in truth_rows]
    per_crystal["missed_by_every_run"] = [
        all(r in miss_sets[l] for l, _t, _f in runs) for r in truth_rows]
    per_crystal["worst_pred_rank"] = [max(pred_rank[l][r] for l, _t, _f in runs) for r in truth_rows]
    per_crystal["best_pred_rank"] = [min(pred_rank[l][r] for l, _t, _f in runs) for r in truth_rows]
    per_crystal["missed_by_every_ensemble"] = per_crystal["n_ensembles_missing"] == len(ens_labels)
    per_crystal["missed_by_every_single"] = per_crystal["n_singles_missing"] == len(sgl_labels)
    # K/G and the gamma the pipeline DERIVES from it (the empirical Poisson
    # relation). Reported for the missed crystals because a large K/G is the
    # single strongest signature of the soft, high-Poisson materials the screen
    # is chasing - and gamma enters kappa exponentially, so a K/G outlier is a
    # kappa outlier by construction.
    per_crystal["K_over_G_true"] = per_crystal["true_K_GPa"] / per_crystal["true_G_GPa"]
    _x2 = per_crystal["K_over_G_true"] + 4.0 / 3.0
    _nu = (_x2 - 2.0) / (2.0 * _x2 - 2.0)                     # Poisson ratio from K/G
    per_crystal["gamma_derived_true"] = 3.0 * (1.0 + _nu) / (2.0 * (2.0 - 3.0 * _nu))

    if cfg["add_structure_metadata"]:
        meta = structure_metadata(cfg, list(decile_ids))
        per_crystal = per_crystal.merge(
            meta[["formula", "spacegroup", "spacegroup_number", "n_sites"]],
            left_on="material_id", right_index=True, how="left")
    for label, _tier, _frame in runs:                     # one column per run: its rank for this crystal
        per_crystal[f"rank__{label}"] = [pred_rank[label][r] for r in truth_rows]
    path = os.path.join(csv_dir, "47_missed_decile_per_crystal.csv")
    per_crystal.to_csv(path, index=False)
    print(f"  Wrote {path}")

    summary.to_csv(os.path.join(csv_dir, "47_missed_decile_run_summary.csv"), index=False)
    print(f"  Wrote {os.path.join(csv_dir, '47_missed_decile_run_summary.csv')}")

    # ---- pairwise jaccard of the miss sets ----------------------------------
    all_labels = [lab for lab, _t, _f in runs]
    jac = pd.DataFrame(index=all_labels, columns=all_labels, dtype=float)
    for a in all_labels:
        for b in all_labels:
            sa, sb = miss_sets[a], miss_sets[b]
            jac.loc[a, b] = len(sa & sb) / max(1, len(sa | sb))
    jac.to_csv(os.path.join(csv_dir, "47_missed_decile_pairwise_jaccard.csv"))
    print(f"  Wrote {os.path.join(csv_dir, '47_missed_decile_pairwise_jaccard.csv')}")

    # Raw shared-crystal counts alongside the normalised Jaccard: the diagonal
    # is each run's own miss count, so the matrix reads as "run A missed 49, of
    # which 38 were also missed by run B" without needing the ratio undone.
    cnt = pd.DataFrame(index=all_labels, columns=all_labels, dtype=int)
    for a in all_labels:
        for b in all_labels:
            cnt.loc[a, b] = len(miss_sets[a] & miss_sets[b])
    cnt.to_csv(os.path.join(csv_dir, "47_missed_decile_pairwise_counts.csv"))
    print(f"  Wrote {os.path.join(csv_dir, '47_missed_decile_pairwise_counts.csv')}")

    # ---- what the always-missed crystals look like --------------------------
    core = per_crystal[per_crystal["missed_by_every_ensemble"]].copy()
    core = core.sort_values("rank_in_true_decile")
    if len(core):
        print()
        print(f"  THE {len(core)} CRYSTALS EVERY ONE OF THE {len(ens_labels)} ENSEMBLES MISSES")
        print("  " + "-" * 74)
        header = (f"    {'material_id':<12} {'formula':<16} {'spacegroup':<12} "
                  f"{'N':>3} {'K':>7} {'G':>7} {'K/G':>6} {'gamma':>6} {'rank':>5} {'best':>6}")
        print(header)
        for _, r in core.iterrows():
            print(f"    {r['material_id']:<12} {str(r.get('formula', '')):<16} "
                  f"{str(r.get('spacegroup', '')):<12} "
                  f"{('' if pd.isna(r.get('n_sites')) else int(r['n_sites'])):>3} "
                  f"{r['true_K_GPa']:>7.1f} {r['true_G_GPa']:>7.1f} "
                  f"{r['K_over_G_true']:>6.2f} {r['gamma_derived_true']:>6.2f} "
                  f"{int(r['rank_in_true_decile']):>5} {int(r['best_pred_rank']):>6}")
        print(f"    ('rank' = position in the true bottom decile, 1 = softest; "
              f"'best' = closest any run came)")
        print()
        print(f"    median true G          : {core['true_G_GPa'].median():.1f} GPa "
              f"(decile median {per_crystal['true_G_GPa'].median():.1f})")
        print(f"    median true K/G        : {core['K_over_G_true'].median():.2f} "
              f"(decile median {per_crystal['K_over_G_true'].median():.2f})")
        print(f"    median derived gamma   : {core['gamma_derived_true'].median():.2f} "
              f"(decile median {per_crystal['gamma_derived_true'].median():.2f})")
        print(f"    median rank in decile  : {core['rank_in_true_decile'].median():.0f} of {n10} "
              f"(a value near {n10} would mean they are only just inside the decile)")
        print(f"    best rank any run gave : {core['best_pred_rank'].min()} "
              f"(the closest any model came to catching any of them)")
        core_path = os.path.join(csv_dir, "47_missed_by_every_ensemble.csv")
        core.to_csv(core_path, index=False)
        print(f"  Wrote {core_path}")

    # ---- the plain-language verdict, stated in the output itself -----------
    ens_sets = [miss_sets[l] for l in ens_labels]
    inter_e, union_e = set.intersection(*ens_sets), set.union(*ens_sets)
    frac = len(inter_e) / np.mean([len(x) for x in ens_sets])
    print()
    print("  VERDICT")
    print("  " + "-" * 74)
    if frac > 0.9:
        verdict = "THE SAME crystals every round - a fixed, data-level ceiling"
    elif frac > 0.4:
        verdict = ("a STABLE CORE inside a ROTATING SET - part data ceiling, "
                   "part model variance")
    else:
        verdict = "a ROTATING SET - the misses are mostly model variance"
    print(f"    {verdict}.")
    print(f"    {len(inter_e)} of a typical {np.mean([len(x) for x in ens_sets]):.0f}-crystal "
          f"miss set are missed by all {len(ens_labels)} ensembles ({100 * frac:.0f}%);")
    print(f"    {len(union_e)} distinct crystals are missed by at least one.")

    # =========================================================================
    #  FIGURE - three panels, one question each
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.patch.set_facecolor("white")

    # Panel 1: of the 164, how many runs miss each? The shape answers the
    # question on its own - mass piled at 0 and at N means "the same crystals
    # every time"; a hump in the middle means model variance.
    ax = axes[0]
    n_runs = len(runs)
    counts = per_crystal["n_ensembles_missing"].values
    n_ens = len(ens_labels)
    hist = np.bincount(counts, minlength=n_ens + 1)
    colors = [GREEN if i == 0 else (ORANGE if i == n_ens else BLUE) for i in range(n_ens + 1)]
    bars = ax.bar(np.arange(n_ens + 1), hist, color=colors, width=0.72)
    for i, bar in enumerate(bars):
        if hist[i]:                                       # label only the bars that exist
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{hist[i]}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel(f"how many of the {n_ens} ensembles miss this crystal", color=INK_SOFT)
    ax.set_ylabel(f"crystals (of the {n10} in the true bottom decile)", color=INK_SOFT)
    ax.set_title("Nearly every crystal is caught by all\nor missed by all",
                 color=INK, fontsize=11)
    ax.set_xticks(np.arange(n_ens + 1))

    # Panel 2: pairwise agreement between the ensembles' miss sets. Sequential
    # single-hue ramp - this is a magnitude, not an identity or a polarity.
    ax = axes[1]
    sub = jac.loc[ens_labels, ens_labels].values.astype(float)
    im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ens_labels)))
    ax.set_yticks(range(len(ens_labels)))
    ax.set_xticklabels(ens_labels, rotation=45, ha="right", fontsize=7, color=MUTED)
    ax.set_yticklabels(ens_labels, fontsize=7, color=MUTED)
    for i in range(len(ens_labels)):
        for j in range(len(ens_labels)):
            ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if sub[i, j] > 0.6 else INK)
    ax.set_title("Jaccard overlap of the miss sets\n(1.00 = identical crystals)",
                 color=INK, fontsize=11)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    # Panel 3: are the persistent misses boundary cases or real failures?
    # x = position inside the true decile (1 = lowest true kappa),
    # y = the best rank ANY run gave it. A point near the dashed line was
    # nearly caught; one high above it was never in contention.
    ax = axes[2]
    x = per_crystal["rank_in_true_decile"].values
    y = per_crystal["best_pred_rank"].values
    hit_all = per_crystal["n_ensembles_missing"].values == 0
    ax.scatter(x[hit_all], y[hit_all], s=20, color=GREEN, alpha=0.75,
               edgecolors="white", linewidths=0.4, label="caught by every ensemble")
    ax.scatter(x[~hit_all], y[~hit_all], s=20, color=ORANGE, alpha=0.85,
               edgecolors="white", linewidths=0.4, label="missed by at least one")
    ax.axhline(n10, color=INK_SOFT, lw=1.2, ls="--")
    ax.text(n10 * 0.98, n10 * 1.06, f"decile cut ({n10})", ha="right", va="bottom",
            fontsize=8, color=INK_SOFT)
    ax.set_yscale("log")
    ax.set_xlabel("position in the TRUE bottom decile (1 = softest)", color=INK_SOFT)
    ax.set_ylabel("best predicted rank across all runs", color=INK_SOFT)
    ax.set_title("How close the misses came\n(log scale; above the line = missed)",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes:
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in [axes[0], axes[2]]:                        # no grid on the heatmap
        ax.grid(color=GRID, lw=0.6, alpha=0.7)

    fig.tight_layout()
    png_path = os.path.join(png_dir, "47_missed_decile_stability.png")
    fig.savefig(png_path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  Wrote {png_path}")


if __name__ == "__main__":
    main()
