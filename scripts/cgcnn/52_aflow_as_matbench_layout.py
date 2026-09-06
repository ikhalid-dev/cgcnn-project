#!/usr/bin/env python3
"""
STEP 52 - Present the AFLOW set in the layout 02_train.py already expects.
================================================================================

    python scripts/cgcnn/52_aflow_as_matbench_layout.py

WHY THIS EXISTS
-----------------
Steps 43/44 found that the composition-only tree baseline beats round 9 on
AFLOW, and diagnosed the cause as UNDERFITTING rather than a real advantage:
round 9's training error on AFLOW K (0.103) is worse than the tree's test
error (0.059), and its test/train ratios sit at 1.08-1.23, the band where a
model never learned. Step 53's experiment tests that directly by training the
ORIGINAL single-target CGCNN on AFLOW with the matbench recipe - no dropout,
MSE, one target per model, the settings the 0.0630/0.0781 baselines were
trained with.

For that test to mean anything, `02_train.py` must run UNMODIFIED. It is the
script every baseline in this project was produced by; editing it to accept a
second dataset layout would change the very recipe the experiment is trying to
reproduce, and would silently invalidate every earlier number it produced.

So instead of changing the reader, this changes the data to match it.
`load_dataset_for` wants exactly two things in a directory:

    graphs.pt    the {"ids", "graphs"} cache
    labels.csv   a table with an `mb_id` column and the target column

AFLOW already has both, under different names (`gamma_graphs.pt`,
`gamma_labels.csv`, keyed on `gid`). This writes a directory that satisfies
the reader, with the graph cache HARD-LINKED rather than copied so the ~240 MB
of graphs are not duplicated on disk.

WHAT IS AND IS NOT CHANGED
----------------------------
Renamed:  gid -> mb_id.  That is the whole transformation on the labels.
Kept:     every other column, including gamma and kappa_agl, so the same
          directory also serves 01_train_direct_kappa.py.
Order:    the row order of gamma_labels.csv and the id order of the graph
          cache are both preserved untouched. This matters more than it looks:
          `split_indices` permutes POSITIONS, so any reordering here would
          hand the experiment a different train/test split than round 9 and
          the tree were scored on, and the comparison would be meaningless.

The id order is CHECKED against the cache rather than assumed, and the script
exits if they disagree.
"""

# =============================================================================
#  CONFIG - every path and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "data_dir": "data_full",
    "src_cache": "gamma_graphs.pt",       # the AFLOW graph cache built by step 36
    "src_labels": "gamma_labels.csv",     # gid, formula, n_sites, K_VRH, G_VRH, gamma, kappa_agl, ...
    "id_column": "gid",                   # what the AFLOW labels key on

    # ---- output -------------------------------------------------------------
    # A subdirectory of data_dir, so `--data-dir data_full/aflow_matbench_layout`
    # is all 02_train.py needs.
    "out_subdir": "aflow_matbench_layout",
    "dest_id_column": "mb_id",            # what cgcnn_scratch/data.py's clean_labels expects
    "link_cache": 1,                      # 1 = hard-link the graph cache, 0 = copy it
}
# =============================================================================

import os                    # paths, linking, directory creation
import sys                   # early exit on a schema mismatch

# torch first: MKL loads its own OpenMP runtime and a duplicate libiomp5 aborts
# the process if numpy/pandas get in first. Project-wide rule, not optional.
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    cfg = dict(CONFIG)
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])
    out_dir = os.path.join(data_dir, cfg["out_subdir"])
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 52 - AFLOW in the layout 02_train.py expects")
    print("=" * 78)

    src_cache = os.path.join(data_dir, cfg["src_cache"])
    src_labels = os.path.join(data_dir, cfg["src_labels"])
    for path in (src_cache, src_labels):
        if not os.path.exists(path):
            sys.exit(f"ERROR: missing {path}. Build it first with "
                     f"scripts/cgcnn/36_prepare_gamma_dataset.py")

    # ---- labels: rename the id column, change nothing else ------------------
    labels = pd.read_csv(src_labels)
    if cfg["id_column"] not in labels.columns:
        sys.exit(f"ERROR: {cfg['src_labels']} has no '{cfg['id_column']}' column; "
                 f"found {list(labels.columns)}")
    labels = labels.rename(columns={cfg["id_column"]: cfg["dest_id_column"]})

    # ---- graph cache: hard-link if possible, else copy -----------------------
    dest_cache = os.path.join(out_dir, "graphs.pt")
    if os.path.exists(dest_cache):
        os.remove(dest_cache)              # stale link from a previous run
    if cfg["link_cache"]:
        try:
            os.link(src_cache, dest_cache)
            how = "hard-linked"
        except OSError:
            # Different filesystem, or a platform that refuses the link. Copying
            # is correct but costs ~240 MB, so it is reported rather than silent.
            import shutil
            shutil.copy(src_cache, dest_cache)
            how = "COPIED (hard link unavailable)"
    else:
        import shutil
        shutil.copy(src_cache, dest_cache)
        how = "copied"

    # ---- the check that makes the split trustworthy -------------------------
    # 02_train.py's split_indices permutes POSITIONS in the dataset, and
    # GraphCacheData builds its row order from the cache's own id list filtered
    # to the ids present in the labels. So the two must line up exactly, or this
    # experiment would be scored on a different split from round 9 and the tree
    # and nothing would be comparable.
    blob = torch.load(dest_cache, weights_only=False)
    cache_ids = list(blob["ids"])
    label_ids = labels[cfg["dest_id_column"]].astype(str).tolist()
    if cache_ids != label_ids:
        missing = set(label_ids) - set(cache_ids)
        extra = set(cache_ids) - set(label_ids)
        sys.exit(
            "ERROR: the graph cache's id order does not match the labels table.\n"
            f"  cache ids: {len(cache_ids)}   label ids: {len(label_ids)}\n"
            f"  in labels but not cache: {len(missing)}\n"
            f"  in cache but not labels: {len(extra)}\n"
            "Refusing to write a dataset whose split would not match the one "
            "round 9 and the tree baseline were scored on.")

    dest_labels = os.path.join(out_dir, "labels.csv")
    labels.to_csv(dest_labels, index=False)

    print(f"  graphs.pt   {how}  ({len(cache_ids)} crystals)")
    print(f"  labels.csv  {len(labels)} rows, "
          f"'{cfg['id_column']}' -> '{cfg['dest_id_column']}'")
    print(f"  id order CHECKED: cache and labels agree exactly")
    print()
    print(f"  wrote {cfg['data_dir']}/{cfg['out_subdir']}/")
    print()
    print("  02_train.py can now read it unmodified, e.g.:")
    print(f"    python scripts/cgcnn/02_train.py \\")
    print(f"        --data-dir {cfg['data_dir']}/{cfg['out_subdir']} \\")
    print(f"        --target K_VRH --tag aflow_mbrecipe_K_s42")
    for col in ("K_VRH", "G_VRH", "gamma", "kappa_agl"):
        if col in labels.columns:
            n_bad = int((labels[col] <= 0).sum())
            print(f"    column {col:<10} present, {n_bad} non-positive rows")


if __name__ == "__main__":
    main()
