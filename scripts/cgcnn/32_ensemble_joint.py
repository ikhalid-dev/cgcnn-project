#!/usr/bin/env python3
"""
STEP 32 - Combine several joint two-head models into one ensemble.
===================================================================

    python scripts/cgcnn/32_ensemble_joint.py --tags ens_s42,ens_s1,ens_s2

WHAT AN ENSEMBLE ACTUALLY DOES, AND WHY IT SHOULD HELP HERE
-----------------------------------------------------------
Each model is trained from a different random initialisation on the same data.
They therefore make DIFFERENT mistakes on the same crystal: one reads a bond
environment slightly too stiff, another slightly too soft. Averaging their
predictions cancels the part of the error that is idiosyncratic to a particular
initialisation (variance) and leaves only the part they all share (bias).

That is exactly the gap this project currently has. After three rounds:

    3-model ensemble of SEPARATE models   kappa 0.1897   prefactor 0.1160
    single joint model + Huber            kappa 0.2050   prefactor 0.1416
                                                  ^ anharmonic term 0.0891,
                                                    BETTER than the ensemble's
                                                    0.0911

The joint model already wins on the anharmonic half - the half it was built to
fix. It loses on the prefactor half, which is nothing but raw accuracy on K and
G, and that is precisely what a single model cannot match three of. Ensembling
the joint models attacks the one term that is still behind.

Averaging the SEPARATE models bought roughly 10% on individual accuracy
(K 0.0696 -> 0.0630, G 0.0836 -> 0.0781). If the same holds here the prefactor
should fall toward ~0.128 while the anharmonic advantage is kept.

WHY THE AVERAGE IS TAKEN IN LOG SPACE
--------------------------------------
The models are trained on log10 of the targets, the error is measured in log10,
and the physics is multiplicative (kappa is a product of factors). Averaging
raw GPa would let one member's large value dominate; averaging logs is a
geometric mean, which is the right notion of "middle" for a quantity whose
error is proportional rather than absolute.

THE SPREAD IS A FREE UNCERTAINTY ESTIMATE
------------------------------------------
The standard deviation across members, per crystal, is the output this project
uses for its Monte Carlo p05/p50/p95 intervals - the thing PINK's single
pre-trained model cannot produce at all. A tight spread means the members
agreed; a wide one flags a crystal whose prediction should not be trusted, and
is what stops the "best candidate" ranking being won by whichever prediction
happened to be noisiest.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # Directory holding predictions_31_<tag>.csv for each member.
    "results_dir": "results/cgcnn",
    # Comma-separated tags of the members to combine. These must be models that
    # shared one train/val/test split (same --split-seed), or the "test" rows
    # of one member are training rows of another and the score is meaningless.
    "tags": "ens_s42,ens_s1,ens_s2",
    # Name for the combined output files.
    "out_tag": "ens",
    # Which split to report the headline metrics on. "test" always; "train" and
    # "val" are printed too so the test/train ratio stays visible.
    "report_split": "test",
}
# =============================================================================

import argparse                       # CLI flag parsing, auto-generated from CONFIG
import json                           # writing the combined summary as JSON
import os                             # path joining and existence checks
import sys                            # sys.path manipulation to import the project package

# torch first: MKL loads its own OpenMP runtime and a duplicate libiomp5
# aborts the process if numpy/pandas get in first.
import torch  # noqa: F401           # imported only for its import-order side effect, not called directly
import numpy as np                    # array stacking/statistics for the ensemble average and spread
import pandas as pd                   # per-member CSV loading and combined-output writing

# __file__ is this script's own path; three dirname() calls climb
# scripts/cgcnn/ -> scripts/ -> the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)      # make the project root importable so `cgcnn_scratch` resolves below

from cgcnn_scratch.joint import (  # noqa: E402
    kappa_error_budget, kappa_quintile_breakdown)   # shared scoring helpers, same ones 31_train_joint.py uses


def parse_overrides():
    """One --flag per CONFIG key, so CONFIG stays the single source of truth."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in CONFIG.items():             # register one CLI flag per CONFIG entry
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default), default=None,
                            help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()                      # parse sys.argv against the registered flags
    cfg = dict(CONFIG)                              # start from the defaults
    for key, value in vars(args).items():           # walk each parsed argument
        if value is not None:                       # None means the flag was not passed on the CLI
            cfg[key] = value                        # otherwise the CLI value overrides the default
    return cfg                                      # effective configuration for this run


def main():
    cfg = parse_overrides()                         # resolve CONFIG + CLI overrides
    results_dir = os.path.join(PROJECT_ROOT, cfg["results_dir"])   # absolute path to the shared results directory
    tags = [t.strip() for t in cfg["tags"].split(",") if t.strip()]   # split the comma-separated tag list, dropping blanks

    print("=" * 78)
    print(f"  JOINT ENSEMBLE - {len(tags)} members: {', '.join(tags)}")
    print("=" * 78)

    # ---- load every member --------------------------------------------------
    members = []                                    # one DataFrame per member tag, indexed for alignment below
    for tag in tags:
        # NOTE: this filename is 31_train_joint.py's own per-member output
        # convention. 31 used to write this as predictions_joint_<tag>.csv;
        # the script-number rename changed it to predictions_31_<tag>.csv
        # (dropping "joint" in favour of the number, same as this script's
        # own combined output below) - it is a cross-script dependency, not
        # this script's own result, so this must track whatever 31 writes.
        path = os.path.join(results_dir, f"predictions_31_{tag}.csv")
        if not os.path.exists(path):                # fail fast with a clear, actionable message
            raise SystemExit(f"missing {path}\n"
                             f"Train it first, or check --tags.")
        df = pd.read_csv(path)                      # load this member's per-crystal predictions
        # Index by (crystal, split) so members can be aligned row for row. The
        # loaders shuffle, so raw row order differs between members and simply
        # stacking the columns would pair up different crystals.
        members.append(df.set_index(["material_id", "split"]))   # re-index for row-for-row alignment across members
        print(f"  {tag:<16} {len(df):5d} rows")

    # ---- check the members are actually comparable --------------------------
    base = members[0]                               # use the first member's index as the reference set of (crystal, split) pairs
    for tag, df in zip(tags[1:], members[1:]):       # compare every other member against the reference
        if not base.index.equals(df.index.reindex(base.index)[0]):
            # Reindexing rather than requiring identical order, but the SET of
            # (crystal, split) pairs must match exactly - a mismatch means the
            # members were trained on different splits and must not be combined.
            missing = base.index.difference(df.index)   # (crystal, split) pairs in base but absent from this member
            if len(missing):                          # a genuine mismatch, not just a reordering
                raise SystemExit(
                    f"{tag} is missing {len(missing)} (crystal, split) pairs "
                    f"present in {tags[0]}. The members were trained on "
                    f"different splits - combining them would score on data "
                    f"some members trained on.")
    # Ground truth must agree; if it does not, the members disagree about the
    # labels themselves and something is badly wrong upstream.
    for tag, df in zip(tags[1:], members[1:]):
        aligned = df.reindex(base.index)              # reorder this member's rows to match the reference index exactly
        for col in ("true_log10_K", "true_log10_G"):
            if not np.allclose(base[col].values, aligned[col].values, atol=1e-9):
                raise SystemExit(f"{tag} disagrees with {tags[0]} on {col}")   # labels must be bit-identical across members
    print(f"  all members share the same {len(base)} (crystal, split) pairs\n")

    # ---- average in log space, and keep the spread --------------------------
    stack_K = np.stack([m.reindex(base.index)["pred_log10_K"].values for m in members])
    # ^ one row per member, one column per (crystal, split), all aligned to the same order
    stack_G = np.stack([m.reindex(base.index)["pred_log10_G"].values for m in members])

    out = pd.DataFrame({
        "true_log10_K": base["true_log10_K"].values,
        "true_log10_G": base["true_log10_G"].values,
        # Mean of the logs = geometric mean of the moduli. See the docstring.
        "pred_log10_K": stack_K.mean(axis=0),        # average across members (axis 0), per crystal
        "pred_log10_G": stack_G.mean(axis=0),
        # Spread across members = the free per-crystal uncertainty estimate.
        # ddof=0 because these are the whole population of members we have, not
        # a sample drawn from a larger pool.
        "K_spread_log10": stack_K.std(axis=0, ddof=0),   # population standard deviation across members, per crystal
        "G_spread_log10": stack_G.std(axis=0, ddof=0),
    }, index=base.index).reset_index()               # turn the (crystal, split) index back into ordinary columns

    # ---- score, per split ---------------------------------------------------
    print(f"{'split':<8}{'n':>6}{'K':>8}{'G':>8}{'K/G':>9}{'corr':>8}"
          f"{'kappa':>9}{'pre':>8}{'anh':>8}")
    print("-" * 72)
    summary = {}                                     # per-split score dicts, collected for the final JSON dump
    for split in ("train", "val", "test"):
        sub = out[out.split == split]                # rows belonging to this split only
        if not len(sub):                             # skip a split with no rows (should not normally happen)
            continue
        b = kappa_error_budget(sub.true_log10_K.values, sub.true_log10_G.values,
                               sub.pred_log10_K.values, sub.pred_log10_G.values)
        # ^ shared helper: returns a dict of MAE/correlation/kappa-error-decomposition metrics for this split
        summary[split] = b                           # keep this split's metrics for later reporting/saving
        print(f"{split:<8}{len(sub):6d}{b['mae_log_K']:8.4f}{b['mae_log_G']:8.4f}"
              f"{b['mae_log_ratio']:9.4f}{b['residual_corr']:+8.3f}"
              f"{b['kappa_mae_total']:9.4f}{b['kappa_mae_prefactor']:8.4f}"
              f"{b['kappa_mae_anharmonic']:8.4f}")

    # ---- the comparison this whole exercise is about ------------------------
    t = summary[cfg["report_split"]]                 # pull out the split (usually "test") used for the headline comparison
    print()
    print("  vs the benchmarks on the same 1,648-crystal test set:")
    print(f"  {'':<26}{'this ens.':>11}{'sep. 3-ens':>12}{'1 joint+huber':>15}")
    print(f"  {'MAE log10(K)':<26}{t['mae_log_K']:11.4f}{0.0630:12.4f}{0.0827:15.4f}")
    print(f"  {'MAE log10(G)':<26}{t['mae_log_G']:11.4f}{0.0781:12.4f}{0.0950:15.4f}")
    print(f"  {'MAE log10(K/G)':<26}{t['mae_log_ratio']:11.4f}{0.0887:12.4f}{0.0863:15.4f}")
    print(f"  {'residual corr':<26}{t['residual_corr']:+11.3f}{0.297:+12.3f}{0.505:+15.3f}")
    print(f"  {'kappa prefactor':<26}{t['kappa_mae_prefactor']:11.4f}{0.1160:12.4f}{0.1416:15.4f}")
    print(f"  {'kappa anharmonic':<26}{t['kappa_mae_anharmonic']:11.4f}{0.0911:12.4f}{0.0891:15.4f}")
    print(f"  {'KAPPA TOTAL':<26}{t['kappa_mae_total']:11.4f}{0.1897:12.4f}{0.2050:15.4f}")
    verdict = ("BEATS the separate-model ensemble"    # simple threshold check against the known separate-model baseline
               if t["kappa_mae_total"] < 0.1897 else
               "still behind the separate-model ensemble")
    print(f"\n  -> {verdict}")

    # ---- stratified by true kappa: the metric a screen actually needs -------
    te_all = out[out.split == "test"]                # test-split rows only, for the stratified breakdown
    strat = kappa_quintile_breakdown(te_all.true_log10_K.values,
                                     te_all.true_log10_G.values,
                                     te_all.pred_log10_K.values,
                                     te_all.pred_log10_G.values)
    # ^ shared helper: buckets crystals into quintiles of true kappa and scores each bucket plus recall@10% separately
    print()
    print("  BY TRUE KAPPA QUINTILE  (Q1 = lowest = what a screen selects on)")
    print(f"    {'bin':<5}{'n':>6}{'kappa MAE':>11}{'MAE K':>9}{'MAE G':>9}{'median G':>11}")
    for qb in [f"Q{i}" for i in range(1, 6)]:        # Q1 (lowest kappa) through Q5 (highest)
        if qb in strat:                              # a quintile might be absent if the test set is very small
            d = strat[qb]
            print(f"    {qb:<5}{d['n']:6d}{d['kappa_mae']:11.4f}{d['mae_log_K']:9.4f}"
                  f"{d['mae_log_G']:9.4f}{d['median_true_G_GPa']:10.1f} GPa")
    print(f"    recall@10% = {100 * strat['recall_at_10pct']:.1f}%"
          f"   [separate 3-model baseline: 70.1%]")
    summary["test_by_kappa_quintile"] = strat        # fold the stratified breakdown into the same summary dict that gets saved

    # ---- the uncertainty the single model cannot give -----------------------
    te = out[out.split == "test"]                    # test-split rows again, for the spread summary below
    print(f"\n  per-crystal member spread on test (the free uncertainty estimate):")
    print(f"    log10(K) spread   median {te.K_spread_log10.median():.4f}"
          f"   90th pct {te.K_spread_log10.quantile(0.9):.4f}")
    print(f"    log10(G) spread   median {te.G_spread_log10.median():.4f}"
          f"   90th pct {te.G_spread_log10.quantile(0.9):.4f}")

    # ---- save ---------------------------------------------------------------
    # This script's OWN combined-output files (not a per-member file another
    # script depends on) - stamped with "32" so they can be told apart from
    # any other script's output landing in the same results/ directory.
    pred_path = os.path.join(results_dir, f"predictions_32_{cfg['out_tag']}.csv")
    out.to_csv(pred_path, index=False)               # write the combined per-crystal predictions, no row-index column
    sum_path = os.path.join(results_dir, f"summary_32_{cfg['out_tag']}.json")
    with open(sum_path, "w") as fh:
        json.dump({"tag": cfg["out_tag"], "members": tags,
                   "n_members": len(tags), "results": summary}, fh, indent=2)   # write the combined metrics as JSON, indented for readability
    print(f"\n  saved  {os.path.relpath(pred_path, PROJECT_ROOT)}")
    print(f"         {os.path.relpath(sum_path, PROJECT_ROOT)}")


if __name__ == "__main__":                           # only run main() when executed directly, not on import
    main()
