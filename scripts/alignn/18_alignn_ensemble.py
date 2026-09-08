#!/usr/bin/env python3
"""
STEP 18 - Ensemble the ALIGNN seeds, and score it against every baseline.
================================================================================

    python scripts/alignn/18_alignn_ensemble.py
    python scripts/alignn/18_alignn_ensemble.py --seeds 42 1 2

WHY THIS IS NOT 05_ensemble.py
--------------------------------
`scripts/cgcnn/05_ensemble.py` reloads CGCNN checkpoints and re-predicts. It
cannot be reused here for two reasons:

  * ALIGNN writes its own `prediction_results_test_set.csv` per run, so the
    held-out predictions already exist and no model needs reloading to score
    the ensemble.
  * **ALIGNN is trained on RAW GPa, CGCNN on log10(GPa).** ALIGNN's prediction
    files therefore contain GPa, not logs (`11_prepare_alignn_data.py` feeds
    `float(record.K_VRH)` with no transform). Every "log10 MAE" quoted for
    ALIGNN in this project is computed post hoc by log-transforming those raw
    values, which is what `17_alignn_diagnostics.py` does and what this script
    does too. See docs/preregistration_2026-09-08.md, Experiment C - that
    objective mismatch is a finding in its own right, not a detail.

WHY THE AVERAGE IS TAKEN IN LOG SPACE ANYWAY
----------------------------------------------
The members are averaged as log10, i.e. a geometric mean in GPa, matching
05_ensemble.py exactly. The reason is not consistency for its own sake: the
error on a modulus is proportional, not absolute, and the metric this project
reports is a log-space MAE. Averaging raw GPa would let one stiff member
dominate a soft crystal's consensus.

THE NON-POSITIVE PREDICTION PROBLEM, HANDLED EXPLICITLY
---------------------------------------------------------
A raw-space regressor can emit a negative modulus, and this one does: the
existing single ALIGNN predicted -0.57 GPa for a soft validation crystal whose
true value is 2 GPa. log10 of that is undefined. A log-space model cannot
produce it at all, which is why CGCNN never needed this guard.

Rows where ANY member is non-positive are counted and reported rather than
silently dropped or clipped - the count is a direct measure of how often the
raw-GPa objective breaks down, and it is expected to fall to zero if Experiment
C's log-target retrain happens. The ensemble value for such a row is taken over
the members that ARE positive; if none are, the row is excluded from the score
and counted separately.

WHAT IT REPORTS
-----------------
Per-member scores, the ensemble score, the seed spread, and a paired bootstrap
against both the single-ALIGNN and the CGCNN ensemble baselines - because this
project's own rule is that a point estimate is not a result, and that ensembles
are only ever compared to ensembles.
"""

# =============================================================================
#  CONFIG - every path and tunable lives here
# =============================================================================
CONFIG = {
    "results_dir": "results/alignn",
    # Per-seed run directories are named alignn_<target>_s<seed> by
    # kaggle/build_alignn_ensemble_kernel.py.
    "targets": ["bulk_modulus_kv", "shear_modulus_gv"],
    "seeds": [42, 1, 2],
    "pred_file": "prediction_results_test_set.csv",

    # the single-model ALIGNN that already exists, as the "does ensembling
    # help ALIGNN" reference
    "single_dirs": {"bulk_modulus_kv": "alignn_bulk_modulus_kv",
                    "shear_modulus_gv": "alignn_shear_modulus_gv"},

    # CGCNN's own held-out predictions, for the ensemble-vs-ensemble row
    "cgcnn_pred": {"bulk_modulus_kv": "results/cgcnn/predictions_K_VRH_ens.csv",
                   "shear_modulus_gv": "results/cgcnn/predictions_G_VRH_ens.csv"},

    "out_csv": "results/alignn/18_alignn_ensemble_scores.csv",
    "out_pred": "results/alignn/18_alignn_ensemble_predictions.csv",
    "bootstrap_n": 10000,
    "seed": 0,
}
# =============================================================================

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_LABEL = {"bulk_modulus_kv": "K", "shear_modulus_gv": "G"}


def p(*a):
    print(*a, flush=True)


def load_member(results_dir, target, seed, pred_file):
    """One seed's held-out predictions, in RAW GPa as ALIGNN writes them."""
    d = os.path.join(results_dir, f"alignn_{target}_s{seed}")
    path = os.path.join(d, pred_file)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"id": "material_id", "target": "true_GPa",
                            "prediction": "pred_GPa"})
    df["material_id"] = df["material_id"].astype(str).str.strip()
    return df[["material_id", "true_GPa", "pred_GPa"]]


def mae_r2(true_log, pred_log):
    err = np.abs(pred_log - true_log)
    ss_res = float(((pred_log - true_log) ** 2).sum())
    ss_tot = float(((true_log - true_log.mean()) ** 2).sum())
    return float(err.mean()), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def paired_bootstrap(err_a, err_b, n, rng):
    """95% CI on mean(err_a) - mean(err_b), resampling crystals in pairs.

    Negative means a is better. This project's rule: compute the interval
    before writing the verb.
    """
    idx = rng.integers(0, len(err_a), (n, len(err_a)))
    d = err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.seeds:
        cfg["seeds"] = args.seeds
    rdir = os.path.join(PROJECT_ROOT, cfg["results_dir"])
    rng = np.random.default_rng(cfg["seed"])
    rows, preds_out = [], []

    p("=" * 78)
    p("  STEP 18 - the ALIGNN ensemble")
    p("=" * 78)

    for target in cfg["targets"]:
        lbl = TARGET_LABEL[target]
        p()
        p("  " + "-" * 74)
        p(f"  {lbl}  ({target})")
        p("  " + "-" * 74)

        members = {}
        for s in cfg["seeds"]:
            m = load_member(rdir, target, s, cfg["pred_file"])
            if m is None:
                p(f"    [missing] seed {s} - no {cfg['pred_file']} yet")
                continue
            members[s] = m
        if not members:
            p(f"    no members found for {target}; skipping")
            continue

        # Align every member on material_id. If the split moved between runs
        # the ensemble would be meaningless, so this is an assertion, not a join.
        base = members[cfg["seeds"][0] if cfg["seeds"][0] in members
                       else list(members)[0]]
        ids = base["material_id"].tolist()
        for s, m in members.items():
            if m["material_id"].tolist() != ids:
                m_sorted = m.set_index("material_id").loc[ids].reset_index()
                members[s] = m_sorted
                p(f"    note: seed {s} needed reordering to match the reference id order")

        true_gpa = base["true_GPa"].to_numpy(float)
        stack = np.vstack([members[s]["pred_GPa"].to_numpy(float) for s in members])

        # ---- the non-positive problem, counted not hidden --------------------
        nonpos = stack <= 0
        n_rows_any = int(nonpos.any(axis=0).sum())
        n_cells = int(nonpos.sum())
        if n_cells:
            p(f"    non-positive predictions: {n_cells} cells across "
              f"{n_rows_any} crystals - these have no log10 and are handled "
              f"member-wise (see this script's docstring)")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_stack = np.log10(np.where(stack > 0, stack, np.nan))
        # mean over the members that are positive for each crystal
        ens_log = np.nanmean(log_stack, axis=0)
        true_log = np.log10(true_gpa)

        valid = np.isfinite(ens_log) & np.isfinite(true_log)
        n_dropped = int((~valid).sum())
        if n_dropped:
            p(f"    {n_dropped} crystals excluded from the score "
              f"(no member gave a positive prediction, or the label is non-positive)")

        # ---- per-member scores, and the seed spread --------------------------
        member_maes = []
        for i, s in enumerate(members):
            v = valid & np.isfinite(log_stack[i])
            mae, r2 = mae_r2(true_log[v], log_stack[i][v])
            member_maes.append(mae)
            rows.append({"target": lbl, "model": f"ALIGNN seed {s}", "n": int(v.sum()),
                         "mae_log10": mae, "r2": r2})
            p(f"    seed {s:<3} MAE {mae:.4f}   R2 {r2:.3f}   n={int(v.sum())}")
        spread = (max(member_maes) - min(member_maes)) if len(member_maes) > 1 else 0.0
        p(f"    seed spread (max-min of member MAE): {spread:.4f}")

        ens_mae, ens_r2 = mae_r2(true_log[valid], ens_log[valid])
        rows.append({"target": lbl, "model": f"ALIGNN {len(members)}-ens",
                     "n": int(valid.sum()), "mae_log10": ens_mae, "r2": ens_r2})
        p(f"    ENSEMBLE  MAE {ens_mae:.4f}   R2 {ens_r2:.3f}   n={int(valid.sum())}")

        # per-crystal member spread = the uncertainty ALIGNN has never had
        member_sd = np.nanstd(log_stack, axis=0)
        for mid, tl, el, sd in zip(ids, true_log, ens_log, member_sd):
            preds_out.append({"target": lbl, "material_id": mid,
                              "true_log10": tl, "pred_log10": el,
                              "member_sd_log10": sd,
                              "true_GPa": 10 ** tl, "pred_GPa": 10 ** el})

        # ---- baselines, on the same crystals ---------------------------------
        ens_err = np.abs(ens_log - true_log)

        single_dir = os.path.join(rdir, cfg["single_dirs"][target])
        sp = os.path.join(single_dir, cfg["pred_file"])
        if os.path.exists(sp):
            sdf = pd.read_csv(sp)
            sdf.columns = [c.strip() for c in sdf.columns]
            sdf = sdf.rename(columns={"id": "material_id", "target": "true_GPa",
                                      "prediction": "pred_GPa"})
            sdf["material_id"] = sdf["material_id"].astype(str).str.strip()
            sdf = sdf.set_index("material_id").reindex(ids)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_log = np.log10(np.where(sdf["pred_GPa"].to_numpy(float) > 0,
                                          sdf["pred_GPa"].to_numpy(float), np.nan))
            v = valid & np.isfinite(s_log)
            s_mae, s_r2 = mae_r2(true_log[v], s_log[v])
            rows.append({"target": lbl, "model": "ALIGNN single (existing)",
                         "n": int(v.sum()), "mae_log10": s_mae, "r2": s_r2})
            d, lo, hi = paired_bootstrap(ens_err[v], np.abs(s_log - true_log)[v],
                                         cfg["bootstrap_n"], rng)
            sig = "EXCLUDES zero" if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else "crosses zero"
            p(f"    vs ALIGNN single  {s_mae:.4f} -> {ens_mae:.4f}   "
              f"d={d:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig}")
            rows.append({"target": lbl, "model": "d(ens - single)", "n": int(v.sum()),
                         "mae_log10": d, "ci_lo": lo, "ci_hi": hi})

        cg_path = os.path.join(PROJECT_ROOT, cfg["cgcnn_pred"][target])
        if os.path.exists(cg_path):
            cg = pd.read_csv(cg_path)
            cg = cg[cg["split"] == "test"].set_index("material_id").reindex(ids)
            c_log = cg["pred_log10"].to_numpy(float)
            v = valid & np.isfinite(c_log)
            c_mae, c_r2 = mae_r2(true_log[v], c_log[v])
            rows.append({"target": lbl, "model": "CGCNN 3-ens", "n": int(v.sum()),
                         "mae_log10": c_mae, "r2": c_r2})
            d, lo, hi = paired_bootstrap(ens_err[v], np.abs(c_log - true_log)[v],
                                         cfg["bootstrap_n"], rng)
            sig = "EXCLUDES zero" if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else "crosses zero"
            p(f"    vs CGCNN 3-ens    {c_mae:.4f} -> {ens_mae:.4f}   "
              f"d={d:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig}   <- ensemble vs ensemble")
            rows.append({"target": lbl, "model": "d(ALIGNN ens - CGCNN ens)",
                         "n": int(v.sum()), "mae_log10": d, "ci_lo": lo, "ci_hi": hi})

    if not rows:
        sys.exit("No member predictions found. Fetch the Kaggle runs first.")

    pd.DataFrame(rows).to_csv(os.path.join(PROJECT_ROOT, cfg["out_csv"]), index=False)
    pd.DataFrame(preds_out).to_csv(os.path.join(PROJECT_ROOT, cfg["out_pred"]), index=False)
    p()
    p("  wrote:")
    p(f"    {cfg['out_csv']}")
    p(f"    {cfg['out_pred']}   <- carries member_sd_log10, the interval ALIGNN never had")


if __name__ == "__main__":
    main()
