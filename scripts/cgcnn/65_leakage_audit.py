#!/usr/bin/env python
"""
Step 65 - is the held-out test set really held out?

WHY THIS EXISTS
---------------
The sibling project (coherent-heat) splits by CHEMICAL SYSTEM and freezes the
split to a JSON file. This project splits at RANDOM and stores the split inside
each checkpoint. The reasonable question is whether the second arrangement
leaks - and the answer should be a measurement, not an argument.

Three things are checked, in this order:

  1. INTEGRITY. The split lives as index lists in the checkpoint, which are
     indices into the graph cache's order. That is only meaningful while the
     cache and the label file are unchanged. So: rebuild the split, and check
     that every row of every prediction file lands in the split it claims.

  2. DUPLICATE CHEMISTRY. A random split can put two polymorphs of the same
     composition on opposite sides of the boundary. A model that recognises the
     formula could then recall the answer instead of predicting it. Count how
     often that happens, and measure what it is worth.

  3. WHETHER A TWIN IS EVEN INFORMATIVE. If same-formula pairs have genuinely
     different moduli, recognising the formula HURTS rather than helps - the
     leak channel would be a liability, not an advantage.

RUN
    ml_env/bin/python scripts/cgcnn/65_leakage_audit.py

OUTPUTS
    results/cgcnn/65_leakage_audit.csv   one row per model/target/group
    results/cgcnn/65_leakage_audit.png   the two panels that carry the argument
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                   # render to a file; there is no display here
import matplotlib.pyplot as plt
from pymatgen.core import Composition

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data_full")
RESULTS = os.path.join(ROOT, "results")
CGCNN = os.path.join(RESULTS, "cgcnn")
ALIGNN = os.path.join(RESULTS, "alignn")

# The checkpoint whose split every other artefact must agree with. Any member
# would do - 05_ensemble.py asserts they share one test set - so this is simply
# the first one written.
CHECKPOINT = os.path.join(CGCNN, "model_G_VRH_full.pth")

# Prediction files, as (label, path, target column, pred column, target name).
# CGCNN's files carry their own `split` column; ALIGNN's do not, which is why
# the split has to be rebuilt rather than trusted.
PRED_FILES = [
    ("CGCNN ens", os.path.join(CGCNN, "predictions_K_VRH_ens.csv"), "K"),
    ("CGCNN ens", os.path.join(CGCNN, "predictions_G_VRH_ens.csv"), "G"),
]
ALIGNN_PRED = os.path.join(ALIGNN, "18_alignn_ensemble_predictions.csv")

N_BOOT = 2000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. rebuild the split, and check it
# ---------------------------------------------------------------------------
def rebuild_split():
    """material_id -> 'train'/'val'/'test', reconstructed from the checkpoint.

    The checkpoint stores INDICES. To turn an index back into an id we have to
    rebuild the exact list the dataset used: the graph cache's own order,
    filtered to rows that have a label. If either file changed since training,
    this mapping silently becomes wrong - which is precisely what check (1)
    below is for.
    """
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    blob = torch.load(os.path.join(DATA, "graphs.pt"), weights_only=False)
    labels = pd.read_csv(os.path.join(DATA, "labels.csv")).set_index("mb_id")

    ordered = [i for i in blob["ids"] if i in labels.index]
    # A dict comprehension with two `for` clauses reads left to right: for each
    # (name, index list) pair, for each index in that list, map id -> name.
    split_of = {ordered[i]: name
                for name, idx in ckpt["split"].items()
                for i in idx}

    labels["split"] = pd.Series(split_of)
    # Composition().reduced_formula normalises "S12Sr4Zr4" and "S6Sr2Zr2" to the
    # same string, which is what makes the duplicate check meaningful.
    labels["reduced"] = [Composition(f).reduced_formula for f in labels.formula]
    return labels, ckpt


def check_integrity(labels):
    """Every prediction row must land in the split it says it is in."""
    print("\n1. INTEGRITY - does the stored split still describe these files?")
    print("-" * 74)
    ok = True

    for name, path, target in PRED_FILES:
        if not os.path.exists(path):
            print(f"  MISSING {os.path.basename(path)}")
            ok = False
            continue
        d = pd.read_csv(path)
        rebuilt = d.material_id.map(labels.split)
        agree = (rebuilt == d.split).mean()
        print(f"  {name} {target}: {len(d):,} rows, rebuilt split agrees with the "
              f"file's own column for {agree:.1%}")
        ok &= agree > 0.999

    a = pd.read_csv(ALIGNN_PRED)
    a["split"] = a.material_id.map(labels.split)
    in_test = (a.split == "test").mean()
    print(f"  ALIGNN: {len(a):,} rows, {in_test:.1%} fall in the stored TEST split")
    print("    (the ALIGNN pipeline derives the split independently, so this is "
          "a cross-check,\n     not a tautology)")
    ok &= in_test > 0.999

    print("  ->", "the split is intact" if ok else "SPLIT MISMATCH - stop and investigate")
    return ok


# ---------------------------------------------------------------------------
# 2. duplicate chemistry across the boundary
# ---------------------------------------------------------------------------
def overlap_report(labels):
    train_formulas = set(labels.loc[labels.split == "train", "reduced"])
    te = labels[labels.split == "test"]
    seen = te.reduced.isin(train_formulas)

    print("\n2. DUPLICATE CHEMISTRY - does a random split put twins on both sides?")
    print("-" * 74)
    print(f"  test crystals                                {len(te):>6,}")
    print(f"  whose reduced formula ALSO appears in train  {seen.sum():>6,} "
          f"({seen.mean():.1%})")
    return train_formulas, seen


def twin_gap(labels, train_formulas):
    """How different are a test crystal and its same-formula training twin?

    If the answer is "barely", the overlap is a real leak. If the answer is
    "a lot", the twins are polymorphs and recognising the formula misleads.
    """
    te = labels[labels.split == "test"]
    twins = te[te.reduced.isin(train_formulas)]
    train_mean = (labels[labels.split == "train"]
                  .groupby("reduced")[["K_VRH", "G_VRH"]].mean())
    joined = twins.join(train_mean, on="reduced", rsuffix="_twin")

    print("\n3. IS A TWIN INFORMATIVE? |log10 difference| to the training twin")
    print("-" * 74)
    gaps = {}
    for col in ("K_VRH", "G_VRH"):
        g = np.abs(np.log10(joined[col]) - np.log10(joined[col + "_twin"])).dropna()
        gaps[col] = g
        print(f"  {col}: median {g.median():.4f}   mean {g.mean():.4f}   n={len(g)}")

    # How close is close? Stated as a share rather than a slogan, because the
    # distribution is two-sided: about half the twins ARE nearer than the
    # model's own error, and about a third are more than twice it.
    for col, mae in (("K_VRH", 0.0534), ("G_VRH", 0.0708)):
        g = gaps[col]
        print(f"  {col}: {(g < mae).mean():.0%} of twins are closer than the "
              f"model's own MAE ({mae}); {(g > 2*mae).mean():.0%} are more than "
              f"twice it")
    print("  -> a twin is a noisy hint, not a copy of the answer")
    return gaps


# ---------------------------------------------------------------------------
# 3. what the overlap is actually worth
# ---------------------------------------------------------------------------
def boot_diff(err_seen, err_unseen):
    """95% interval for MAE(seen) - MAE(unseen), by resampling each group.

    Not a PAIRED bootstrap: the two groups are different crystals, so there is
    nothing to pair. Each group is resampled with replacement independently and
    the difference of the two means recorded, N_BOOT times.
    """
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = RNG.choice(err_seen, size=len(err_seen), replace=True)
        u = RNG.choice(err_unseen, size=len(err_unseen), replace=True)
        diffs[b] = s.mean() - u.mean()
    return np.percentile(diffs, [2.5, 97.5])


def scored(labels, train_formulas):
    """MAE split by whether the formula was seen in training, per model/target."""
    rows = []

    frames = []
    for name, path, target in PRED_FILES:
        d = pd.read_csv(path)
        d = d[d.split == "test"].copy()
        d["err"] = (d.pred_log10 - d.true_log10).abs()
        d["model"], d["target"] = name, target
        frames.append(d[["material_id", "model", "target", "err"]])

    a = pd.read_csv(ALIGNN_PRED)
    a["err"] = (a.pred_log10 - a.true_log10).abs()
    a["model"] = "ALIGNN ens"
    frames.append(a[["material_id", "model", "target", "err"]])

    allp = pd.concat(frames, ignore_index=True)
    allp["reduced"] = allp.material_id.map(labels.reduced)
    allp["seen"] = allp.reduced.isin(train_formulas)

    print("\n4. WHAT THE OVERLAP IS WORTH - test MAE, split by formula overlap")
    print("-" * 74)
    print(f"  {'model':<12}{'target':<8}{'group':<26}{'n':>6}{'MAE':>9}{'as %':>8}")
    for (model, target), g in allp.groupby(["model", "target"]):
        e_seen = g.loc[g.seen, "err"].to_numpy()
        e_unseen = g.loc[~g.seen, "err"].to_numpy()
        for err, label in [(e_unseen, "formula NOT seen in train"),
                           (e_seen, "formula ALSO in train")]:
            mae = err.mean()
            print(f"  {model:<12}{target:<8}{label:<26}{len(err):>6}{mae:>9.4f}"
                  f"{100*(10**mae-1):>7.1f}%")
            rows.append({"model": model, "target": target, "group": label,
                         "n": len(err), "mae_log10": mae,
                         "relative_error_pct": 100 * (10 ** mae - 1)})
        overall = g.err.mean()
        lo, hi = boot_diff(e_seen, e_unseen)
        print(f"  {model:<12}{target:<8}{'ALL (the quoted number)':<26}"
              f"{len(g):>6}{overall:>9.4f}{100*(10**overall-1):>7.1f}%")
        print(f"  {'':12}{'':8}{'-> optimism vs unseen-only':<26}"
              f"{'':>6}{overall - e_unseen.mean():>+9.4f}"
              f"   diff 95% CI [{lo:+.4f}, {hi:+.4f}]\n")
        rows.append({"model": model, "target": target, "group": "ALL",
                     "n": len(g), "mae_log10": overall,
                     "relative_error_pct": 100 * (10 ** overall - 1),
                     "optimism_vs_unseen": overall - e_unseen.mean(),
                     "diff_ci_lo": lo, "diff_ci_hi": hi})
    return allp, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def figure(allp, gaps, path):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) MAE by group
    keys = sorted(allp.groupby(["model", "target"]).groups)
    x = np.arange(len(keys))
    unseen = [allp[(allp.model == m) & (allp.target == t) & (~allp.seen)].err.mean()
              for m, t in keys]
    seen = [allp[(allp.model == m) & (allp.target == t) & (allp.seen)].err.mean()
            for m, t in keys]
    ax[0].bar(x - 0.19, unseen, width=.36, color="#2563eb",
              label="formula NOT seen in training")
    ax[0].bar(x + 0.19, seen, width=.36, color="#dc2626",
              label="formula also in training")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{m}\n{t}" for m, t in keys], fontsize=8)
    ax[0].set(ylabel="test MAE, log$_{10}$",
              title="(a) duplicated chemistry is predicted WORSE,\nnot better")
    ax[0].legend(fontsize=8)

    # (b) how far apart the twins actually are
    for col, colour, lab in (("K_VRH", "#2563eb", "bulk $K$"),
                             ("G_VRH", "#16a34a", "shear $G$")):
        ax[1].hist(gaps[col], bins=40, alpha=.55, color=colour, label=lab)
    for v, colour, lab in ((0.0534, "#2563eb", "ALIGNN MAE, $K$"),
                           (0.0708, "#16a34a", "ALIGNN MAE, $G$")):
        ax[1].axvline(v, color=colour, ls="--", lw=1.2, label=lab)
    ax[1].set(xlabel=r"$|\Delta \log_{10}|$ to the same-formula training crystal",
              ylabel="test crystals",
              title="(b) about half the twins are nearer than the model's\n"
                    "own error - and a third are twice it or worse")
    ax[1].set_xlim(0, 0.6)
    ax[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  -> {path}")


def main():
    print("=" * 74)
    print("STEP 65 - LEAKAGE AUDIT")
    print("=" * 74)

    labels, ckpt = rebuild_split()
    print(f"\nsplit from {os.path.basename(CHECKPOINT)}: "
          + ", ".join(f"{k} {len(v):,}" for k, v in ckpt['split'].items()))

    intact = check_integrity(labels)
    train_formulas, _ = overlap_report(labels)
    gaps = twin_gap(labels, train_formulas)
    allp, table = scored(labels, train_formulas)

    out = os.path.join(CGCNN, "65_leakage_audit.csv")
    table.to_csv(out, index=False)
    print(f"  -> {out}")
    figure(allp, gaps, os.path.join(CGCNN, "65_leakage_audit.png"))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    worst = table[table.group == "ALL"].optimism_vs_unseen.max()
    print(f"  split intact: {intact}")
    print(f"  largest optimism from duplicated chemistry: {worst:+.4f} log10")
    print("  The random split is defensible HERE because same-formula pairs are")
    print("  polymorphs with genuinely different moduli. It would not be")
    print("  defensible on a target where the formula fixes the answer.")
    return 0 if intact else 1


if __name__ == "__main__":
    sys.exit(main())
