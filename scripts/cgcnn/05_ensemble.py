#!/usr/bin/env python3
"""
STEP 5 - Combine several trained models into an ensemble, and score it.
=======================================================================

    python scripts/05_ensemble.py --target K_VRH \
        --tags K_VRH_full,K_VRH_s1,K_VRH_s2 --data-dir data_full

WHY AN ENSEMBLE HELPS
---------------------
Two networks trained on the same data with different random initialisations
land in different minima. They agree about the signal - the part of the modulus
that is genuinely predictable from structure - and disagree about the noise,
because their noise comes from their own initialisation and batch order rather
than from the data. Averaging their predictions therefore keeps the signal and
partially cancels the noise.

The gain is real but bounded. Ensembling typically removes 10-15% of the error;
it cannot remove the part of the error that every member gets wrong the same
way, which for this task means the label noise in the DFT reference values.

WE AVERAGE IN LOG SPACE, NOT IN GPa
-----------------------------------
The models emit log10(modulus) and are trained against a log-space loss, so
log space is where their errors are symmetric and roughly Gaussian. Averaging
two predictions of 10 and 1000 GPa gives 505 GPa in linear space but 100 GPa in
log space - and 100 is the defensible answer when the two members disagree by
two orders of magnitude. Averaging in linear space would let a single wild
over-prediction drag the ensemble with it.

EVERY MEMBER MUST SHARE THE SPLIT
---------------------------------
If member A trained on a crystal that member B held out, then scoring the
ensemble on B's test set is scoring it partly on A's training data, and the
result is meaningless. This script refuses to run unless all members report an
identical test split - hence 02_train.py's --split-seed, which is held fixed
while --seed varies.
"""

import argparse
import os
import sys
import warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.data import Normalizer, collate_pool, load_dataset_for  # noqa: E402
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

# 03_evaluate.py has a leading digit, so it cannot be imported by name. Reuse
# its plotting and metrics rather than duplicating them.
_eval = import_module("03_evaluate")


def load_member(tag, results_dir):
    """Load one trained model plus its normalizer and split."""
    path = os.path.join(results_dir, f"model_{tag}.pth")
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]

    model = CrystalGraphConvNet(
        ckpt["feature_lens"]["orig_atom_fea_len"],
        ckpt["feature_lens"]["nbr_fea_len"],
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normalizer = Normalizer(torch.zeros(1))
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt


def member_predictions(model, normalizer, dataset, split):
    """Predict log10(modulus) for every crystal, returned as {id: (true, pred)}."""
    out = {}
    for indices in split.values():
        loader = DataLoader(dataset, sampler=SubsetRandomSampler(indices),
                            batch_size=128, collate_fn=collate_pool)
        with torch.no_grad():
            for inputs, target, ids in loader:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                pred = normalizer.denorm(output.data).view(-1).numpy()
                true = target.view(-1).numpy()
                for item_id, t, p in zip(ids, true, pred):
                    out[item_id] = (float(t), float(p))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], required=True)
    parser.add_argument("--tags", required=True,
                        help="comma-separated checkpoint tags to combine")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data_full"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))
    parser.add_argument("--out-tag", default=None,
                        help="tag for the ensemble outputs. Defaults to <target>_ens")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    out_tag = args.out_tag or f"{args.target}_ens"
    print(f"=== Ensembling {len(tags)} models for {args.target} ===")
    print(f"    members: {', '.join(tags)}\n")

    dataset = load_dataset_for(args.data_dir, args.target)

    # --- Load every member, and check they agree about the split ------------
    members, reference_split = [], None
    for tag in tags:
        model, normalizer, ckpt = load_member(tag, args.results_dir)
        if reference_split is None:
            reference_split = ckpt["split"]
        elif sorted(ckpt["split"]["test"]) != sorted(reference_split["test"]):
            sys.exit(
                f"Member '{tag}' has a different test split from '{tags[0]}'.\n"
                f"Retrain it with the same --split-seed. Scoring an ensemble "
                f"whose members disagree about the test set would measure "
                f"performance partly on data some members trained on.")
        members.append((tag, model, normalizer, ckpt))

    # --- Run every member over the whole dataset ---------------------------
    per_member = []
    for tag, model, normalizer, ckpt in members:
        print(f"  running {tag}...")
        per_member.append(member_predictions(model, normalizer, dataset,
                                             reference_split))

    # --- Average in log space ----------------------------------------------
    # The split is stored as indices into the dataset; turn them back into ids.
    split_of = {}
    for split_name, indices in reference_split.items():
        for idx in indices:
            split_of[dataset.ids[idx]] = split_name

    rows = []
    for item_id, (true_log, _) in per_member[0].items():
        preds = [m[item_id][1] for m in per_member]
        mean_log = float(np.mean(preds))
        rows.append({
            "material_id": os.path.splitext(item_id)[0],
            "split": split_of.get(item_id, "unknown"),
            "true_log10": true_log,
            "pred_log10": mean_log,
            "true_GPa": float(10 ** true_log),
            "pred_GPa": float(10 ** mean_log),
            # Spread between members is a free uncertainty estimate: where the
            # members disagree, the ensemble is guessing.
            "member_std_log10": float(np.std(preds)),
        })

    df = pd.DataFrame(rows)
    df["abs_error_log10"] = (df.pred_log10 - df.true_log10).abs()
    df["abs_error_GPa"] = (df.pred_GPa - df.true_GPa).abs()

    # --- Score: every member alone, then the ensemble ----------------------
    print("\nPer-member test MAE (log10 GPa):")
    for (tag, _, _, _), preds in zip(members, per_member):
        test_ids = [i for i in preds if split_of.get(i) == "test"]
        mae = np.mean([abs(preds[i][1] - preds[i][0]) for i in test_ids])
        print(f"  {tag:20s} {mae:.4f}")

    scores = _eval.metrics(df)  # already carries the relative-error columns
    print(f"\nENSEMBLE ({len(tags)} members):")
    print(scores.round(4).to_string(index=False))

    # --- Write outputs ------------------------------------------------------
    df.sort_values(["split", "abs_error_log10"]).to_csv(
        os.path.join(args.results_dir, f"predictions_{out_tag}.csv"), index=False)
    scores.to_csv(os.path.join(args.results_dir, f"metrics_{out_tag}.csv"),
                  index=False)
    _eval.plot_parity(df, args.target,
                      os.path.join(args.results_dir, f"parity_{out_tag}.png"))
    _eval.plot_residuals(df, args.target,
                         os.path.join(args.results_dir, f"residuals_{out_tag}.png"))

    print(f"\nWrote results/predictions_{out_tag}.csv, metrics_{out_tag}.csv, "
          f"parity_{out_tag}.png, residuals_{out_tag}.png")


if __name__ == "__main__":
    main()
