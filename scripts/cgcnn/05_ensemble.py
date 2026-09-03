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

import argparse  # CLI argument parsing (--target, --tags, --data-dir, etc.)
import os  # path joining and filesystem path handling
import sys  # sys.path manipulation and sys.exit() for fatal errors
import warnings  # used below to silence noisy third-party warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch  # checkpoint loading, tensors, no_grad inference
from torch.utils.data import DataLoader  # batches dataset items for inference
from torch.utils.data.sampler import SubsetRandomSampler  # restricts a DataLoader to a specific list of indices (one split at a time)

import numpy as np  # mean/std across ensemble members
import pandas as pd  # builds and writes the per-crystal predictions table

warnings.filterwarnings("ignore")  # suppresses warning spam during model loading/inference

# walk up three directories from this file (scripts/cgcnn/05_...py -> scripts/cgcnn -> scripts -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)  # so `import cgcnn_scratch` resolves regardless of the current working directory

from cgcnn_scratch.data import Normalizer, collate_pool, load_dataset_for  # noqa: E402
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make this script's own directory importable too
from importlib import import_module  # noqa: E402  # imports a module whose filename starts with a digit (not a valid identifier for a normal `import`)

# 03_evaluate.py has a leading digit, so it cannot be imported by name. Reuse
# its plotting and metrics rather than duplicating them.
_eval = import_module("03_evaluate")


def load_member(tag, results_dir):
    """Load one trained model plus its normalizer and split."""
    path = os.path.join(results_dir, f"model_{tag}.pth")  # checkpoint path for this ensemble member
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path}")  # fatal - can't ensemble a model that was never trained

    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # load the checkpoint dict onto CPU regardless of what device trained it
    targs = ckpt["args"]  # the CLI args this member was originally trained with (architecture, etc.)

    model = CrystalGraphConvNet(
        ckpt["feature_lens"]["orig_atom_fea_len"],
        ckpt["feature_lens"]["nbr_fea_len"],
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)  # rebuild the exact same architecture this checkpoint was trained with
    model.load_state_dict(ckpt["state_dict"])  # load the trained weights into that architecture
    model.eval()  # inference mode - disables dropout/batchnorm training behavior

    normalizer = Normalizer(torch.zeros(1))  # placeholder normalizer, immediately overwritten by load_state_dict below
    normalizer.load_state_dict(ckpt["normalizer"])  # restore this member's own mean/std used to denormalize predictions
    return model, normalizer, ckpt


def member_predictions(model, normalizer, dataset, split):
    """Predict log10(modulus) for every crystal, returned as {id: (true, pred)}."""
    out = {}  # accumulates {crystal_id: (true_value, predicted_value)} across every split
    for indices in split.values():  # iterate over train/val/test index lists
        loader = DataLoader(dataset, sampler=SubsetRandomSampler(indices),
                            batch_size=128, collate_fn=collate_pool)  # only yields items at these indices, in batches of 128
        with torch.no_grad():  # no gradient tracking needed for inference - saves memory and time
            for inputs, target, ids in loader:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs  # unpack the four graph tensors collate_pool produces
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # forward pass -> normalized prediction
                pred = normalizer.denorm(output.data).view(-1).numpy()  # undo normalization, flatten to 1D, convert to a numpy array
                true = target.view(-1).numpy()  # ground-truth values for this batch, flattened
                for item_id, t, p in zip(ids, true, pred):  # zip walks the three parallel per-crystal sequences together
                    out[item_id] = (float(t), float(p))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # preserves the docstring's own line breaks when printed by --help
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], required=True)  # which modulus this ensemble is for
    parser.add_argument("--tags", required=True,
                        help="comma-separated checkpoint tags to combine")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data_full"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))
    parser.add_argument("--out-tag", default=None,
                        help="tag for the ensemble outputs. Defaults to <target>_ens")
    args = parser.parse_args()  # parse sys.argv into the `args` namespace

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]  # split the comma-separated tag list, dropping empty entries and whitespace
    out_tag = args.out_tag or f"{args.target}_ens"  # fall back to "<target>_ens" if --out-tag was not given
    print(f"=== Ensembling {len(tags)} models for {args.target} ===")
    print(f"    members: {', '.join(tags)}\n")

    dataset = load_dataset_for(args.data_dir, args.target)  # the full graph dataset for this target property

    # --- Load every member, and check they agree about the split ------------
    members, reference_split = [], None  # reference_split is set from the FIRST member, every later member is checked against it
    for tag in tags:
        model, normalizer, ckpt = load_member(tag, args.results_dir)
        if reference_split is None:
            reference_split = ckpt["split"]  # first member defines the split every other member must match
        elif sorted(ckpt["split"]["test"]) != sorted(reference_split["test"]):  # sort both so index ORDER differences don't cause a false mismatch
            sys.exit(
                f"Member '{tag}' has a different test split from '{tags[0]}'.\n"
                f"Retrain it with the same --split-seed. Scoring an ensemble "
                f"whose members disagree about the test set would measure "
                f"performance partly on data some members trained on.")
        members.append((tag, model, normalizer, ckpt))

    # --- Run every member over the whole dataset ---------------------------
    per_member = []  # one {id: (true, pred)} dict per ensemble member, in the same order as `members`
    for tag, model, normalizer, ckpt in members:
        print(f"  running {tag}...")
        per_member.append(member_predictions(model, normalizer, dataset,
                                             reference_split))

    # --- Average in log space ----------------------------------------------
    # The split is stored as indices into the dataset; turn them back into ids.
    split_of = {}  # maps crystal id -> "train"/"val"/"test"
    for split_name, indices in reference_split.items():
        for idx in indices:
            split_of[dataset.ids[idx]] = split_name  # dataset.ids[idx] turns a numeric index back into its crystal id string

    rows = []
    for item_id, (true_log, _) in per_member[0].items():  # iterate over crystal ids using the first member as the reference set of keys
        preds = [m[item_id][1] for m in per_member]  # this crystal's predicted log10 value from every member
        mean_log = float(np.mean(preds))  # ensemble prediction = arithmetic mean IN LOG SPACE (equivalent to a geometric mean in GPa)
        rows.append({
            "material_id": os.path.splitext(item_id)[0],  # strip the file extension (e.g. ".cif") off the raw item id
            "split": split_of.get(item_id, "unknown"),
            "true_log10": true_log,
            "pred_log10": mean_log,
            "true_GPa": float(10 ** true_log),  # convert back out of log space for a human-readable column
            "pred_GPa": float(10 ** mean_log),
            # Spread between members is a free uncertainty estimate: where the
            # members disagree, the ensemble is guessing.
            "member_std_log10": float(np.std(preds)),  # how much the members disagree on this one crystal
        })

    df = pd.DataFrame(rows)  # one row per crystal, columns as built above
    df["abs_error_log10"] = (df.pred_log10 - df.true_log10).abs()  # absolute ensemble error in log space
    df["abs_error_GPa"] = (df.pred_GPa - df.true_GPa).abs()  # absolute ensemble error in linear GPa

    # --- Score: every member alone, then the ensemble ----------------------
    print("\nPer-member test MAE (log10 GPa):")
    for (tag, _, _, _), preds in zip(members, per_member):  # zip pairs each member's metadata with its own prediction dict
        test_ids = [i for i in preds if split_of.get(i) == "test"]  # restrict scoring to the held-out test split only
        mae = np.mean([abs(preds[i][1] - preds[i][0]) for i in test_ids])  # mean absolute error for this member alone
        print(f"  {tag:20s} {mae:.4f}")

    scores = _eval.metrics(df)  # already carries the relative-error columns
    print(f"\nENSEMBLE ({len(tags)} members):")
    print(scores.round(4).to_string(index=False))

    # --- Write outputs ------------------------------------------------------
    df.sort_values(["split", "abs_error_log10"]).to_csv(
        os.path.join(args.results_dir, f"predictions_{out_tag}.csv"), index=False)  # sorted so the worst-error rows within each split are easy to find
    scores.to_csv(os.path.join(args.results_dir, f"metrics_{out_tag}.csv"),
                  index=False)
    _eval.plot_parity(df, args.target,
                      os.path.join(args.results_dir, f"parity_{out_tag}.png"))  # reuses 03_evaluate.py's own parity-plot function
    _eval.plot_residuals(df, args.target,
                         os.path.join(args.results_dir, f"residuals_{out_tag}.png"))  # reuses 03_evaluate.py's own residuals-plot function

    print(f"\nWrote results/predictions_{out_tag}.csv, metrics_{out_tag}.csv, "
          f"parity_{out_tag}.png, residuals_{out_tag}.png")


if __name__ == "__main__":  # only run main() when executed as a script, not when imported as a module
    main()
