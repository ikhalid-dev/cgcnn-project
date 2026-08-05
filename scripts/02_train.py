#!/usr/bin/env python3
"""
STEP 2 - Train a CGCNN from scratch to predict an elastic modulus.
==================================================================

Run this once per target:

    python scripts/02_train.py --target K_VRH    # bulk modulus
    python scripts/02_train.py --target G_VRH    # shear modulus

WHAT "FROM SCRATCH" MEANS HERE
------------------------------
No pre-trained weights are loaded. Every parameter starts from PyTorch's
default random initialisation and is learned only from our ~278 labelled
crystals. This is deliberately a small-data experiment - see the honesty note
at the bottom of this docstring.

KEY DESIGN DECISIONS, AND WHY
-----------------------------
1. WE TRAIN ON log10(modulus), NOT THE MODULUS ITSELF.
   Moduli in our set span about 4 GPa to 400 GPa. With a plain MSE loss on raw
   values, a 40 GPa error on a stiff material contributes 100x more gradient
   than the same *relative* error on a soft one, so the model would learn to
   care only about hard materials. Taking log10 turns "within a factor of X"
   into "within a fixed distance", which is both easier to optimise and closer
   to what we actually care about. This is also exactly what matbench and the
   PINK paper do.

2. WE SPLIT BEFORE WE NORMALISE.
   The Normalizer's mean and standard deviation are computed on the TRAINING
   SPLIT ONLY. Computing them over the full dataset would leak information
   about the test set into training and quietly inflate our scores.

3. WE USE A SMALLER MODEL THAN THE PAPER.
   The paper trained on 10,987 crystals with atom_fea_len=64, h_fea_len=128.
   With 278 samples that many parameters will memorise the training set almost
   immediately. The defaults here are deliberately trimmed. Tune them with the
   command line flags if you want to see overfitting happen - it is instructive.

4. WE KEEP THE BEST MODEL BY VALIDATION MAE, NOT THE LAST ONE.
   Small datasets produce noisy validation curves. The final epoch is very
   often not the best epoch, so we checkpoint whenever validation improves and
   restore that checkpoint at the end.

HONEST EXPECTATION
------------------
278 training crystals is a very small dataset for a graph network with tens of
thousands of parameters. Expect the model to learn the broad trend (soft vs
stiff) but not to reach the paper's accuracy, which had ~40x more data. The
point of this stage is a correct, readable, end-to-end pipeline we can then
scale - not a competitive number.
"""

import argparse
import json
import os
import sys
import time
import warnings

# torch MUST be imported before numpy/pandas/pymatgen in this conda env:
# MKL loads its own OpenMP runtime first and the duplicate libiomp5 segfaults
# torch. Do not "tidy" these imports into alphabetical order.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.data import CIFData, Normalizer, collate_pool  # noqa: E402
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402


def build_id_prop(data_dir, target):
    """Write the id_prop.csv that CIFData reads, for the requested target.

    CIFData is generic - it just wants "filename, number". This function picks
    which column of labels.csv becomes that number, and takes the log10.
    Returns the number of rows written.
    """
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    # Guard against non-positive moduli, which would make log10 explode.
    # A modulus of zero or below is unphysical and indicates a bad DFT entry.
    bad = labels[labels[target] <= 0]
    if len(bad):
        print(f"  dropping {len(bad)} rows with non-positive {target}")
        labels = labels[labels[target] > 0]

    rows = pd.DataFrame({
        "cif_id": labels.material_id + ".cif",
        "target": np.log10(labels[target]),
    })
    rows.to_csv(os.path.join(data_dir, "cifs", "id_prop.csv"), index=False)
    return len(rows)


def split_indices(n_total, train_ratio, val_ratio, seed):
    """Split [0, n_total) into train / validation / test index lists.

    CIFData already shuffled its internal list with a fixed seed, but we
    shuffle again here with our own seed so that the *split* can be varied
    independently of the dataset ordering.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    n_train = int(round(train_ratio * n_total))
    n_val = int(round(val_ratio * n_total))

    return (indices[:n_train].tolist(),
            indices[n_train:n_train + n_val].tolist(),
            indices[n_train + n_val:].tolist())


def mean_absolute_error(prediction, target):
    """MAE in whatever units the inputs are in (we call it on de-normalised logs)."""
    return torch.mean(torch.abs(target - prediction)).item()


def run_epoch(loader, model, criterion, normalizer, optimizer=None):
    """Run one pass over `loader`. Trains if an optimizer is given, else evaluates.

    Returns (mean loss, MAE in log10 units).

    Note the two different spaces in play:
      * the LOSS is computed in normalised space, because that is what the
        network outputs and what gradients flow through;
      * the MAE is reported in de-normalised log10 space, because that is a
        number we can actually interpret and compare against the paper.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    losses, maes, n_seen = 0.0, 0.0, 0

    # torch.enable_grad/no_grad as a context manager keeps this one function
    # usable for both phases without duplicating the loop.
    with torch.enable_grad() if training else torch.no_grad():
        for inputs, target, _ in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            target_normed = normalizer.norm(target)

            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(output, target_normed)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = target.size(0)
            losses += loss.item() * batch_size
            maes += mean_absolute_error(normalizer.denorm(output.data), target) * batch_size
            n_seen += batch_size

    return losses / n_seen, maes / n_seen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], default="K_VRH",
                        help="which modulus to predict")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 penalty; small but non-zero helps on tiny datasets")
    # Model size - deliberately smaller than the paper's, see docstring point 3.
    parser.add_argument("--atom-fea-len", type=int, default=32)
    parser.add_argument("--h-fea-len", type=int, default=64)
    parser.add_argument("--n-conv", type=int, default=3)
    parser.add_argument("--n-h", type=int, default=1)
    # Splits
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed everything we can so a rerun reproduces the same numbers.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cif_dir = os.path.join(args.data_dir, "cifs")
    print(f"=== Training CGCNN from scratch for {args.target} ===\n")

    n_rows = build_id_prop(args.data_dir, args.target)
    print(f"Dataset: {n_rows} labelled crystals")

    # Building the graphs is the slow part; CIFData caches them after first use.
    dataset = CIFData(cif_dir)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset), args.train_ratio, args.val_ratio, args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test\n")

    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_pool, num_workers=0)
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **loader_kwargs)
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **loader_kwargs)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **loader_kwargs)

    # --- Normalizer fitted on the TRAINING TARGETS ONLY --------------------
    print("Fitting normalizer on training targets...")
    train_targets = torch.stack([dataset[i][1] for i in train_idx]).view(-1)
    normalizer = Normalizer(train_targets)
    print(f"  log10({args.target}): mean={normalizer.mean:.3f} std={normalizer.std:.3f}")

    # --- Model -------------------------------------------------------------
    # Feature widths are read off a real sample rather than hard-coded, so the
    # model automatically matches whatever atom_init.json and Gaussian settings
    # the dataset was built with.
    sample_structures, _, _ = dataset[0]
    orig_atom_fea_len = sample_structures[0].shape[-1]
    nbr_fea_len = sample_structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h, classification=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}  "
          f"({n_params / max(len(train_idx), 1):,.0f} per training sample)\n")

    criterion = nn.MSELoss()
    # Adam rather than the paper's SGD: with only a few hundred samples the
    # per-parameter adaptive step sizes converge far more reliably.
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # Decay the learning rate when validation MAE plateaus, so the model can
    # settle into a minimum instead of bouncing around it.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20)

    # --- Training loop -----------------------------------------------------
    history = []
    best_val_mae = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(args.epochs):
        train_loss, train_mae = run_epoch(train_loader, model, criterion,
                                          normalizer, optimizer)
        val_loss, val_mae = run_epoch(val_loader, model, criterion, normalizer)
        scheduler.step(val_mae)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae": train_mae, "val_mae": val_mae,
                        "lr": optimizer.param_groups[0]["lr"]})

        # Keep the best weights seen so far (see design decision 4).
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"train_mae {train_mae:.4f}  val_mae {val_mae:.4f}"
                  f"{'  <- best' if val_mae == best_val_mae else ''}")

    elapsed = time.time() - start
    print(f"\nTrained {args.epochs} epochs in {elapsed:.0f}s")

    # Restore the best checkpoint before touching the test set.
    model.load_state_dict(best_state)
    test_loss, test_mae = run_epoch(test_loader, model, criterion, normalizer)
    print(f"Best val MAE : {best_val_mae:.4f} log10(GPa)")
    print(f"Test MAE     : {test_mae:.4f} log10(GPa)")

    # --- Save everything the evaluation script will need -------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.target

    torch.save({
        "state_dict": best_state,
        "normalizer": normalizer.state_dict(),
        "args": vars(args),
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        # Saving the split makes the evaluation reproducible and stops us from
        # ever accidentally scoring on training data.
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        "feature_lens": {"orig_atom_fea_len": orig_atom_fea_len,
                         "nbr_fea_len": nbr_fea_len},
    }, os.path.join(args.out_dir, f"model_{tag}.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(args.out_dir, f"history_{tag}.csv"), index=False)

    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump({"target": tag, "n_total": len(dataset),
                   "n_train": len(train_idx), "n_val": len(val_idx),
                   "n_test": len(test_idx), "best_val_mae": best_val_mae,
                   "test_mae": test_mae, "epochs": args.epochs,
                   "n_params": n_params, "seconds": elapsed}, fh, indent=2)

    print(f"\nSaved to {args.out_dir}/:")
    print(f"  model_{tag}.pth      weights + normalizer + split")
    print(f"  history_{tag}.csv    per-epoch losses")
    print(f"  summary_{tag}.json   final metrics")


if __name__ == "__main__":
    main()
