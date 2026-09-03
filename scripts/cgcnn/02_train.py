#!/usr/bin/env python3
"""
STEP 2 - Train a CGCNN from scratch to predict an elastic modulus.
==================================================================

Run this once per target:

    python scripts/02_train.py --target K_VRH    # bulk modulus
    python scripts/02_train.py --target G_VRH    # shear modulus

THE DATASET
-----------
`--data-dir` points at a directory prepared by 01b_prepare_full_dataset.py:
`data_full/` holds all 10,987 crystals of matbench's elastic benchmark, with
their graphs pre-built into graphs.pt. Nothing here parses CIFs - the graphs
were converted once and pickled, because the periodic neighbour search is the
slowest step in the pipeline.

WHAT "FROM SCRATCH" MEANS HERE
------------------------------
No pre-trained weights are loaded. Every parameter starts from PyTorch's
default random initialisation and is learned only from the labelled crystals in
`--data-dir`.

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

3. MODEL SIZE MATCHES THE PAPER.
   atom_fea_len=64, h_fea_len=128, 3 convolutions - the same widths the paper
   used on this same 10,987-crystal dataset, giving ~81k parameters.

4. WE KEEP THE BEST MODEL BY VALIDATION MAE, NOT THE LAST ONE.
   Small datasets produce noisy validation curves. The final epoch is very
   often not the best epoch, so we checkpoint whenever validation improves and
   restore that checkpoint at the end.

WHAT TO EXPECT
--------------
This is the paper's own training set, so the paper's numbers (~0.07 log10 GPa
test MAE) are the target rather than an aspiration. A single model lands at
about 0.070 for bulk and 0.084 for shear; three of them ensembled by
05_ensemble.py reach 0.063 and 0.078.
"""

import argparse  # stdlib CLI argument parser, defines the --flag options below
import json  # stdlib JSON encode/decode, used to write summary_<tag>.json
import os  # stdlib path/filesystem helpers (join, makedirs, exists, remove)
import sys  # stdlib interpreter access, used to extend the module import path
import time  # stdlib wall-clock timer, used to measure training duration
import warnings  # stdlib warning-filter control

# torch MUST be imported before numpy/pandas/pymatgen in this conda env:
# MKL loads its own OpenMP runtime first and the duplicate libiomp5 segfaults
# torch. Do not "tidy" these imports into alphabetical order.
import torch  # PyTorch core: tensors, autograd, device management
import torch.nn as nn  # neural-network layers and loss functions (nn.MSELoss)
import torch.optim as optim  # optimizers (Adam) and LR schedulers
from torch.utils.data import DataLoader  # batches a dataset via a sampler + collate function
from torch.utils.data.sampler import SubsetRandomSampler  # draws shuffled indices from one fixed subset

import numpy as np  # array ops, used here for the seeded RNG that builds the split
import pandas as pd  # DataFrame, used to write history_<tag>.csv

warnings.filterwarnings("ignore")  # silence deprecation/runtime warnings so they don't clutter the training log

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ absolute path of this file, walked up three directories: scripts/cgcnn/02_train.py -> project root
sys.path.insert(0, PROJECT_ROOT)  # prepend project root to the import search path so `cgcnn_scratch` resolves

# Normalizer z-scores targets; collate_pool batches graph samples into one padded tensor;
# load_dataset_for reads the cached graphs.pt/labels.csv pair for one target column.
from cgcnn_scratch.data import Normalizer, collate_pool, load_dataset_for  # noqa: E402
# the CGCNN model class itself (graph-conv layers + pooling + fully-connected head)
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402


def split_indices(n_total, train_ratio, val_ratio, seed):
    """Split [0, n_total) into train / validation / test index lists.

    The dataset order is fixed by the graph cache, so this shuffle is what
    makes the split random - and seeding it separately is what lets ensemble
    members share one split while differing in initialisation.
    """
    rng = np.random.RandomState(seed)  # seeded PRNG object, independent of numpy's global RNG state
    indices = rng.permutation(n_total)  # a random shuffle of the integers 0..n_total-1

    n_train = int(round(train_ratio * n_total))  # number of rows assigned to the training split
    n_val = int(round(val_ratio * n_total))  # number of rows assigned to the validation split

    # first n_train shuffled indices -> train; next n_val -> val; everything left over -> test
    return (indices[:n_train].tolist(),
            indices[n_train:n_train + n_val].tolist(),
            indices[n_train + n_val:].tolist())


def mean_absolute_error(prediction, target):
    """MAE in whatever units the inputs are in (we call it on de-normalised logs)."""
    # elementwise |target-prediction|, averaged over the batch, converted from a 0-d tensor to a Python float
    return torch.mean(torch.abs(target - prediction)).item()


def pick_device(requested):
    """Resolve --device auto to the best thing actually available.

    CUDA first (Colab, any NVIDIA box), then Apple's MPS, then CPU. MPS is
    checked second because on the Macs this project runs on it is usually
    unavailable anyway - macOS 12 with torch 2.2 reports False - but on a newer
    machine it is a large win over CPU.
    """
    if requested != "auto":
        return torch.device(requested)  # an explicit device string ("cpu"/"cuda"/"mps") is honoured as-is
    if torch.cuda.is_available():
        return torch.device("cuda")  # an NVIDIA GPU with a working CUDA driver is present
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple's Metal GPU backend is present
    return torch.device("cpu")  # fall back to the CPU when no accelerator is available


def to_device(inputs, target, device):
    """Move one collated batch onto `device`.

    Everything the model touches has to live on the same device as its weights.
    Three of the four input tensors move the obvious way; `crystal_atom_idx` is
    a LIST of index tensors (one per crystal in the batch), so it needs mapping
    element by element rather than a single .to() call.
    """
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs  # unpack the 4-tuple produced by collate_pool
    # .to(device, non_blocking=True) copies each tensor onto the target device asynchronously;
    # crystal_atom_idx is copied element-by-element since it is a Python list, not a single tensor.
    return (atom_fea.to(device, non_blocking=True),
            nbr_fea.to(device, non_blocking=True),
            nbr_fea_idx.to(device, non_blocking=True),
            [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
            target.to(device, non_blocking=True))


def run_epoch(loader, model, criterion, normalizer, optimizer=None,
              device=torch.device("cpu")):
    """Run one pass over `loader`. Trains if an optimizer is given, else evaluates.

    Returns (mean loss, MAE in log10 units).

    Note the two different spaces in play:
      * the LOSS is computed in normalised space, because that is what the
        network outputs and what gradients flow through;
      * the MAE is reported in de-normalised log10 space, because that is a
        number we can actually interpret and compare against the paper.
    """
    training = optimizer is not None  # True in the training phase, False when just evaluating
    model.train() if training else model.eval()  # toggles dropout/batchnorm behaviour for train vs eval mode

    losses, maes, n_seen = 0.0, 0.0, 0  # running totals accumulated across batches

    # torch.enable_grad/no_grad as a context manager keeps this one function
    # usable for both phases without duplicating the loop.
    with torch.enable_grad() if training else torch.no_grad():
        for inputs, target, _ in loader:  # iterate batches; the 3rd element (crystal ids) is unused here
            (atom_fea, nbr_fea, nbr_fea_idx,
             crystal_atom_idx, target) = to_device(inputs, target, device)  # move this batch onto `device`
            target_normed = normalizer.norm(target)  # z-score the raw log10 targets using train-set mean/std

            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # forward pass: predicted normalised modulus
            loss = criterion(output, target_normed)  # MSE between normalised prediction and normalised target

            if training:
                optimizer.zero_grad()  # clear gradients accumulated from the previous step
                loss.backward()  # backprop: populate .grad on every trainable parameter
                optimizer.step()  # apply one gradient-descent update using those gradients

            batch_size = target.size(0)  # number of crystals in this batch
            losses += loss.item() * batch_size  # accumulate loss weighted by batch size (batches can be uneven)
            # de-normalise the prediction back to real log10 units before scoring, so MAE is interpretable
            maes += mean_absolute_error(normalizer.denorm(output.data), target) * batch_size
            n_seen += batch_size  # running total of crystals processed, used as the divisor below

    return losses / n_seen, maes / n_seen  # mean loss and mean MAE over the whole epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # __doc__ (the module docstring above) becomes the --help text, shown verbatim
    parser.add_argument("--target", choices=["K_VRH", "G_VRH"], default="K_VRH",
                        help="which modulus to predict")  # restricted to these two exact target-column names
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data_full"))
    # ^ directory holding graphs.pt/labels.csv, defaults to <project root>/data_full
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))
    # ^ directory checkpoints/history/summary get written into
    parser.add_argument("--tag", default=None,
                        help="suffix for the output files. Defaults to --target. "
                             "Ensemble members need distinct tags, e.g. "
                             "K_VRH_full / K_VRH_s1 / K_VRH_s2.")
    # None here means "use --target", resolved just below once args are parsed
    parser.add_argument("--epochs", type=int, default=200)  # number of passes over the training set
    parser.add_argument("--batch-size", type=int, default=32)  # crystals per gradient step
    parser.add_argument("--lr", type=float, default=0.02)  # Adam's initial learning rate
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 penalty; small but non-zero helps on tiny datasets")
    # Model size - the paper's widths, see docstring point 3.
    parser.add_argument("--atom-fea-len", type=int, default=64)  # per-atom embedding width inside the conv layers
    parser.add_argument("--h-fea-len", type=int, default=128)  # width of the pooled crystal-level hidden layer
    parser.add_argument("--n-conv", type=int, default=3)  # number of graph-convolution layers
    parser.add_argument("--n-h", type=int, default=1)  # number of fully-connected layers after pooling
    # Splits
    parser.add_argument("--train-ratio", type=float, default=0.7)  # fraction of crystals assigned to training
    parser.add_argument("--val-ratio", type=float, default=0.15)  # fraction assigned to validation (rest = test)
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds WEIGHT INITIALISATION and batch shuffling. "
                             "Vary this to build an ensemble.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="seeds the TRAIN/VAL/TEST SPLIT. Defaults to "
                             "--seed. Every member of an ensemble must pass the "
                             "SAME value here, or their test sets differ and "
                             "the ensemble score is measured on data some "
                             "members trained on.")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | mps. 'auto' picks CUDA, then "
                             "MPS, then CPU.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. Use 2-4 on a GPU "
                             "box; 0 is faster on a 2-core laptop.")
    parser.add_argument("--scheduler", choices=["plateau", "cosine"],
                        default="plateau",
                        help="cosine anneals the LR smoothly to ~0 over the run "
                             "and generally beats plateau for a fixed budget")
    args = parser.parse_args()  # parse sys.argv into an args namespace using the flags defined above
    if args.split_seed is None:
        args.split_seed = args.seed  # default split-seed to the init seed when the user didn't override it
    if args.tag is None:
        args.tag = args.target  # default the output-file tag to the target name

    # Seed everything we can so a rerun reproduces the same numbers.
    torch.manual_seed(args.seed)  # seeds torch's global RNG (weight init, dropout, shuffling)
    np.random.seed(args.seed)  # seeds numpy's global RNG

    device = pick_device(args.device)  # resolve "auto"/"cpu"/"cuda"/"mps" to a concrete torch.device
    print(f"=== Training CGCNN from scratch for {args.target} ===")
    print(f"    device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}\n")
    # ^ appends the GPU's model name only when actually running on CUDA

    dataset = load_dataset_for(args.data_dir, args.target)  # load the cached graph list + target column
    print(f"Dataset: {len(dataset)} labelled crystals")

    # Note --split-seed, not --seed: ensemble members vary their init seed but
    # must share one split, so the held-out test set stays held out for all.
    train_idx, val_idx, test_idx = split_indices(
        len(dataset), args.train_ratio, args.val_ratio, args.split_seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test\n")

    # num_workers>0 pays off on a GPU box, where collating batches on the main
    # thread becomes the bottleneck instead of the maths. On this 2-core laptop
    # the worker processes cost more than they save, hence the 0 default.
    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_pool,
                         num_workers=args.num_workers)  # shared kwargs so all 3 loaders batch identically
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **loader_kwargs)
    # ^ yields shuffled batches drawn only from the training indices
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **loader_kwargs)
    # ^ batches drawn only from the validation indices
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **loader_kwargs)
    # ^ batches drawn only from the held-out test indices

    # --- Normalizer fitted on the TRAINING TARGETS ONLY --------------------
    print("Fitting normalizer on training targets...")
    train_targets = torch.stack([dataset[i][1] for i in train_idx]).view(-1)
    # ^ gather every training-split target into one flat 1-D tensor
    normalizer = Normalizer(train_targets).to(device)  # mean/std computed from TRAIN ONLY, buffers moved to `device`
    print(f"  log10({args.target}): mean={normalizer.mean:.3f} std={normalizer.std:.3f}")

    # --- Model -------------------------------------------------------------
    # Feature widths are read off a real sample rather than hard-coded, so the
    # model automatically matches whatever atom_init.json and Gaussian settings
    # the dataset was built with.
    sample_structures, _, _ = dataset[0]  # peek at one sample purely to read off feature widths
    orig_atom_fea_len = sample_structures[0].shape[-1]  # width of the raw per-atom feature vector (92 by default)
    nbr_fea_len = sample_structures[1].shape[-1]  # width of the Gaussian-expanded bond-distance feature

    model = CrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h, classification=False).to(device)
    # ^ builds the conv-layers + pooling + FC-head network and moves it onto `device`

    n_params = sum(p.numel() for p in model.parameters())  # total trainable scalar count across every tensor
    print(f"  model parameters: {n_params:,}  "
          f"({n_params / max(len(train_idx), 1):,.0f} per training sample)\n")

    criterion = nn.MSELoss()  # mean-squared-error loss computed on the normalised targets
    # Adam rather than the paper's SGD: with only a few hundred samples the
    # per-parameter adaptive step sizes converge far more reliably.
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)  # optimizer over every model parameter
    # Both schedulers exist to do the same thing - shrink the step size as the
    # model closes in, so it settles into a minimum instead of bouncing around
    # it. They differ in how they decide when:
    #   plateau  reactive: halve the LR after 20 epochs with no improvement.
    #            Safe, but it spends those 20 epochs taking steps that are
    #            already too big.
    #   cosine   scheduled: glide the LR from lr to ~0 over the whole run. With
    #            a fixed epoch budget this is almost always the better choice,
    #            because the late epochs are guaranteed to be fine-tuning steps.
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        # ^ anneals lr down to lr*0.01 following a cosine curve over `epochs` steps
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20)
        # ^ halves lr after 20 epochs with no val_mae improvement

    # --- Training loop -----------------------------------------------------
    # Created up front because the loop checkpoints into it as it goes.
    os.makedirs(args.out_dir, exist_ok=True)  # ensure the output directory exists before anything writes into it
    history = []  # list of per-epoch metric dicts, dumped to history_<tag>.csv
    best_val_mae = float("inf")  # running best validation score seen so far
    best_state = None  # will hold a CPU copy of the best model's weights
    start = time.time()  # wall-clock start, used to report total training time

    for epoch in range(args.epochs):  # one full train+val pass per iteration
        train_loss, train_mae = run_epoch(train_loader, model, criterion,
                                          normalizer, optimizer, device)
        # ^ passing `optimizer` puts run_epoch in training mode (weights get updated)
        val_loss, val_mae = run_epoch(val_loader, model, criterion, normalizer,
                                      device=device)
        # ^ no optimizer passed -> eval mode, no weight updates, just scoring
        # ReduceLROnPlateau needs the metric it is watching; CosineAnnealingLR
        # is on a fixed schedule and takes no argument.
        scheduler.step(val_mae) if args.scheduler == "plateau" else scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae": train_mae, "val_mae": val_mae,
                        "lr": optimizer.param_groups[0]["lr"]})
        # ^ record this epoch's metrics for the CSV log

        # Keep the best weights seen so far (see design decision 4).
        if val_mae < best_val_mae:  # only checkpoint when validation actually improved
            best_val_mae = val_mae  # update the running best
            # .cpu() so a checkpoint trained on Colab loads on a laptop with
            # no CUDA. Without it torch.load would demand a GPU.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # ^ detach from the autograd graph, move to CPU, deep-copy every weight tensor
            # Also flush them to disk. On the 11k-crystal run an epoch costs
            # tens of seconds and the whole thing runs for hours, so a crash
            # partway through must not throw away the weights we already have.
            # The file is ~0.5 MB; writing it is free next to an epoch.
            torch.save({"state_dict": best_state,
                        "normalizer": normalizer.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "best_val_mae": best_val_mae,
                        "split": {"train": train_idx, "val": val_idx,
                                  "test": test_idx},
                        "feature_lens": {"orig_atom_fea_len": orig_atom_fea_len,
                                         "nbr_fea_len": nbr_fea_len}},
                       os.path.join(args.out_dir, f"model_{args.tag}.partial.pth"))
            # ^ write the current-best checkpoint to disk immediately, in case the run crashes later
            # Per-epoch history too, so the training curve is inspectable while
            # the run is still going.
            pd.DataFrame(history).to_csv(
                os.path.join(args.out_dir, f"history_{args.tag}.csv"), index=False)
            # ^ overwrite the running per-epoch log every time a new best is found

        if epoch % 10 == 0 or epoch == args.epochs - 1:  # print progress every 10 epochs, plus the final one
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"train_mae {train_mae:.4f}  val_mae {val_mae:.4f}"
                  f"{'  <- best' if val_mae == best_val_mae else ''}")

    elapsed = time.time() - start  # total wall-clock seconds spent in the training loop
    print(f"\nTrained {args.epochs} epochs in {elapsed:.0f}s")

    # Restore the best checkpoint before touching the test set.
    model.load_state_dict(best_state)  # load the best-epoch weights back into the live model
    test_loss, test_mae = run_epoch(test_loader, model, criterion, normalizer,
                                    device=device)
    # ^ one pass over the held-out test set using the restored best weights
    print(f"Best val MAE : {best_val_mae:.4f} log10(GPa)")
    print(f"Test MAE     : {test_mae:.4f} log10(GPa)")

    # --- Save everything the evaluation script will need -------------------
    tag = args.tag  # local alias, just for brevity in the save calls below

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
    # ^ final checkpoint: weights, normalizer stats, args, scores and the exact split used

    pd.DataFrame(history).to_csv(
        os.path.join(args.out_dir, f"history_{tag}.csv"), index=False)
    # ^ final overwrite of the per-epoch log, now complete

    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump({"tag": tag, "target": args.target,
                   "data_dir": os.path.basename(args.data_dir),
                   "n_total": len(dataset),
                   "n_train": len(train_idx), "n_val": len(val_idx),
                   "n_test": len(test_idx), "best_val_mae": best_val_mae,
                   "test_mae": test_mae, "epochs": args.epochs,
                   "n_params": n_params, "seconds": elapsed}, fh, indent=2)
        # ^ human-readable run summary, consumed later by 06_summarise.py

    # The run finished, so the crash-recovery copy is now just clutter.
    partial = os.path.join(args.out_dir, f"model_{tag}.partial.pth")  # path of the crash-recovery checkpoint
    if os.path.exists(partial):  # only attempt removal if it was actually created
        os.remove(partial)  # delete it now that the final checkpoint above supersedes it

    print(f"\nSaved to {args.out_dir}/:")
    print(f"  model_{tag}.pth      weights + normalizer + split")
    print(f"  history_{tag}.csv    per-epoch losses")
    print(f"  summary_{tag}.json   final metrics")


if __name__ == "__main__":  # standard entry-point guard so importing this module doesn't auto-run training
    main()  # invoke the CLI when run directly as a script
