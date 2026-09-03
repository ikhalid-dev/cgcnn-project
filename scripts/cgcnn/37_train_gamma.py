#!/usr/bin/env python3
"""
STEP 37 - Train the THREE-HEAD model: G, K/G, and gamma.
=========================================================

    python scripts/cgcnn/37_train_gamma.py                    # 3 heads
    python scripts/cgcnn/37_train_gamma.py --n-heads 2        # ablation

THE QUESTION
------------
The pipeline currently DERIVES the Grueneisen parameter from predicted K/G
through the empirical Poisson relation. Measured against AFLOW's own tabulated
gamma on 1,460 crystals, that derivation is poor:

    MAE 0.4326    correlation 0.4205

and it costs real accuracy. Substituting the tabulated value into the Slack
formula, changing nothing else, improves agreement with AFLOW's independently
computed kappa by 25%:

    kappa from DERIVED gamma   MAE log10 0.2395   r 0.9118
    kappa from REAL    gamma   MAE log10 0.1806   r 0.9125    <- the ceiling

So: can a third head learn gamma well enough to capture some of that 25%?

WHY THIS TRAINS ON AFLOW AND NOT MATBENCH
------------------------------------------
Because matbench cannot answer the question without circularity. Its "true
kappa" is not measured - it is Slack(true K, true G) with gamma DERIVED from
true K/G. A model predicting a better gamma scores WORSE against that target
by construction, because the target assumes the derivation is right.

AFLOW supplies gamma AND an independently computed kappa, so both the label
and the yardstick are real. It is also a better-suited dataset for this
project's actual goal: median kappa 1.68 W/m/K with 472 crystals below
1 W/m/K, against matbench where the low-kappa tail is the thin end.

WHAT IS REPORTED
----------------
Three kappa numbers on the same held-out crystals, all against AFLOW's kappa:

    derived      gamma from PREDICTED K/G      the current pipeline
    predicted    gamma from the third head     the thing under test
    oracle       gamma from AFLOW's table      the ceiling, with predicted K,G

plus recall@10% of the true lowest-kappa decile, which is the screening
metric and the one that seven rounds of earlier work never moved.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- data ---------------------------------------------------------------
    "data_dir": "data_full",
    "cache_file": "gamma_graphs.pt",
    "labels_file": "gamma_labels.csv",
    "out_dir": "results/cgcnn",
    "tag": "gamma",

    # ---- heads --------------------------------------------------------------
    # 3 = log10(G), log10(K/G), log10(gamma).  2 drops the gamma head and is
    # the honest ablation: same data, same recipe, gamma derived as before.
    "n_heads": 3,
    # Relative weight of each head. The gamma head is the point of the
    # exercise, so it is not down-weighted; raise it above 1 to trade moduli
    # accuracy for gamma accuracy.
    "loss_weight_G": 1.0,
    "loss_weight_ratio": 1.0,
    "loss_weight_gamma": 1.0,

    # ---- architecture -------------------------------------------------------
    # Smaller than the matbench model: 1,460 crystals is a seventh of the data,
    # so the 113k-parameter version would overfit hard.
    "atom_fea_len": 64,
    "n_conv": 3,
    "h_fea_len": 128,
    "n_shared_fc": 1,
    "n_head_fc": 1,
    "head_fea_len": 64,
    "dropout": 0.10,

    # ---- loss and optimisation ---------------------------------------------
    "loss_fn": "huber",          # round 3's winner on the matbench task
    "huber_delta": 1.0,
    "epochs": 300,               # more than matbench's 200: far fewer batches
    "batch_size": 64,            # smaller dataset, smaller batches
    "learning_rate": 0.01,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "cosine",
    "early_stop_patience": 80,

    # ---- split --------------------------------------------------------------
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,
    "init_seed": 42,

    # ---- runtime ------------------------------------------------------------
    "device": "auto",
    "num_workers": 0,
    "print_every": 50,
}
# =============================================================================

import argparse  # stdlib CLI argument parser; parse_overrides() builds one flag per CONFIG key
import json  # stdlib JSON encode/decode, used to write the summary file
import os  # stdlib path/filesystem helpers
import sys  # stdlib interpreter access, used to extend the import path
import time  # stdlib wall-clock timer
import warnings  # stdlib warning-filter control

import torch  # PyTorch core: tensors, autograd, device management
import torch.nn as nn  # neural-network loss functions (MSE/L1/Huber)
import torch.optim as optim  # optimizers (Adam/AdamW) and LR schedulers
from torch.utils.data import DataLoader  # batches a dataset via a sampler + collate function
from torch.utils.data.sampler import SubsetRandomSampler  # draws shuffled indices from one fixed subset

import numpy as np  # array ops: log10, sqrt, correlation, argsort for the recall metric
import pandas as pd  # DataFrame, reads the AFLOW labels CSV

warnings.filterwarnings("ignore")  # silence deprecation/runtime warnings so they don't clutter the log

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ absolute path of this file, walked up 3 directories: scripts/cgcnn/37_train_gamma.py -> project root
sys.path.insert(0, PROJECT_ROOT)  # prepend project root to the import search path so `cgcnn_scratch` resolves

from cgcnn_scratch.data import collate_pool  # noqa: E402 - batches graph samples into one padded tensor
from cgcnn_scratch.joint import (  # noqa: E402
    JointCrystalGraphConvNet, VectorNormalizer, JointGraphCacheData)
# ^ JointCrystalGraphConvNet: the shared-trunk, multi-head model class.
#   VectorNormalizer: per-column z-scoring for a multi-target vector.
#   JointGraphCacheData: Dataset wrapper that pairs cached graphs with an
#   arbitrary per-crystal target tuple plus a trailing sample weight.

AMU_KG = 1.66053906660e-27  # 1 atomic mass unit in kilograms (unused directly below, kept for reference/reuse)


def parse_overrides():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # __doc__ (the module docstring above) becomes the --help text
    for key, default in CONFIG.items():  # auto-generate one CLI flag per CONFIG entry
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default), default=None,
                            help=f"override CONFIG['{key}'] (default: {default!r})")
        # ^ dest=key stores the value under the same name as the CONFIG key;
        #   type=type(default) coerces the CLI string to whatever type the default already is;
        #   default=None means "not overridden" so it can be distinguished from a genuine override below.
    args = parser.parse_args()  # parse sys.argv into an args namespace
    cfg = dict(CONFIG)  # shallow copy so the module-level CONFIG constant is never mutated
    for key, value in vars(args).items():  # vars(args) turns the namespace into a plain dict
        if value is not None:  # only overwrite entries the user actually passed on the command line
            cfg[key] = value
    assert cfg["n_heads"] in (2, 3), "n_heads must be 2 or 3"  # only these two architectures are supported
    return cfg


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)  # an explicit device string ("cpu"/"cuda"/"mps") is honoured as-is
    if torch.cuda.is_available():
        return torch.device("cuda")  # an NVIDIA GPU with a working CUDA driver is present
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple's Metal GPU backend is present
    return torch.device("cpu")  # fall back to the CPU when no accelerator is available


def split_indices(n_total, train_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)  # seeded PRNG, independent of numpy's global RNG state
    idx = rng.permutation(n_total)  # a random shuffle of the integers 0..n_total-1
    n_tr = int(round(train_ratio * n_total))  # number of rows assigned to the training split
    n_va = int(round(val_ratio * n_total))  # number of rows assigned to the validation split
    # first n_tr shuffled indices -> train; next n_va -> val; the remainder -> test
    return idx[:n_tr].tolist(), idx[n_tr:n_tr + n_va].tolist(), idx[n_tr + n_va:].tolist()


def kappa_full(log_K, log_G, gamma, volume_m3, n_sites, density_g_cm3):
    """Slack kappa in W/m/K - the FULL formula, no cancelling.

    Every earlier script dropped density and volume because they cancel in a
    predicted/true ratio. Scoring against AFLOW's ABSOLUTE kappa means they do
    not cancel, so they are carried explicitly here. Units are SI throughout:
    moduli Pa, density kg/m3, volume m3.
    """
    K = (10.0 ** np.asarray(log_K, float)) * 1e9  # undo log10 and GPa->Pa (1e9) in one step
    G = (10.0 ** np.asarray(log_G, float)) * 1e9  # same for shear modulus
    rho = np.asarray(density_g_cm3, float) * 1e3  # g/cm^3 -> kg/m^3
    with np.errstate(all="ignore"):  # some rows can produce NaN/inf (e.g. non-physical predictions); suppress the warning, not the NaN itself
        v_long = np.sqrt((K + 4.0 / 3.0 * G) / rho)  # longitudinal elastic wave speed
        v_trans = np.sqrt(G / rho)  # transverse elastic wave speed
        v_s = ((1.0 / v_long ** 3 + 2.0 / v_trans ** 3) / 3.0) ** (-1.0 / 3.0)  # 1:2 harmonic-mean sound velocity
        return (G * v_s * np.asarray(volume_m3, float) ** (1.0 / 3.0)
                / (np.asarray(n_sites, float) * 300.0) * np.exp(-np.asarray(gamma, float)))
        # ^ the Slack model: kappa proportional to G * v_sound * V^(1/3) / (N_atoms * T), with anharmonic
        #   damping exp(-gamma); T is fixed at 300 K here.


def derived_gamma(log_K, log_G):
    """The empirical Poisson relation the third head is meant to replace."""
    with np.errstate(all="ignore"):  # guard against divide-by-zero for degenerate K/G ratios
        x2 = (10.0 ** np.asarray(log_K, float)) / (10.0 ** np.asarray(log_G, float)) + 4.0 / 3.0
        # ^ x2 is an intermediate term in the standard K/G -> Poisson-ratio conversion
        nu = (x2 - 2.0) / (2.0 * x2 - 2.0)  # Poisson's ratio, computed purely from the K/G ratio
        return 3.0 * (1.0 + nu) / (2.0 * (2.0 - 3.0 * nu))  # the empirical Gruneisen-from-Poisson relation


def build_loss(cfg):
    name = cfg["loss_fn"]  # which loss function CONFIG/CLI selected
    if name == "mse":
        return nn.MSELoss(reduction="none")  # per-element squared error, unreduced so per-head weights can be applied
    if name == "l1":
        return nn.L1Loss(reduction="none")  # per-element absolute error, unreduced
    if name == "huber":
        return nn.HuberLoss(reduction="none", delta=cfg["huber_delta"])  # quadratic near 0, linear beyond `delta`
    raise ValueError(f"unknown loss_fn {name!r}")  # fail loudly on a typo'd --loss-fn value


def to_device(inputs, target, device):
    atom_fea, nbr_fea, nbr_idx, crys_idx = inputs  # unpack the 4-tuple produced by collate_pool
    return (atom_fea.to(device), nbr_fea.to(device), nbr_idx.to(device),
            [i.to(device) for i in crys_idx], target.to(device))
    # ^ crys_idx is a Python list of per-crystal index tensors, so it is moved element-by-element


def run_epoch(loader, model, criterion, normalizer, head_w, n_heads,
              optimizer=None, device=torch.device("cpu")):
    """One pass. Returns (loss, predictions, truths, ids) in physical units."""
    training = optimizer is not None  # True in the training phase, False when just evaluating
    model.train() if training else model.eval()  # toggles dropout behaviour for train vs eval mode
    total, seen, preds, trues, ids = 0.0, 0, [], [], []
    # ^ running loss total, running crystal count, and per-batch accumulators for predictions/truths/ids

    with torch.enable_grad() if training else torch.no_grad():
        for inputs, target, batch_ids in loader:  # iterate batches; batch_ids are this batch's crystal identifiers
            ids.extend(batch_ids)  # accumulate ids across all batches, in the order they were seen
            a, n, ni, ci, target = to_device(inputs, target, device)  # move this batch's tensors onto `device`
            # The dataset appends a per-crystal loss weight as the LAST column,
            # so the batch arrives as (B, n_heads + 1). Drop it here - this
            # script does not use sample weighting, and leaving it in would
            # hand the normaliser a column it has no statistics for.
            target = target[:, :n_heads]  # keep only the n_heads target columns, drop the trailing weight column
            target_normed = normalizer.norm(target)  # z-score each target column using its own train-set mean/std
            output = model(a, n, ni, ci)  # forward pass: predicted normalised (G, K/G[, gamma]) per crystal
            loss = (criterion(output, target_normed) * head_w).mean()
            # ^ per-element loss scaled by each head's weight, then averaged into one scalar
            if training:
                optimizer.zero_grad()  # clear gradients accumulated from the previous step
                loss.backward()  # backprop: populate .grad on every trainable parameter
                optimizer.step()  # apply one gradient-descent update
            bs = target.size(0)  # number of crystals in this batch
            total += loss.item() * bs  # accumulate loss weighted by batch size
            seen += bs  # running total of crystals processed
            preds.append(normalizer.denorm(output.detach()).cpu().numpy())
            # ^ de-normalise back to real log10 units, detach from autograd, move to CPU, convert to numpy
            trues.append(target.detach().cpu().numpy())  # same detach/CPU/numpy treatment for the targets
    return total / seen, np.concatenate(preds), np.concatenate(trues), ids
    # ^ mean loss over the epoch, plus every prediction/truth/id stacked into one array/list


def score(pred, true, ids, meta, n_heads):
    """Every metric that matters, on one split.

    Columns are log10(G), log10(K/G), and - when present - log10(gamma).
    K is recovered as log10(G) + log10(K/G), which is exact.
    """
    log_G_p, log_G_t = pred[:, 0], true[:, 0]  # predicted and true log10(G), column 0
    log_K_p, log_K_t = pred[:, 0] + pred[:, 1], true[:, 0] + true[:, 1]
    # ^ log10(K) = log10(G) + log10(K/G); exact because it is the inverse of how the targets were built

    m = meta.loc[ids]  # look up this split's rows from the full labels table, in prediction order
    vol, nst, dens = m.volume_m3.values, m.n_sites.values, m.density_g_cm3.values  # per-crystal physical constants
    kappa_true = m.kappa_agl.values  # AFLOW's own independently computed kappa, the scoring yardstick
    gamma_true = m.gamma.values  # AFLOW's own tabulated Gruneisen parameter

    variants = {
        # The current pipeline: gamma derived from the model's own K/G.
        "derived": derived_gamma(log_K_p, log_G_p),
        # The ceiling: perfect gamma, the model's own moduli. Isolates how much
        # of the remaining error is the moduli rather than gamma.
        "oracle": gamma_true,
    }
    if n_heads == 3:
        variants["predicted"] = 10.0 ** pred[:, 2]  # undo log10 on the third head's raw output

    out = {
        "mae_log_K": float(np.abs(log_K_p - log_K_t).mean()),  # mean absolute error of the recovered log10(K)
        "mae_log_G": float(np.abs(log_G_p - log_G_t).mean()),  # mean absolute error of log10(G)
        "mae_log_ratio": float(np.abs(pred[:, 1] - true[:, 1]).mean()),  # mean absolute error of log10(K/G)
    }
    if n_heads == 3:
        out["mae_gamma"] = float(np.abs(10.0 ** pred[:, 2] - gamma_true).mean())  # gamma-head MAE vs AFLOW's real gamma
        out["gamma_corr"] = float(np.corrcoef(10.0 ** pred[:, 2], gamma_true)[0, 1])  # Pearson r, gamma-head vs real gamma
    out["mae_gamma_derived"] = float(np.abs(variants["derived"] - gamma_true).mean())  # derived-gamma MAE vs real gamma
    out["gamma_corr_derived"] = float(
        np.corrcoef(variants["derived"][np.isfinite(variants["derived"])],
                    gamma_true[np.isfinite(variants["derived"])])[0, 1])
    # ^ np.isfinite mask drops any row where derived_gamma() produced NaN/inf before computing the correlation

    for name, gam in variants.items():  # score kappa under each of the derived/oracle[/predicted] gamma choices
        kp = kappa_full(log_K_p, log_G_p, gam, vol, nst, dens)  # this variant's predicted kappa, W/m/K
        ok = np.isfinite(kp) & (kp > 0) & (kappa_true > 0)  # only score rows with a physically valid kappa on both sides
        err = np.abs(np.log10(kp[ok]) - np.log10(kappa_true[ok]))  # per-row absolute log10 kappa error
        out[f"kappa_mae_{name}"] = float(err.mean())
        # Screening metric: of the true lowest-kappa decile, how many does the
        # model put in ITS lowest decile? Ranking only, calibration-free.
        kt, kpo = kappa_true[ok], kp[ok]  # aligned true/predicted kappa arrays, valid rows only
        n10 = max(1, int(0.10 * len(kt)))  # size of the bottom decile, at least 1 crystal
        out[f"recall10_{name}"] = len(set(np.argsort(kt)[:n10])
                                      & set(np.argsort(kpo)[:n10])) / n10
        # ^ argsort ascending, so [:n10] is the n10 lowest-kappa crystals by that ranking;
        #   set intersection size divided by n10 gives the fraction of the true bottom decile recovered
    return out


def main():
    cfg = parse_overrides()  # CONFIG merged with any CLI overrides
    torch.manual_seed(cfg["init_seed"])  # seeds torch's global RNG (weight init, dropout, shuffling)
    np.random.seed(cfg["init_seed"])  # seeds numpy's global RNG
    device = pick_device(cfg["device"])  # resolve "auto"/"cpu"/"cuda"/"mps" to a concrete torch.device
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])  # absolute path to the data directory
    out_dir = os.path.join(PROJECT_ROOT, cfg["out_dir"])  # absolute path to the results directory
    os.makedirs(out_dir, exist_ok=True)  # ensure the output directory exists before anything writes into it

    print("=" * 78)
    print(f"  {cfg['n_heads']}-HEAD model on AFLOW  (tag={cfg['tag']}, device={device})")
    print("=" * 78)

    meta = pd.read_csv(os.path.join(data_dir, cfg["labels_file"])).set_index("gid")
    # ^ the full AFLOW labels table (gamma_labels.csv), indexed by its graph-id column for fast .loc lookups
    log_K = np.log10(meta.K_VRH.values)  # log10 bulk modulus for every crystal in the labels table
    log_G = np.log10(meta.G_VRH.values)  # log10 shear modulus for every crystal
    log_gamma = np.log10(meta.gamma.values)  # log10 Gruneisen parameter for every crystal

    cols = [log_G, log_K - log_G] + ([log_gamma] if cfg["n_heads"] == 3 else [])
    # ^ the head targets: log10(G), log10(K/G) = log10(K)-log10(G), and log10(gamma) only when n_heads==3
    targets = {i: tuple(c[k] for c in cols) for k, i in enumerate(meta.index)}
    # ^ maps each crystal's graph-id to its tuple of target values, one entry per head

    # JointGraphCacheData yields (targets..., weight); with no weights dict it
    # appends a constant 1.0, which the slice below drops.
    dataset = JointGraphCacheData(os.path.join(data_dir, cfg["cache_file"]), targets)
    # ^ pairs the cached graph objects (gamma_graphs.pt) with the target tuples built above
    n_heads = cfg["n_heads"]  # local alias, used throughout the rest of main()
    print(f"  {len(dataset)} crystals, {n_heads} heads")

    tr, va, te = split_indices(len(dataset), cfg["train_ratio"],
                               cfg["val_ratio"], cfg["split_seed"])
    print(f"  split: {len(tr)} train / {len(va)} val / {len(te)} test")

    lk = dict(batch_size=cfg["batch_size"], collate_fn=collate_pool,
              num_workers=cfg["num_workers"])  # shared DataLoader kwargs so all 3 loaders batch identically
    loaders = {
        "train": DataLoader(dataset, sampler=SubsetRandomSampler(tr), **lk),  # shuffled batches from the train indices
        "val": DataLoader(dataset, sampler=SubsetRandomSampler(va), **lk),  # batches from the validation indices
        "test": DataLoader(dataset, sampler=SubsetRandomSampler(te), **lk),  # batches from the held-out test indices
    }

    train_t = torch.stack([dataset[i][1] for i in tr])[:, :n_heads]
    # ^ every training-split target row stacked into one tensor, weight column sliced off
    normalizer = VectorNormalizer(train_t).to(device)  # per-column mean/std computed from TRAIN ONLY
    names = ["log10(G)", "log10(K/G)", "log10(gamma)"][:n_heads]  # human-readable label per head, for the printout
    for i, nm in enumerate(names):
        print(f"  normalizer {nm:<13} mean={normalizer.mean[i]:.3f} "
              f"std={normalizer.std[i]:.3f}")

    sample, _, _ = dataset[0]  # peek at one sample purely to read off feature widths
    model = JointCrystalGraphConvNet(
        sample[0].shape[-1], sample[1].shape[-1],
        atom_fea_len=cfg["atom_fea_len"], n_conv=cfg["n_conv"],
        h_fea_len=cfg["h_fea_len"], n_shared_fc=cfg["n_shared_fc"],
        n_head_fc=cfg["n_head_fc"], head_fea_len=cfg["head_fea_len"],
        dropout=cfg["dropout"], n_heads=n_heads).to(device)
    # ^ sample[0].shape[-1] is the per-atom feature width, sample[1].shape[-1] the bond-feature width;
    #   builds the shared-trunk multi-head network and moves it onto `device`
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    criterion = build_loss(cfg)  # the configured loss function (mse/l1/huber), unreduced
    head_w = torch.tensor(
        [cfg["loss_weight_G"], cfg["loss_weight_ratio"], cfg["loss_weight_gamma"]][:n_heads],
        device=device)  # per-head loss weight vector, trimmed to n_heads entries, on the training device
    optimizer = (optim.AdamW if cfg["optimizer"] == "adamw" else optim.Adam)(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    # ^ picks AdamW or plain Adam based on CONFIG, then constructs it over every model parameter
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["learning_rate"] * 0.01)
    # ^ anneals the learning rate down to 1% of its start value over the full `epochs` budget

    # Model selection on the DERIVED-gamma kappa would reward a model for
    # matching the thing we are trying to replace. Select on the quantity the
    # project actually wants: kappa error against AFLOW's own kappa, using
    # whatever gamma this model produces.
    select_key = "kappa_mae_predicted" if n_heads == 3 else "kappa_mae_derived"
    # ^ the score() dict key used to decide which epoch is "best"

    best, best_state, best_epoch, stale = float("inf"), None, -1, 0
    # ^ best score so far, its weights, its epoch number, and epochs since the last improvement
    start = time.time()  # wall-clock start, used to report total training time
    for epoch in range(cfg["epochs"]):  # one full train+val pass per iteration
        run_epoch(loaders["train"], model, criterion, normalizer, head_w,
                  n_heads, optimizer, device)
        # ^ passing `optimizer` puts run_epoch in training mode; its return values are unused here
        scheduler.step()  # advance the cosine schedule by one epoch
        _, vp, vt, vi = run_epoch(loaders["val"], model, criterion, normalizer,
                                  head_w, n_heads, device=device)
        # ^ no optimizer passed -> eval mode; vp/vt/vi are this epoch's validation predictions/truths/ids
        vs = score(vp, vt, vi, meta, n_heads)  # full metric dict for this epoch's validation predictions
        improved = vs[select_key] < best - 1e-5  # small margin avoids flagging floating-point noise as improvement
        if improved:
            best, best_epoch, stale = vs[select_key], epoch, 0  # record the new best and reset the patience counter
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # ^ detach from autograd, move to CPU, deep-copy every weight tensor
        else:
            stale += 1  # one more epoch without improvement
        if epoch % cfg["print_every"] == 0 or improved or epoch == cfg["epochs"] - 1:
            g = f"  gamma {vs['mae_gamma']:.3f}" if n_heads == 3 else ""  # only show a gamma column when it exists
            print(f"  e{epoch:3d}{' *' if improved else '  '} val  "
                  f"K {vs['mae_log_K']:.4f}  G {vs['mae_log_G']:.4f}"
                  f"{g}  kappa {vs[select_key]:.4f}")
        if cfg["early_stop_patience"] and stale >= cfg["early_stop_patience"]:
            print(f"\n  early stop (best epoch {best_epoch})")
            break  # stop training once validation has not improved for `early_stop_patience` epochs

    model.load_state_dict(best_state)  # restore the best-epoch weights before final scoring
    results = {}
    for split, loader in loaders.items():  # score every split (train/val/test) with the restored best weights
        _, p, t, i = run_epoch(loader, model, criterion, normalizer, head_w,
                               n_heads, device=device)
        results[split] = score(p, t, i, meta, n_heads)

    t = results["test"]  # shorthand for the test-set metric dict, used repeatedly below
    print()
    print("=" * 78)
    print(f"  TEST SET ({len(te)} crystals) - kappa scored against AFLOW's own kappa")
    print("=" * 78)
    print(f"  MAE log10(K) {t['mae_log_K']:.4f}   log10(G) {t['mae_log_G']:.4f}"
          f"   log10(K/G) {t['mae_log_ratio']:.4f}")
    print()
    print(f"  gamma:  derived   MAE {t['mae_gamma_derived']:.4f}"
          f"   corr {t['gamma_corr_derived']:+.4f}")
    if n_heads == 3:
        print(f"          PREDICTED MAE {t['mae_gamma']:.4f}"
              f"   corr {t['gamma_corr']:+.4f}")
    print()
    print(f"  {'kappa source':<14}{'MAE log10':>11}{'recall@10%':>13}")
    for name in ("derived", "predicted", "oracle"):
        if f"kappa_mae_{name}" in t:  # "predicted" only exists when n_heads == 3
            print(f"  {name:<14}{t['kappa_mae_' + name]:11.4f}"
                  f"{100 * t['recall10_' + name]:12.1f}%")
    print(f"\n  reference, using TRUE moduli: derived 0.2395, real gamma 0.1806")

    # Script number (37) stamped into every output filename so results sitting
    # next to other scripts' output in the same results/ directory can be told
    # apart by filename alone - see the project-wide naming convention.
    torch.save({"state_dict": best_state, "normalizer": normalizer.state_dict(),
                "config": cfg, "split": {"train": tr, "val": va, "test": te}},
               os.path.join(out_dir, f"model_37_{cfg['tag']}.pth"))
    with open(os.path.join(out_dir, f"summary_37_{cfg['tag']}.json"), "w") as fh:
        json.dump({"tag": cfg["tag"], "config": cfg, "best_epoch": best_epoch,
                   "seconds": time.time() - start, "results": results}, fh, indent=2)
    print(f"\n  saved results/cgcnn/model_37_{cfg['tag']}.pth "
          f"and summary_37_{cfg['tag']}.json")


if __name__ == "__main__":  # standard entry-point guard so importing this module doesn't auto-run training
    main()  # invoke the CLI when run directly as a script
