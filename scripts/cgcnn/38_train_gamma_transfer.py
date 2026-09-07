#!/usr/bin/env python3
"""
STEP 38 - Two-stage transfer: matbench moduli, then AFLOW gamma.
=================================================================

    python scripts/cgcnn/38_train_gamma_transfer.py

WHY TWO STAGES
--------------
37_train_gamma.py trained all three heads on AFLOW's 1,460 crystals and gave a
clear, instructive result:

    gamma:  derived   MAE 0.3551   corr +0.3336
            PREDICTED MAE 0.1923   corr +0.5662     <- the head WORKS

    kappa   derived   0.3585
            predicted 0.3408                        <- only 5% better
            oracle    0.2979

The gamma head halves the gamma error, and it barely moves kappa. The reason
is in the moduli: MAE log10(K) was 0.1817 there against 0.063 on matbench,
because 1,022 training crystals is a seventh of what matbench offers.
Decomposing the four combinations:

    moduli      gamma        kappa MAE
    predicted   derived        0.3585
    predicted   oracle         0.2979      bad gamma costs 0.061
    TRUE        derived        0.2395      bad moduli cost 0.119
    TRUE        real           0.1806

Bad moduli cost twice what bad gamma costs. Training both on the small set
cripples the moduli in order to enable gamma.

The two label types want different datasets and the model can have both:

    STAGE 1   trunk + G head + K/G head, trained on matbench's 10,987 crystals
    STAGE 2   trunk and moduli heads FROZEN; only the gamma head trains, on
              AFLOW's 1,460 - the sole source of gamma labels

This is KappaFormer's (Wu et al. 2026) transfer scheme. Their paper does it
without explaining why it matters; the table above is the missing measurement.

THE NORMALISER IS THE SUBTLE PART
----------------------------------
Head statistics come from two different datasets. Columns 0 and 1 (G, K/G) are
standardised with matbench training statistics, because that is what stage 1
learned against and those weights are frozen. Column 2 (gamma) is standardised
with AFLOW training statistics, because that is the only place gamma exists.
Fitting all three on either dataset alone would silently rescale the frozen
heads' outputs and destroy stage 1's work.

A KNOWN CONFOUND, STATED UP FRONT
----------------------------------
Matbench and AFLOW moduli disagree on tightly-matched shared compounds, so the
frozen moduli heads carry that disagreement onto the evaluation set and
ABSOLUTE kappa will suffer.

Two corrections to how this used to be described, both from step 59:

  * NOT a convention difference. AFLOW's own columns are
    `ael_bulk_modulus_vrh` / `ael_shear_modulus_vrh` - AEL is the METHOD, and
    it reports VRH averages exactly as matbench does. Both numbers are the same
    physical quantity from two different DFT workflows, so the disagreement is
    workflow (k-points, strain magnitude, relaxation tolerance, functional),
    not definition. The phrase "elastic-convention offset" that this project
    used to attach to this paragraph was wrong.

  * NOT 0.1085, and mostly NOT an offset. That figure is the SOFT SUBSET's;
    on the full 2,666-compound overlap the disagreement is 0.0539 with a mean
    offset of -0.006 - i.e. centred, heavy-tailed scatter, and subtracting a
    constant improves it by 0.6%, which is nothing. The soft subset's apparent
    -0.047 offset is a selection artifact: aflow_soft.csv is selected because
    AFLOW says the crystal is soft, which over-represents AFLOW's downward
    errors, and the bias flips to +0.030 if matbench does the selecting.

The comparison this script exists to make is unaffected: derived gamma and
predicted gamma are computed from the SAME predicted moduli, so any convention
offset hits both equally and cancels. Read the derived-vs-predicted gap, not
the absolute numbers. Set --freeze-moduli 0 to let stage 2 adapt the moduli
heads too, which trades that purity for absolute accuracy.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- stage 1 data: matbench, the big moduli set -------------------------
    "s1_cache": "data_full/graphs.pt",         # cached CGCNN graphs for the matbench crystal set
    "s1_labels": "data_full/labels.csv",       # K_VRH/G_VRH labels matching s1_cache, keyed by mb_id
    # ---- stage 2 data: AFLOW, the only gamma source -------------------------
    "s2_cache": "data_full/gamma_graphs.pt",   # cached CGCNN graphs for the AFLOW-AGL crystal set
    "s2_labels": "data_full/gamma_labels.csv", # K/G/gamma/kappa labels matching s2_cache, keyed by gid

    "out_dir": "results/cgcnn",                # where the final checkpoint and summary JSON are written
    "tag": "xfer",                             # run identifier, used to build every output filename

    # ---- architecture (shared by both stages) -------------------------------
    "atom_fea_len": 64,      # length of the per-atom embedding after the first linear layer
    "n_conv": 3,              # number of graph-convolution layers in the shared trunk
    "h_fea_len": 128,         # width of the shared fully-connected layer after pooling
    "n_shared_fc": 1,         # number of shared fully-connected layers before the heads fork
    "n_head_fc": 1,           # number of fully-connected layers inside each individual head
    "head_fea_len": 64,       # width of the hidden layers inside each head
    "dropout": 0.10,          # dropout probability applied in the shared trunk and heads

    # ---- stage 1 -------------------------------------------------------------
    "s1_epochs": 200,          # maximum training epochs for stage 1
    "s1_batch_size": 128,      # mini-batch size for stage 1 (matbench is large, so a bigger batch)
    "s1_lr": 0.01,             # initial learning rate for stage 1's optimizer
    "s1_weight_decay": 1e-5,   # L2 weight decay for stage 1's optimizer
    "s1_patience": 60,         # stop stage 1 if validation hasn't improved for this many epochs

    # ---- stage 2 -------------------------------------------------------------
    # Freeze the trunk and both moduli heads, so only the gamma head learns.
    # 0 unfreezes everything, which lets the moduli adapt to AFLOW's convention
    # at the cost of no longer isolating the gamma effect.
    "freeze_moduli": 1,        # 1 = freeze trunk+moduli heads in stage 2, 0 = train everything
    "s2_epochs": 300,          # maximum training epochs for stage 2
    "s2_batch_size": 64,       # mini-batch size for stage 2 (AFLOW is smaller, so a smaller batch)
    # Lower than stage 1: a single small head on 1,022 crystals, and if the
    # trunk is unfrozen this is also the rate that would disturb it.
    "s2_lr": 0.003,            # initial learning rate for stage 2's optimizer
    "s2_weight_decay": 1e-5,   # L2 weight decay for stage 2's optimizer
    "s2_patience": 80,         # stop stage 2 if validation hasn't improved for this many epochs

    "loss_fn": "huber",        # which loss function build_loss() constructs below
    "huber_delta": 1.0,        # the delta (transition point) for Huber loss

    # ---- splits --------------------------------------------------------------
    "train_ratio": 0.70,       # fraction of each dataset assigned to the train split
    "val_ratio": 0.15,         # fraction assigned to validation (remainder goes to test)
    "split_seed": 42,          # RNG seed controlling which crystals land in which split
    "init_seed": 42,           # RNG seed controlling model weight initialisation and batch order

    "device": "auto",          # "auto" picks cuda > mps > cpu; can be forced via --device
    "num_workers": 0,          # DataLoader worker processes (0 = load in the main process)
    "print_every": 50,         # print a progress line every this many epochs
}
# =============================================================================

import argparse   # command-line argument parsing (auto-generates one flag per CONFIG key)
import json       # writes the summary JSON output
import os         # path joining and directory creation
import sys        # extends the import search path with PROJECT_ROOT
import time       # wall-clock timing of each stage
import warnings   # suppresses noisy library warnings

import torch                                              # tensor library and autograd
import torch.nn as nn                                     # loss function classes
import torch.optim as optim                                # optimizers and LR schedulers
from torch.utils.data import DataLoader                    # batches dataset items
from torch.utils.data.sampler import SubsetRandomSampler   # draws only the indices belonging to one split

import numpy as np    # array math, RNG for the train/val/test split
import pandas as pd   # reads the label CSVs

warnings.filterwarnings("ignore")   # suppress warning messages from cluttering output

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# three dirname() calls walk up from this file's absolute path to the project root
sys.path.insert(0, PROJECT_ROOT)    # makes `cgcnn_scratch` importable regardless of the current working directory

from cgcnn_scratch.data import collate_pool                     # noqa: E402 - assembles CGCNN's custom graph-batch format
from cgcnn_scratch.joint import (                               # noqa: E402
    JointCrystalGraphConvNet, VectorNormalizer, JointGraphCacheData)  # multi-head model, per-column normalizer, dataset class


def parse_overrides():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in CONFIG.items():                          # auto-generate one --flag per CONFIG entry
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,   # e.g. "s1_epochs" -> "--s1-epochs"
                            type=type(default), default=None,          # cast to the same type as the CONFIG default
                            help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()               # parse sys.argv against the flags just defined
    cfg = dict(CONFIG)                       # start from a copy of the defaults
    for key, value in vars(args).items():    # walk every parsed argument
        if value is not None:                # None means "not passed on the command line"
            cfg[key] = value                 # override the default with whatever was passed
    return cfg


def pick_device(requested):
    if requested != "auto":                   # an explicit device string was requested
        return torch.device(requested)
    if torch.cuda.is_available():             # prefer an NVIDIA GPU if one is visible
        return torch.device("cuda")
    if torch.backends.mps.is_available():     # otherwise prefer Apple Silicon's GPU backend
        return torch.device("mps")
    return torch.device("cpu")                # fall back to the CPU


def split_indices(n, train_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)     # seeded PRNG, deterministic given `seed`
    idx = rng.permutation(n)              # shuffled array of indices 0..n-1
    a = int(round(train_ratio * n))       # number of indices assigned to train
    b = int(round(val_ratio * n))         # number of indices assigned to val
    return idx[:a].tolist(), idx[a:a + b].tolist(), idx[a + b:].tolist()   # train / val / remainder-as-test slices


def build_loss(cfg):
    if cfg["loss_fn"] == "mse":
        return nn.MSELoss(reduction="none")     # per-element squared error, not yet averaged
    if cfg["loss_fn"] == "l1":
        return nn.L1Loss(reduction="none")      # per-element absolute error, not yet averaged
    return nn.HuberLoss(reduction="none", delta=cfg["huber_delta"])   # quadratic near zero, linear beyond delta


def to_device(inputs, target, device):
    a, n, ni, ci = inputs                                  # unpack the four CGCNN graph-batch tensors
    return (a.to(device), n.to(device), ni.to(device),     # move atom/bond/neighbour-index tensors to the target device
            [i.to(device) for i in ci], target.to(device))  # crystal_atom_idx is a list of tensors, move each; move target too


def epoch_pass(loader, model, criterion, normalizer, head_w, n_cols,
               optimizer=None, device=torch.device("cpu")):
    """One pass. head_w selects WHICH heads contribute loss this stage."""
    training = optimizer is not None              # an optimizer was passed in => this is a training pass, not evaluation
    model.train() if training else model.eval()   # switch dropout/batchnorm to train or eval behaviour accordingly
    tot, seen, preds, trues, ids = 0.0, 0, [], [], []    # running loss sum, sample count, and per-batch accumulators
    with torch.enable_grad() if training else torch.no_grad():   # only track gradients during training
        for inputs, target, batch_ids in loader:          # one mini-batch at a time
            ids.extend(batch_ids)                          # accumulate crystal ids across all batches
            a, n, ni, ci, target = to_device(inputs, target, device)   # move everything to the training device
            target = target[:, :n_cols]        # drop the trailing weight column
            out = model(a, n, ni, ci)[:, :n_cols]   # forward pass, keep only the columns this stage supervises
            loss = (criterion(out, normalizer.norm(target)) * head_w).mean()   # per-head loss, weighted, then averaged
            if training:
                optimizer.zero_grad()    # clear gradients from the previous step
                loss.backward()          # backpropagate this batch's loss
                optimizer.step()         # apply the optimizer update
            bs = target.size(0)                  # number of crystals in this batch
            tot += loss.item() * bs              # accumulate the batch's total (not mean) loss
            seen += bs                           # accumulate total crystals seen
            preds.append(normalizer.denorm(out.detach()).cpu().numpy())    # de-normalized predictions, detached from the graph
            trues.append(target.detach().cpu().numpy())                   # true targets, detached from the graph
    return tot / seen, np.concatenate(preds), np.concatenate(trues), ids   # mean loss, stacked predictions, stacked truths, ids


def kappa_full(log_K, log_G, gamma, volume_m3, n_sites, density_g_cm3):
    """Full Slack kappa in W/m/K. Density and volume do NOT cancel when the
    yardstick is an absolute kappa, so they are carried explicitly."""
    K = (10.0 ** np.asarray(log_K, float)) * 1e9    # log10(GPa) -> Pa
    G = (10.0 ** np.asarray(log_G, float)) * 1e9    # log10(GPa) -> Pa
    rho = np.asarray(density_g_cm3, float) * 1e3    # g/cm^3 -> kg/m^3
    with np.errstate(all="ignore"):                 # suppress numpy warnings from invalid/negative intermediate values
        vl = np.sqrt((K + 4.0 / 3.0 * G) / rho)     # longitudinal sound velocity
        vt = np.sqrt(G / rho)                       # transverse sound velocity
        vs = ((1.0 / vl ** 3 + 2.0 / vt ** 3) / 3.0) ** (-1.0 / 3.0)   # averaged sound velocity
        return (G * vs * np.asarray(volume_m3, float) ** (1.0 / 3.0)    # Slack model numerator
                / (np.asarray(n_sites, float) * 300.0)                   # normalised by atom count and 300 K
                * np.exp(-np.asarray(gamma, float)))                     # anharmonic (Gruneisen) suppression factor


def derived_gamma(log_K, log_G):
    with np.errstate(all="ignore"):    # suppress numpy warnings from invalid intermediate values
        x2 = (10.0 ** np.asarray(log_K, float)) / (10.0 ** np.asarray(log_G, float)) + 4.0 / 3.0   # K/G ratio term
        nu = (x2 - 2.0) / (2.0 * x2 - 2.0)            # Poisson's ratio, derived from K/G
        return 3.0 * (1.0 + nu) / (2.0 * (2.0 - 3.0 * nu))   # empirical Gruneisen parameter from Poisson's ratio


def score_aflow(pred, true, ids, meta):
    """All three gamma variants on the same crystals, against AFLOW's kappa."""
    log_G_p = pred[:, 0]                       # predicted log10(G), column 0
    log_K_p = pred[:, 0] + pred[:, 1]          # predicted log10(K) = log10(G) + log10(K/G)
    m = meta.loc[ids]                          # align the metadata table to this batch's crystal order
    gam_true = m.gamma.values                  # AFLOW's own tabulated gamma for these crystals
    variants = {"derived": derived_gamma(log_K_p, log_G_p),   # gamma computed from predicted K/G via the empirical relation
                "predicted": 10.0 ** pred[:, 2],                # gamma straight from the model's third head
                "oracle": gam_true}                              # AFLOW's true gamma (an upper bound on what the head could do)
    out = {
        "mae_log_K": float(np.abs(log_K_p - (true[:, 0] + true[:, 1])).mean()),     # MAE of predicted vs true log10(K)
        "mae_log_G": float(np.abs(log_G_p - true[:, 0]).mean()),                    # MAE of predicted vs true log10(G)
        "mae_gamma": float(np.abs(variants["predicted"] - gam_true).mean()),        # MAE of the predicted-gamma head
        "gamma_corr": float(np.corrcoef(variants["predicted"], gam_true)[0, 1]),    # Pearson correlation, predicted vs true gamma
        "mae_gamma_derived": float(np.abs(variants["derived"] - gam_true).mean()),  # MAE of the derived-gamma baseline
    }
    kt = m.kappa_agl.values                     # AFLOW's own computed kappa, the scoring yardstick
    for name, gam in variants.items():          # score kappa under each of the three gamma variants
        kp = kappa_full(log_K_p, log_G_p, gam, m.volume_m3.values,
                        m.n_sites.values, m.density_g_cm3.values)   # kappa predicted using this gamma variant
        ok = np.isfinite(kp) & (kp > 0) & (kt > 0)    # mask out any non-physical or missing values before scoring
        out[f"kappa_mae_{name}"] = float(
            np.abs(np.log10(kp[ok]) - np.log10(kt[ok])).mean())     # MAE in log10 kappa space
        a, b = kt[ok], kp[ok]                         # true and predicted kappa, filtered to valid rows
        n10 = max(1, int(0.10 * len(a)))              # size of the bottom 10% bucket (at least 1 crystal)
        out[f"recall10_{name}"] = len(set(np.argsort(a)[:n10])            # true lowest-10% crystal indices
                                      & set(np.argsort(b)[:n10])) / n10    # fraction also predicted in the lowest 10%
    return out


def main():
    cfg = parse_overrides()                # CONFIG defaults merged with any CLI overrides
    torch.manual_seed(cfg["init_seed"])    # seed torch's RNG for reproducible weight init and dropout
    np.random.seed(cfg["init_seed"])       # seed numpy's global RNG too
    device = pick_device(cfg["device"])    # resolve "auto" to an actual torch.device
    out_dir = os.path.join(PROJECT_ROOT, cfg["out_dir"])   # absolute path to the results directory
    os.makedirs(out_dir, exist_ok=True)    # create it if it doesn't already exist
    criterion = build_loss(cfg)            # the loss function used by both stages

    # =====================================================================
    #  STAGE 1 - moduli on matbench
    # =====================================================================
    print("=" * 78)
    print("  STAGE 1: trunk + moduli heads on matbench")
    print("=" * 78)
    lab1 = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["s1_labels"]))    # matbench's K_VRH/G_VRH labels
    lab1 = lab1[(lab1.K_VRH > 0) & (lab1.G_VRH > 0)]     # drop any non-physical (zero or negative) modulus rows
    lK, lG = np.log10(lab1.K_VRH.values), np.log10(lab1.G_VRH.values)   # log10 of each modulus
    t1 = {i: (g, k - g) for i, k, g in zip(lab1.mb_id, lK, lG)}    # per-crystal target tuple: (log10 G, log10 K/G)
    ds1 = JointGraphCacheData(os.path.join(PROJECT_ROOT, cfg["s1_cache"]), t1)   # graph dataset paired with these targets
    tr1, va1, te1 = split_indices(len(ds1), cfg["train_ratio"],
                                  cfg["val_ratio"], cfg["split_seed"])    # deterministic train/val/test index split
    print(f"  {len(ds1)} crystals   split {len(tr1)}/{len(va1)}/{len(te1)}")

    lk1 = dict(batch_size=cfg["s1_batch_size"], collate_fn=collate_pool,
               num_workers=cfg["num_workers"])                  # shared DataLoader keyword args for stage 1
    ld1 = {k: DataLoader(ds1, sampler=SubsetRandomSampler(v), **lk1)
           for k, v in [("train", tr1), ("val", va1)]}           # one DataLoader per split, sampling only that split's indices

    # Matbench statistics for the two moduli columns. Kept and reused in
    # stage 2 - the frozen heads were trained against exactly these.
    mb_stats = VectorNormalizer(torch.stack([ds1[i][1] for i in tr1])[:, :2])          # mean/std of the moduli cols, train rows only
    norm1 = VectorNormalizer(torch.stack([ds1[i][1] for i in tr1])[:, :2]).to(device)  # a second copy, moved to the training device

    sample, _, _ = ds1[0]                     # peek at one dataset item to read off feature dimensions
    model = JointCrystalGraphConvNet(
        sample[0].shape[-1], sample[1].shape[-1],    # atom feature length, bond feature length
        atom_fea_len=cfg["atom_fea_len"], n_conv=cfg["n_conv"],
        h_fea_len=cfg["h_fea_len"], n_shared_fc=cfg["n_shared_fc"],
        n_head_fc=cfg["n_head_fc"], head_fea_len=cfg["head_fea_len"],
        dropout=cfg["dropout"], n_heads=3).to(device)   # build the 3-head model and move it to the training device
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params "
          f"(3 heads; the gamma head is untrained until stage 2)")

    # Stage 1 supervises only the first two columns; the gamma head is simply
    # not part of the loss and its weights stay at initialisation.
    hw1 = torch.tensor([1.0, 1.0], device=device)    # per-head loss weights: both moduli heads equally weighted
    opt1 = optim.Adam(model.parameters(), lr=cfg["s1_lr"],
                      weight_decay=cfg["s1_weight_decay"])   # optimizer over every parameter (gamma head included, unsupervised)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=cfg["s1_epochs"], eta_min=cfg["s1_lr"] * 0.01)   # cosine LR decay from s1_lr down to 1% of it

    best1, state1, bep1, stale = float("inf"), None, -1, 0   # best val score so far, its checkpoint, its epoch, patience counter
    start = time.time()               # stage 1 start time, for the wall-clock report below
    for ep in range(cfg["s1_epochs"]):             # one iteration per training epoch
        epoch_pass(ld1["train"], model, criterion, norm1, hw1, 2, opt1, device)   # one training pass over all train batches
        sch1.step()                                # advance the LR schedule by one epoch
        _, vp, vt, _ = epoch_pass(ld1["val"], model, criterion, norm1, hw1, 2,
                                  device=device)    # one evaluation pass over the validation set (no optimizer passed)
        # Select on the ratio error: it is what gamma depends on and what the
        # whole two-head design exists to improve.
        v = float(np.abs(vp[:, 1] - vt[:, 1]).mean())   # validation MAE on the log10(K/G) column specifically
        if v < best1 - 1e-5:                # meaningfully better than the best seen so far
            best1, bep1, stale = v, ep, 0   # record the new best score, epoch, and reset the patience counter
            state1 = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}   # snapshot the weights
        else:
            stale += 1                      # one more epoch without improvement
        if ep % cfg["print_every"] == 0 or ep == cfg["s1_epochs"] - 1:   # print on a schedule and on the final epoch
            print(f"  e{ep:3d}  val G {np.abs(vp[:,0]-vt[:,0]).mean():.4f}"
                  f"  K/G {v:.4f}")
        if cfg["s1_patience"] and stale >= cfg["s1_patience"]:   # no improvement for `patience` epochs in a row
            print(f"  early stop (best epoch {bep1})")
            break
    model.load_state_dict(state1)     # restore the best-validation checkpoint before moving to stage 2
    print(f"  stage 1 done in {(time.time()-start)/60:.1f} min, "
          f"best val K/G {best1:.4f} at epoch {bep1}")

    # =====================================================================
    #  STAGE 2 - gamma head on AFLOW, trunk frozen
    # =====================================================================
    print()
    print("=" * 78)
    print(f"  STAGE 2: gamma head on AFLOW  (freeze_moduli={cfg['freeze_moduli']})")
    print("=" * 78)
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["s2_labels"])).set_index("gid")   # AFLOW labels, indexed by crystal id
    aK, aG = np.log10(meta.K_VRH.values), np.log10(meta.G_VRH.values)   # log10 of AFLOW's own K and G
    aGam = np.log10(meta.gamma.values)     # log10 of AFLOW's own tabulated gamma
    t2 = {i: (g, k - g, gm) for i, k, g, gm in zip(meta.index, aK, aG, aGam)}   # per-crystal target triple: (G, K/G, gamma)
    ds2 = JointGraphCacheData(os.path.join(PROJECT_ROOT, cfg["s2_cache"]), t2)   # AFLOW graph dataset paired with these targets
    tr2, va2, te2 = split_indices(len(ds2), cfg["train_ratio"],
                                  cfg["val_ratio"], cfg["split_seed"])   # deterministic train/val/test split for AFLOW
    print(f"  {len(ds2)} crystals   split {len(tr2)}/{len(va2)}/{len(te2)}")

    lk2 = dict(batch_size=cfg["s2_batch_size"], collate_fn=collate_pool,
               num_workers=cfg["num_workers"])                  # shared DataLoader keyword args for stage 2
    ld2 = {k: DataLoader(ds2, sampler=SubsetRandomSampler(v), **lk2)
           for k, v in [("train", tr2), ("val", va2), ("test", te2)]}   # one DataLoader per split, this time including test

    # THE COMPOSITE NORMALISER. Columns 0-1 keep matbench's statistics because
    # the heads that produce them are frozen and were trained against those.
    # Column 2 takes AFLOW's, the only place gamma exists. Refitting 0-1 on
    # AFLOW would silently rescale the frozen heads and undo stage 1.
    gam_train = torch.stack([ds2[i][1] for i in tr2])[:, 2:3]    # just the gamma column, train rows only
    gstats = VectorNormalizer(gam_train)         # mean/std of log10(gamma) on AFLOW's training crystals
    norm2 = VectorNormalizer(torch.zeros(2, 3))  # placeholder 3-column normalizer, values overwritten below
    norm2.mean = torch.cat([mb_stats.mean, gstats.mean])   # columns 0-1 from matbench, column 2 from AFLOW
    norm2.std = torch.cat([mb_stats.std, gstats.std])      # same composition for the standard deviations
    norm2 = norm2.to(device)     # move the composite normalizer to the training device
    print(f"  normaliser  log10(G)     mean={norm2.mean[0]:.3f} (matbench)")
    print(f"              log10(K/G)   mean={norm2.mean[1]:.3f} (matbench)")
    print(f"              log10(gamma) mean={norm2.mean[2]:.3f} (AFLOW)")

    if cfg["freeze_moduli"]:              # stage-2 mode: freeze everything except the gamma head
        trainable = []                    # names of parameters left trainable, for the printed count below
        for name, p in model.named_parameters():   # walk every named parameter tensor in the model
            # heads.2 is the gamma head; head0/head1 are aliases of heads.0/1
            # and share storage, so freezing by name on "heads.2" is enough.
            p.requires_grad = name.startswith("heads.2")   # only gamma-head parameters keep gradients
            if p.requires_grad:
                trainable.append(name)    # record this parameter's name as still-trainable
        print(f"  frozen everything except the gamma head "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
              f"trainable of {sum(p.numel() for p in model.parameters()):,})")
    else:                                  # stage-2 mode: adapt the whole model, not just gamma
        for p in model.parameters():
            p.requires_grad = True        # re-enable gradients on every parameter
        print("  NOTHING frozen - moduli heads will adapt to AFLOW's convention")

    hw2 = torch.tensor([0.0, 0.0, 1.0] if cfg["freeze_moduli"]            # frozen: only the gamma column contributes loss
                       else [1.0, 1.0, 1.0], device=device)                 # unfrozen: all three columns contribute loss
    params = [p for p in model.parameters() if p.requires_grad]    # only the parameters this stage is allowed to update
    opt2 = optim.Adam(params, lr=cfg["s2_lr"], weight_decay=cfg["s2_weight_decay"])   # optimizer over just those parameters
    sch2 = optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg["s2_epochs"], eta_min=cfg["s2_lr"] * 0.01)   # cosine LR decay for stage 2

    best2, state2, bep2, stale = float("inf"), None, -1, 0   # best val kappa MAE so far, its checkpoint, epoch, patience counter
    start = time.time()               # stage 2 start time
    for ep in range(cfg["s2_epochs"]):
        epoch_pass(ld2["train"], model, criterion, norm2, hw2, 3, opt2, device)   # one training pass over all 3 columns
        sch2.step()                                # advance stage 2's LR schedule
        _, vp, vt, vi = epoch_pass(ld2["val"], model, criterion, norm2, hw2, 3,
                                   device=device)   # evaluation pass over the validation set
        s = score_aflow(vp, vt, vi, meta)          # full scoring dict (moduli MAE, gamma variants, kappa MAE/recall)
        if s["kappa_mae_predicted"] < best2 - 1e-5:   # select the checkpoint on predicted-gamma kappa error, the real target
            best2, bep2, stale = s["kappa_mae_predicted"], ep, 0
            state2 = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}   # snapshot the weights
        else:
            stale += 1
        if ep % cfg["print_every"] == 0 or ep == cfg["s2_epochs"] - 1:
            print(f"  e{ep:3d}  val gamma MAE {s['mae_gamma']:.4f}"
                  f"  kappa {s['kappa_mae_predicted']:.4f}")
        if cfg["s2_patience"] and stale >= cfg["s2_patience"]:
            print(f"  early stop (best epoch {bep2})")
            break
    model.load_state_dict(state2)      # restore the best-validation checkpoint for the final test evaluation

    _, tp, tt, ti = epoch_pass(ld2["test"], model, criterion, norm2, hw2, 3,
                               device=device)   # one evaluation pass over the held-out test set
    t = score_aflow(tp, tt, ti, meta)  # final scoring dict on the test set

    print()
    print("=" * 78)
    print(f"  TEST ({len(te2)} AFLOW crystals) - kappa against AFLOW's own kappa")
    print("=" * 78)
    print(f"  MAE log10(K) {t['mae_log_K']:.4f}   log10(G) {t['mae_log_G']:.4f}")
    print(f"  gamma  derived MAE {t['mae_gamma_derived']:.4f}"
          f"   PREDICTED MAE {t['mae_gamma']:.4f}  corr {t['gamma_corr']:+.4f}")
    print()
    print(f"  {'kappa source':<14}{'MAE log10':>11}{'recall@10%':>13}")
    for k in ("derived", "predicted", "oracle"):   # print one row per gamma variant
        print(f"  {k:<14}{t['kappa_mae_' + k]:11.4f}{100 * t['recall10_' + k]:12.1f}%")
    print()
    print("  single-stage (37_train_gamma) for comparison:")
    print("    K 0.1817  G 0.1898 | gamma pred 0.1923 | kappa der 0.3585 "
          "pred 0.3408 oracle 0.2979")

    torch.save({"state_dict": state2, "config": cfg,             # the trained weights and the exact config used to produce them
                "normalizer": norm2.state_dict(),                 # so predictions can be de-normalized later without retraining
                "split_s2": {"train": tr2, "val": va2, "test": te2}},   # which AFLOW crystals were in which split
               os.path.join(out_dir, f"model_38_{cfg['tag']}.pth"))
    with open(os.path.join(out_dir, f"summary_38_{cfg['tag']}.json"), "w") as fh:
        json.dump({"tag": cfg["tag"], "config": cfg,
                   "stage1_best_epoch": bep1, "stage1_best_val_ratio": best1,
                   "stage2_best_epoch": bep2, "test": t}, fh, indent=2)   # write everything needed to audit this run later
    print(f"\n  saved model_38_{cfg['tag']}.pth and summary_38_{cfg['tag']}.json")


if __name__ == "__main__":   # only run main() when this file is executed directly, not when imported
    main()
