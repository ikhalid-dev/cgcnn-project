#!/usr/bin/env python3
"""
STEP 31 - Train the JOINT two-head CGCNN (shared trunk, K and G together).
===========================================================================

WHAT THIS REPLACES
------------------
02_train.py trains ONE network for ONE modulus, and you run it twice:

    python scripts/cgcnn/02_train.py --target K_VRH
    python scripts/cgcnn/02_train.py --target G_VRH

That produces two networks whose mistakes are unrelated to each other. This
script trains ONE network that emits both numbers from a shared trunk:

    python scripts/cgcnn/31_train_joint.py

WHY IT SHOULD BE BETTER (measured, not assumed)
-----------------------------------------------
The Slack model's Grueneisen parameter depends ONLY on the ratio K/G, and in
log space that ratio is a DIFFERENCE:  log10(K/G) = log10(K) - log10(G).
Two independent models contribute two independent errors to that difference.
On the current 1,648-crystal test set, with the existing separate models:

    residual correlation between err_K and err_G  = 0.297
    MAE log10(K)    = 0.0630
    MAE log10(G)    = 0.0781
    MAE log10(K/G)  = 0.0887   <- worse than EITHER individual error

    kappa_L error attributable to the moduli:
        total       = 0.1897 log10
          prefactor = 0.1160
          exp(-gam) = 0.0911
    if the two errors were perfectly correlated instead:
        total       = 0.0882 log10   <- 54% of the error is pure decorrelation

So the target of this script is NOT a better log10(K). It is a better
log10(K/G) and a higher residual correlation. Those are printed every epoch.

TWO IDEAS, INDEPENDENTLY SWITCHABLE
-----------------------------------
  1. SHARED TRUNK - always on in this script; it is the architecture.
  2. RATIO REPARAMETERISATION - TARGET_MODE below. Set "G_and_ratio" to have
     the heads predict log10(G) and log10(K/G); set "K_and_G" to have them
     predict log10(K) and log10(G) as before.

Run it both ways to separate the two effects. "K_and_G" is the honest
ablation: same trunk, same loss, same regularisation, only the output
parameterisation differs.

HOW TO USE THIS FILE
--------------------
Every tunable lives in the CONFIG block directly below. Edit it in place for
an experiment, or override any single value from the command line - each
CONFIG key has a matching --flag with the same name in lower-kebab-case:

    python scripts/cgcnn/31_train_joint.py --target-mode K_and_G --tag ablation
    python scripts/cgcnn/31_train_joint.py --dropout 0.15 --learning-rate 0.005

OUTPUTS (all into OUT_DIR, suffixed with TAG)
---------------------------------------------
    model_31_<TAG>.pth        weights + normalizer + split + config
    predictions_31_<TAG>.csv  per-crystal predictions, train+val+test
    history_31_<TAG>.csv      per-epoch metrics, for plotting curves
    summary_31_<TAG>.json     final test metrics + the kappa error budget

    Filenames carry this script's own number (31) rather than "joint" so
    that another script's output sitting in the same results/ directory
    can always be told apart by name alone.
"""

# =============================================================================
#  CONFIG - every tunable parameter lives here. Edit freely.
# =============================================================================
# Each entry says what the knob does and which way to move it. Anything not in
# this dict is a structural choice, not a hyperparameter.

CONFIG = {

    # ---- DATA & OUTPUT ------------------------------------------------------
    # Directory holding labels.csv and the pre-built graphs.pt cache.
    "data_dir": "data_full",
    # Which labels/cache to read from data_dir. Point these at the merged pair
    # (labels_merged.csv / graphs_merged.pt, built by 35_merge_aflow.py) to
    # train with the AFLOW soft-material augmentation. Crystals whose source is
    # not "matbench" are forced into TRAIN, so the 1,648-crystal test set is
    # bit-identical either way and every earlier round stays comparable.
    "labels_file": "labels.csv",
    "cache_file": "graphs.pt",
    # Where checkpoints, history and summaries are written.
    "out_dir": "results/cgcnn",
    # Suffix for every output file. Change this per experiment or you will
    # silently overwrite the previous run's checkpoint.
    "tag": "joint",

    # ---- WHAT THE TWO HEADS PREDICT ----------------------------------------
    # "G_and_ratio" -> head0 = log10(G), head1 = log10(K/G)   [the new idea]
    # "K_and_G"     -> head0 = log10(K), head1 = log10(G)     [ablation baseline]
    # Both are scored on the same physical quantities, so the numbers are
    # directly comparable. This is the single most important switch here.
    "target_mode": "G_and_ratio",

    # ---- ARCHITECTURE -------------------------------------------------------
    # Hidden width carried by each atom inside the convolutions. Bigger = more
    # capacity per atom, and more overfitting on 7.7k training crystals.
    "atom_fea_len": 64,
    # Message-passing rounds. Each round widens an atom's view by one bond hop.
    # The old ensemble used 3 and 4. Beyond ~5 graphs tend to over-smooth:
    # every atom converges to the same vector and the model loses local detail.
    "n_conv": 3,
    # Width of the SHARED fully-connected stack after pooling.
    "h_fea_len": 128,
    # Extra shared FC layers after conv_to_fc. More shared depth = more of the
    # computation is common to both heads = errors correlate harder. This is a
    # lever on the actual hypothesis, not just capacity.
    "n_shared_fc": 1,
    # Hidden FC layers INSIDE each head. Keep small (0 or 1). Large heads let
    # each branch specialise and re-decorrelate the errors, which is exactly
    # what we are trying to prevent.
    "n_head_fc": 1,
    # Width of those per-head hidden layers.
    "head_fea_len": 64,
    # Dropout on the SHARED embedding, at the fork. The original
    # CrystalGraphConvNet only applies dropout when classification=True, so the
    # regression path has never had any - and it shows as a 2.0-2.4x train/test
    # MAE gap. Try 0.0 / 0.1 / 0.2. Applied before the fork so both heads see
    # the same perturbation and stay correlated.
    "dropout": 0.10,

    # Fraction of each atom's neighbour messages randomly dropped every forward
    # pass while training (DropEdge). This is GRAPH AUGMENTATION: the graph
    # cache is fixed, so without it the network sees byte-identical input for a
    # crystal all 150 epochs and has every incentive to memorise - which the
    # 2.0-2.4x train/test gap says it does. Try 0.0 / 0.1 / 0.2. Costs no
    # parameters and nothing at inference (skipped in eval mode).
    # NOTE this is the cheap augmentation. Thermal jitter of atomic positions
    # is better motivated (Srivastava 2024: thermally-populated configurations
    # gave 10x better cubic force constants than finite-difference ones) but
    # needs raw bond distances, which graphs.pt does not keep.
    "edge_dropout": 0.0,

    # Upweight the LOW-kappa tail in the loss. Stratifying the test set by true
    # kappa shows the lowest quintile is predicted 2.5x worse than the rest
    # (MAE K 0.1017 vs 0.0404) because those crystals are SOFT - median shear
    # modulus 10.5 GPa against 102 GPa at the stiff end - and soft crystals are
    # rare in matbench, so an unweighted loss ignores them. That quintile is
    # the ONLY one a low-kappa screen cares about.
    # Weight scales as kappa^(-alpha). 0.0 disables it and reproduces every
    # earlier run exactly; 0.3-0.5 is the first sweep worth running. Larger
    # values chase the tail harder and start costing accuracy on the bulk.
    # Same trick Guo 2023 used for phonon scattering rates (w = Gamma^0.4).
    "kappa_weight_alpha": 0.0,

    # ---- LOSS ---------------------------------------------------------------
    # "mse"   - what 02_train.py used. Squares errors, so a few bad crystals
    #           dominate the gradient.
    # "l1"    - optimises exactly the MAE we report. Usually the right default
    #           when MAE is the headline metric.
    # "huber" - L2 near zero, L1 in the tail. Robust to label outliers while
    #           staying smooth at the optimum. Good middle ground.
    "loss_fn": "huber",
    # Transition point for huber, in NORMALISED target units (targets are
    # z-scored, so ~1.0 is one standard deviation). Smaller = more L1-like.
    "huber_delta": 1.0,
    # Relative weight of each head in the total loss. Raise head1's weight to
    # push the model to care more about the ratio (and therefore gamma) at the
    # cost of absolute accuracy on head0. 1.0 / 1.0 is the neutral starting
    # point; 1.0 / 2.0 is worth trying once the baseline works.
    "loss_weight_head0": 1.0,
    "loss_weight_head1": 1.0,

    # ---- OPTIMISATION -------------------------------------------------------
    # "adam"  - what 02_train.py used. NOTE: torch's Adam implements
    #           weight_decay as L2 ADDED TO THE GRADIENT, which then gets
    #           divided by Adam's per-parameter running variance. Parameters
    #           with large gradients have their decay scaled away, so the
    #           effective regularisation is both weaker and less uniform than
    #           the number suggests. Raising weight_decay under plain Adam
    #           therefore does less than it looks.
    # "adamw" - decoupled weight decay: applied directly to the weights,
    #           independent of the adaptive step. This is what the number is
    #           supposed to mean, and what the project's own ALIGNN config
    #           already uses (results/alignn/*/config.json -> "adamw").
    # Default stays "adam" ONLY so this run stays comparable with the 12-run
    # grid already on Kaggle; "adamw" is in that grid's sweep to settle it.
    "optimizer": "adam",
    "epochs": 200,
    # 128 matched the old runs. Smaller batches add gradient noise, which is
    # mild regularisation; larger batches are faster per epoch on a GPU.
    "batch_size": 128,
    "learning_rate": 0.01,
    # L2 penalty. The old runs used 1e-5, which is close to nothing - one
    # reason for the train/test gap. 1e-4 is a reasonable next rung.
    "weight_decay": 1e-4,
    # "cosine"  - glide the LR to ~0 over the run. Best for a fixed budget.
    # "plateau" - reactive: cut the LR after PLATEAU_PATIENCE stagnant epochs.
    # "onecycle"- warm up then anneal; what the ALIGNN config uses.
    "scheduler": "cosine",
    # cosine only: final LR as a fraction of the initial one.
    "cosine_eta_min_frac": 0.01,
    # plateau only.
    "plateau_factor": 0.5,
    "plateau_patience": 20,
    # Clip gradient norm to this. 0 disables. Cheap insurance against the
    # occasional exploding batch; does nothing in a healthy run.
    "grad_clip": 0.0,

    # ---- STOCHASTIC WEIGHT AVERAGING ----------------------------------------
    # Average the weights over the final fraction of training instead of taking
    # a single point. Validation curves on 7.7k crystals are noisy, so the
    # single "best" epoch is partly luck; averaging lands in a flatter, wider
    # minimum that generalises slightly better. Nearly free - one extra copy of
    # the weights and one pass over the training set at the end to recompute
    # BatchNorm statistics.
    #
    # 0.0 disables it. 0.25 averages over the last quarter of the epochs.
    # When enabled, BOTH the best single checkpoint and the SWA average are
    # scored on validation at the end and the better one is kept, so turning
    # this on can never make the saved model worse.
    "swa_start_frac": 0.0,

    # ---- EARLY STOPPING -----------------------------------------------------
    # The old runs trained a flat 200 epochs while validation stopped improving
    # around epoch 50-130 - all that did was fit training noise. Stop after
    # this many epochs without a new best. Set 0 to disable and run the full
    # EPOCHS.
    "early_stop_patience": 40,
    # How much better a score has to be to count as "improved", in the units of
    # SELECT_METRIC. Guards against declaring victory on numerical noise.
    "early_stop_min_delta": 1e-4,
    # Which validation number decides "best" and drives early stopping:
    #   "kappa"    - MAE of log10(kappa_L) from the moduli. The actual
    #                downstream objective, so the honest default.
    #   "ratio"    - MAE of log10(K/G). Sharpest view of the gamma error.
    #   "mean_mae" - plain mean of MAE log10(K) and MAE log10(G). Closest to
    #                what the old single-target scripts optimised.
    "select_metric": "kappa",

    # ---- SPLIT --------------------------------------------------------------
    # 70/15/15, matching 02_train.py exactly. The remainder is the test set.
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    # Seeds the TRAIN/VAL/TEST SPLIT. Must stay 42 to compare against the
    # existing separate-model results, which all used split_seed 42.
    "split_seed": 42,
    # Seeds WEIGHT INIT and batch shuffling. Vary this (and only this) to build
    # an ensemble whose members share one held-out test set.
    "init_seed": 42,

    # ---- RUNTIME ------------------------------------------------------------
    # "auto" picks cuda, then mps, then cpu.
    "device": "auto",
    # DataLoader worker processes. 0 is fastest on a 2-core laptop; use 2-4 on
    # a GPU box where collating becomes the bottleneck.
    "num_workers": 0,
    # Print a metrics line every N epochs (the best epoch always prints).
    "print_every": 10,
}

# =============================================================================
#  END OF CONFIG - implementation below
# =============================================================================

import argparse
import json
import os
import sys
import time
import warnings

# torch MUST be imported before numpy/pandas/pymatgen in this conda env: MKL
# loads its own OpenMP runtime first and the duplicate libiomp5 segfaults
# torch. Do not "tidy" these imports into alphabetical order.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence deprecation/runtime warnings (pandas, pymatgen, etc.) from stdout

# Make `cgcnn_scratch` importable no matter which directory the script is run
# from: walk up three levels from this file to the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # this_file/../../.. resolved to an absolute path
sys.path.insert(0, PROJECT_ROOT)  # put PROJECT_ROOT first on the import search path

from cgcnn_scratch.data import collate_pool                      # noqa: E402  # batches a list of (graph, target, id) samples into padded tensors
from cgcnn_scratch.joint import (                                # noqa: E402
    JointCrystalGraphConvNet, VectorNormalizer, load_joint_dataset,  # model class, target z-scorer, and dataset loader
    KG_from_targets, kappa_error_budget, kappa_quintile_breakdown,   # target-mode un-parameterisation + scoring helpers
    TARGET_MODES,                                                    # tuple of valid strings for CONFIG["target_mode"]
)


# -----------------------------------------------------------------------------
#  Command-line overrides
# -----------------------------------------------------------------------------
def parse_overrides():
    """Build an argparse flag for every CONFIG key and apply any that are set.

    The flag name is the CONFIG key with underscores turned into dashes, and
    the type is inferred from the default value. This means CONFIG stays the
    single source of truth: adding a knob there automatically gives it a CLI
    flag, with no second list to keep in sync.
    """
    parser = argparse.ArgumentParser(                                    # builds the CLI parser
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # --help prints this module's docstring verbatim, unwrapped

    for key, default in CONFIG.items():                # one flag per CONFIG entry
        # bools would need store_true/store_false handling; there are none in
        # CONFIG today, so infer the type straight from the default.
        arg_type = type(default)                        # e.g. str, int, float - argparse casts the CLI string to this
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,       # "learning_rate" -> flag "--learning-rate", stored back under key "learning_rate"
                            type=arg_type, default=None,                  # default None (not CONFIG's default) so "did the user pass this?" is detectable below
                            help=f"override CONFIG['{key}'] (default: {default!r})")

    args = parser.parse_args()                          # parses sys.argv into a Namespace object

    cfg = dict(CONFIG)  # copy so the module-level dict stays pristine
    for key, value in vars(args).items():    # vars() turns the Namespace into a {flag_name: value} dict
        if value is not None:                # None means "flag not passed" - CONFIG's own default should stand
            cfg[key] = value                 # CLI value overrides the CONFIG default for this key

    assert cfg["target_mode"] in TARGET_MODES, \
        f"target_mode must be one of {TARGET_MODES}, got {cfg['target_mode']!r}"
    return cfg                                # the merged CONFIG+CLI-overrides dict every other function reads from


# -----------------------------------------------------------------------------
#  Small helpers
# -----------------------------------------------------------------------------
def pick_device(requested):
    """Resolve "auto" to the best backend actually present on this machine."""
    if requested != "auto":                        # explicit device string ("cpu"/"cuda"/"mps") passed through as-is
        return torch.device(requested)
    if torch.cuda.is_available():                  # True only if an NVIDIA GPU + working CUDA build are present
        return torch.device("cuda")
    # macOS 12 + torch 2.2 usually reports False here, but on a newer Mac this
    # is a large win over CPU.
    if torch.backends.mps.is_available():          # True on Apple Silicon with a torch build that supports Metal
        return torch.device("mps")
    return torch.device("cpu")                     # fallback: always available


def split_indices(n_total, train_ratio, val_ratio, seed):
    """Split [0, n_total) into train / val / test index lists.

    Identical logic to 02_train.py, including the RandomState seeding, so that
    split_seed=42 here selects exactly the same test crystals as the existing
    separate-model runs. Without that the comparison would be meaningless.
    """
    rng = np.random.RandomState(seed)                       # seeded PRNG, deterministic given `seed`
    indices = rng.permutation(n_total)                      # shuffled array of the integers 0..n_total-1
    n_train = int(round(train_ratio * n_total))             # how many of those go to train
    n_val = int(round(val_ratio * n_total))                 # how many go to val; the rest (n_total - n_train - n_val) is test
    return (indices[:n_train].tolist(),                     # first n_train shuffled indices -> train
            indices[n_train:n_train + n_val].tolist(),      # next n_val shuffled indices -> val
            indices[n_train + n_val:].tolist())              # remaining indices -> test


def build_loss(cfg):
    """Return the per-element loss function named by CONFIG['loss_fn'].

    reduction="none" throughout: we need the per-sample, per-head values so the
    two heads can be weighted separately before anything is averaged.
    """
    name = cfg["loss_fn"]                              # one of "mse" / "l1" / "huber", read from CONFIG
    if name == "mse":
        return nn.MSELoss(reduction="none")            # per-element squared error, no averaging done inside the loss
    if name == "l1":
        return nn.L1Loss(reduction="none")              # per-element absolute error
    if name == "huber":
        return nn.HuberLoss(reduction="none", delta=cfg["huber_delta"])  # quadratic below delta, linear beyond it
    raise ValueError(f"unknown loss_fn {name!r}; expected mse, l1 or huber")


def to_device(inputs, target, device):
    """Move one collated batch onto `device`.

    crystal_atom_idx is a LIST of index tensors (one per crystal), so it needs
    mapping element by element rather than a single .to() call.
    """
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs   # unpack the 4-tuple collate_pool produces
    return (atom_fea.to(device, non_blocking=True),             # per-atom feature matrix, moved to `device`
            nbr_fea.to(device, non_blocking=True),              # per-bond Gaussian-expanded distance features
            nbr_fea_idx.to(device, non_blocking=True),          # neighbour-index tensor (which atoms each bond connects)
            [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],  # one index tensor per crystal, moved individually
            target.to(device, non_blocking=True))               # the (B, 3) target+weight tensor, moved to `device`


# -----------------------------------------------------------------------------
#  One pass over a loader
# -----------------------------------------------------------------------------
def run_epoch(loader, model, criterion, normalizer, cfg,
              optimizer=None, device=torch.device("cpu")):
    """Run one epoch. Trains if an optimizer is passed, otherwise evaluates.

    Two different spaces are in play, and mixing them up is the classic bug:
      * the LOSS is computed in NORMALISED space - that is what the network
        outputs and what gradients flow through;
      * the METRICS are computed in physical log10 space, because those are the
        numbers we compare against the paper and against the old models.

    Returns
    -------
    (mean_loss, metrics_dict, pred_log_K, pred_log_G, true_log_K, true_log_G)
    The four arrays are returned so the caller can compute the kappa error
    budget without a second forward pass.
    """
    training = optimizer is not None                 # True in train mode, False when just evaluating a loader
    model.train() if training else model.eval()      # toggles dropout/BatchNorm behaviour to match the mode

    total_loss, n_seen = 0.0, 0                       # running sum of (loss * batch_size) and total samples, for the epoch mean
    pred_chunks, true_chunks, id_chunks = [], [], []  # accumulate per-batch arrays to concatenate once at the end

    # One context manager for both phases, so the loop body is written once.
    with torch.enable_grad() if training else torch.no_grad():   # tracks gradients only when training
        for inputs, target, batch_ids in loader:                  # one mini-batch per iteration
            id_chunks.extend(batch_ids)                            # remember which crystal each row in this batch belongs to
            (atom_fea, nbr_fea, nbr_fea_idx,
             crystal_atom_idx, target) = to_device(inputs, target, device)  # move the whole batch onto `device`

            # target arrives as (B, 3): two targets plus the per-crystal loss
            # weight. Split them before anything else touches the tensor.
            sample_w = target[:, 2:3]          # (B, 1), broadcasts over heads
            target = target[:, :2]             # (B, 2) physical log10 units
            target_normed = normalizer.norm(target)   # z-scored targets, the space the network is trained in

            # (B, 2): column 0 = head0, column 1 = head1.
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)   # forward pass through the joint CGCNN

            # Per-element loss, then weight the two heads and average.
            per_element = criterion(output, target_normed)     # (B, 2)
            head_w = torch.tensor(
                [cfg["loss_weight_head0"], cfg["loss_weight_head1"]],
                device=per_element.device, dtype=per_element.dtype)   # (2,) tensor, one scalar weight per head
            # Two independent weightings multiply: head_w says which OUTPUT
            # matters more, sample_w says which CRYSTAL matters more.
            loss = (per_element * head_w * sample_w).mean()    # broadcast-multiply then reduce to one scalar

            if training:
                optimizer.zero_grad()           # clear gradients left over from the previous batch
                loss.backward()                 # backprop: populates .grad on every trainable parameter
                if cfg["grad_clip"] > 0:
                    # Rescale the whole gradient if its norm exceeds the clip,
                    # so direction is preserved and only the step size is cut.
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()                # apply one optimiser update using the just-computed gradients

            batch_size = target.size(0)                     # number of crystals in this batch
            total_loss += loss.item() * batch_size          # de-average this batch's loss so the running sum is exact
            n_seen += batch_size                             # tally total crystals seen this epoch

            # Back to physical log10 units for the metrics.
            pred_chunks.append(normalizer.denorm(output.detach()).cpu().numpy())  # undo z-scoring, detach from autograd, move to a numpy array
            true_chunks.append(target.detach().cpu().numpy())                     # same treatment for the true targets

    pred = np.concatenate(pred_chunks, axis=0)   # stack every batch's predictions into one (N, 2) array
    true = np.concatenate(true_chunks, axis=0)   # same for the true targets
    # ids are collected in the SAME order as the prediction rows. The loaders
    # use SubsetRandomSampler, so that order is shuffled and differs between
    # epochs - without carrying the ids along there is no way to line the rows
    # up with a crystal, or with another ensemble member's rows.

    # Undo the parameterisation so both modes are scored on the SAME two
    # physical quantities. Without this step "G_and_ratio" and "K_and_G" would
    # be reporting different things and could not be compared.
    pred_log_K, pred_log_G = KG_from_targets(pred[:, 0], pred[:, 1], cfg["target_mode"])   # map (head0, head1) predictions back to (log10 K, log10 G)
    true_log_K, true_log_G = KG_from_targets(true[:, 0], true[:, 1], cfg["target_mode"])   # same un-parameterisation for the true targets

    metrics = kappa_error_budget(true_log_K, true_log_G, pred_log_K, pred_log_G)   # dict of MAEs, correlation, and the kappa error breakdown
    return (total_loss / n_seen, metrics,               # mean loss for the epoch, plus the metrics dict
            pred_log_K, pred_log_G, true_log_K, true_log_G, id_chunks)   # raw arrays the caller needs for further analysis


def update_bn_stats(loader, model, device):
    """Recompute BatchNorm running statistics for an SWA-averaged model.

    Averaging weights does NOT average BatchNorm's running mean/var - those are
    buffers updated by forward passes, not parameters touched by the optimiser.
    An SWA model therefore carries whichever statistics the last member of the
    average happened to leave behind, which do not match the averaged weights
    and can wreck the predictions. The fix is one extra pass over the training
    data in train() mode with the statistics reset.

    torch.optim.swa_utils.update_bn cannot be used directly here: it assumes a
    loader yielding plain input tensors, and collate_pool yields a 4-tuple plus
    targets and ids. Hence this hand-rolled version.
    """
    bn_modules = [m for m in model.modules()                        # model.modules() walks every submodule recursively
                  if isinstance(m, nn.modules.batchnorm._BatchNorm)]  # keep only BatchNorm layers (there may be none)
    if not bn_modules:
        return                                                       # nothing to recompute, exit early

    # momentum=None makes BatchNorm accumulate a CUMULATIVE average over the
    # pass rather than an exponentially-weighted one, so the statistics reflect
    # the whole training set evenly instead of being dominated by the last few
    # batches.
    saved_momenta = {}                             # remember each module's original momentum so it can be restored after
    for module in bn_modules:
        module.reset_running_stats()               # zero out the running mean/var buffers
        saved_momenta[module] = module.momentum    # stash the original momentum value
        module.momentum = None                     # switch this module to cumulative-average mode

    was_training = model.training     # remember the model's mode so it can be restored at the end
    model.train()                     # BatchNorm only updates running stats in train mode
    with torch.no_grad():             # no gradients needed, this is a stats-only pass
        for inputs, target, _ in loader:            # iterate every batch of the training set once
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, _ = to_device(
                inputs, target, device)              # move the batch to `device` (target's device-copy is unused here)
            model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)   # forward pass only, to update BN running stats as a side effect

    for module, momentum in saved_momenta.items():
        module.momentum = momentum      # restore each module's original momentum
    model.train(was_training)           # restore the model's original train/eval mode


def selection_score(metrics, which):
    """Collapse a metrics dict to the single number that defines 'best'.

    Lower is better in every case, so the caller can always minimise.
    """
    if which == "kappa":
        return metrics["kappa_mae_total"]                                # MAE of log10(kappa_L), the downstream objective
    if which == "ratio":
        return metrics["mae_log_ratio"]                                  # MAE of log10(K/G) only
    if which == "mean_mae":
        return 0.5 * (metrics["mae_log_K"] + metrics["mae_log_G"])       # unweighted average of the two individual MAEs
    raise ValueError(f"unknown select_metric {which!r}")


def format_metrics(prefix, metrics):
    """One compact, aligned line of the numbers that actually matter."""
    return (f"{prefix} "                                    # e.g. "train" or "  val" - left-padded by the caller
            f"K {metrics['mae_log_K']:.4f}  "                # MAE log10(K), 4 decimal places
            f"G {metrics['mae_log_G']:.4f}  "                # MAE log10(G)
            f"K/G {metrics['mae_log_ratio']:.4f}  "          # MAE log10(K/G)
            f"corr {metrics['residual_corr']:+.3f}  "        # correlation between the K and G residuals, sign always shown
            f"kappa {metrics['kappa_mae_total']:.4f}")       # MAE log10(kappa_L) implied by the moduli


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
def main():
    cfg = parse_overrides()                        # CONFIG merged with any CLI overrides

    # Seed everything reachable so a rerun reproduces these numbers exactly.
    torch.manual_seed(cfg["init_seed"])             # seeds torch's global RNG (weight init, dropout masks, etc.)
    np.random.seed(cfg["init_seed"])                # seeds numpy's global RNG (used indirectly by some libraries)

    device = pick_device(cfg["device"])                          # resolves "auto" to a concrete torch.device
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])       # absolute path to the data directory
    out_dir = os.path.join(PROJECT_ROOT, cfg["out_dir"])         # absolute path to the results output directory
    os.makedirs(out_dir, exist_ok=True)                          # create it if missing; no error if it already exists

    print("=" * 78)
    print(f"  JOINT two-head CGCNN   target_mode = {cfg['target_mode']}")
    print("=" * 78)
    print(f"  device : {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"  tag    : {cfg['tag']}")
    print()

    # ---- data ---------------------------------------------------------------
    print("Loading dataset...")
    dataset, ids, log_K_all, log_G_all, sources = load_joint_dataset(     # reads graphs.pt + labels.csv and builds the torch Dataset
        data_dir, cfg["target_mode"], cfg["kappa_weight_alpha"],
        labels_file=cfg["labels_file"], cache_file=cfg["cache_file"])
    print(f"  {len(dataset)} crystals with BOTH moduli positive")

    # Split the MATBENCH block only, exactly as every previous round did, then
    # append every augmented crystal to train. Because 35_merge_aflow.py keeps
    # matbench first and in its original order, mb_positions is [0..N-1] and the
    # resulting val/test sets are bit-identical to rounds 1-6.
    mb_positions = np.where(sources == "matbench")[0]         # array indices whose source column reads "matbench"
    aug_positions = np.where(sources != "matbench")[0]        # every other index (AFLOW-augmented crystals, if any)
    tr, va, te = split_indices(len(mb_positions), cfg["train_ratio"],      # split only within the matbench block
                               cfg["val_ratio"], cfg["split_seed"])
    train_idx = mb_positions[tr].tolist() + aug_positions.tolist()   # matbench-train indices plus every augmented index
    val_idx = mb_positions[va].tolist()                               # matbench-val indices only
    test_idx = mb_positions[te].tolist()                              # matbench-test indices only
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
          f"  (split_seed={cfg['split_seed']})")
    if len(aug_positions):                # only prints this extra line when augmented crystals are actually present
        print(f"         {len(mb_positions)} matbench + {len(aug_positions)} augmented; "
              f"ALL augmented crystals are in train, none in val/test")

    loader_kwargs = dict(batch_size=cfg["batch_size"], collate_fn=collate_pool,   # shared DataLoader settings for all three splits
                         num_workers=cfg["num_workers"])
    # SubsetRandomSampler shuffles within the split each epoch; the split
    # itself is fixed by split_indices above.
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **loader_kwargs)  # yields shuffled train batches
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **loader_kwargs)       # yields shuffled val batches
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **loader_kwargs)     # yields shuffled test batches

    # ---- normalizer, fitted on TRAINING targets only ------------------------
    # Fitting on all 10,987 would leak test-set statistics into training.
    # [:, :2] drops the weight column - normalising it would be meaningless
    # and would corrupt the target statistics.
    train_targets = torch.stack([dataset[i][1] for i in train_idx])[:, :2]   # (n_train, 2) tensor of raw targets, weight column dropped
    normalizer = VectorNormalizer(train_targets).to(device)                  # computes per-column mean/std from train_targets, moves buffers to `device`
    head_names = (("log10(G)", "log10(K/G)") if cfg["target_mode"] == "G_and_ratio"
                  else ("log10(K)", "log10(G)"))                             # display-only labels matching whichever parameterisation is active
    print(f"  normalizer  {head_names[0]:>11}: mean={normalizer.mean[0]:.3f} "
          f"std={normalizer.std[0]:.3f}")
    print(f"              {head_names[1]:>11}: mean={normalizer.mean[1]:.3f} "
          f"std={normalizer.std[1]:.3f}")

    # ---- model --------------------------------------------------------------
    # Read the feature widths off a real sample rather than hard-coding them,
    # so the model always matches whatever atom_init.json and Gaussian settings
    # the cache was built with.
    sample_graph, _, _ = dataset[0]                       # peek at one dataset item; target/id are discarded here
    orig_atom_fea_len = sample_graph[0].shape[-1]     # 92   # width of the raw per-atom feature vector
    nbr_fea_len = sample_graph[1].shape[-1]           # 41   # width of the raw per-bond Gaussian-expansion vector

    model = JointCrystalGraphConvNet(                     # builds the shared-trunk, two-head network
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=cfg["atom_fea_len"], n_conv=cfg["n_conv"],
        h_fea_len=cfg["h_fea_len"], n_shared_fc=cfg["n_shared_fc"],
        n_head_fc=cfg["n_head_fc"], head_fea_len=cfg["head_fea_len"],
        dropout=cfg["dropout"],
        edge_dropout=cfg["edge_dropout"]).to(device)      # move every parameter/buffer onto `device`

    n_params = sum(p.numel() for p in model.parameters())   # total trainable scalar count across every parameter tensor
    print(f"  model: {n_params:,} parameters "
          f"({n_params / len(train_idx):.1f} per training crystal)")
    print()

    # ---- loss, optimiser, scheduler ----------------------------------------
    criterion = build_loss(cfg)                    # the per-element loss module chosen by CONFIG["loss_fn"]
    # Adam family over the paper's SGD: adaptive per-parameter steps converge
    # far more reliably on a dataset this size. See CONFIG["optimizer"] for why
    # the adam/adamw choice matters more than it looks - under plain Adam the
    # weight_decay above is L2-in-the-gradient and gets partly scaled away.
    if cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"],   # decoupled weight decay
                                weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],    # weight decay implemented as L2-in-gradient
                               weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"unknown optimizer {cfg['optimizer']!r}; "
                         f"expected adam or adamw")
    print(f"  optimizer: {cfg['optimizer']}  lr={cfg['learning_rate']} "
          f"weight_decay={cfg['weight_decay']}  loss={cfg['loss_fn']}")

    if cfg["scheduler"] == "cosine":
        # Glide the LR smoothly to ~0 over the run, so late epochs are
        # guaranteed to be fine-tuning steps rather than large jumps.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"],                                          # one full cosine cycle spans all epochs
            eta_min=cfg["learning_rate"] * cfg["cosine_eta_min_frac"])               # LR floor at the end of the cycle
    elif cfg["scheduler"] == "onecycle":
        # Warm up then anneal. Needs to know the total number of steps, and is
        # stepped per BATCH rather than per epoch.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["learning_rate"],
            epochs=cfg["epochs"], steps_per_epoch=len(train_loader))     # total steps = epochs * batches-per-epoch
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(                # "plateau": reactive, cuts LR on stagnation
            optimizer, mode="min", factor=cfg["plateau_factor"],
            patience=cfg["plateau_patience"])

    # ---- training loop ------------------------------------------------------
    history = []                  # one dict per epoch, appended below, written to history_*.csv at the end
    best_score = float("inf")     # lowest (best) validation selection-score seen so far
    best_state = None             # CPU copy of the state_dict at the best epoch
    best_epoch = -1               # which epoch produced best_state
    epochs_since_best = 0         # early-stopping counter
    start = time.time()           # wall-clock start, for the elapsed-time report

    # Stochastic weight averaging. AveragedModel keeps a running mean of the
    # weights; we only start feeding it once training has settled, because
    # averaging in the early high-LR epochs just drags the mean toward noise.
    swa_model, swa_start_epoch, swa_n = None, None, 0    # swa_n counts how many epochs actually got folded into the average
    if cfg["swa_start_frac"] > 0:
        swa_model = optim.swa_utils.AveragedModel(model)                          # wraps `model`, maintains a running weight average
        swa_start_epoch = int(cfg["epochs"] * (1.0 - cfg["swa_start_frac"]))      # e.g. swa_start_frac=0.25 over 200 epochs -> starts at epoch 150
        print(f"  SWA: averaging weights from epoch {swa_start_epoch} onwards")

    print(f"Training (selecting on validation '{cfg['select_metric']}')...")
    print("  columns: MAE log10(K) | log10(G) | log10(K/G) | residual corr | kappa MAE")
    print()

    for epoch in range(cfg["epochs"]):                        # 0-indexed epoch counter, up to CONFIG["epochs"] - 1
        train_loss, train_m, *_ = run_epoch(                  # one training pass; trailing outputs (pred/true arrays, ids) discarded here
            train_loader, model, criterion, normalizer, cfg, optimizer, device)
        val_loss, val_m, *_ = run_epoch(                      # one no-grad validation pass (no optimizer passed)
            val_loader, model, criterion, normalizer, cfg, device=device)

        # ReduceLROnPlateau needs the metric it watches; the others are on a
        # fixed schedule. OneCycle was already stepped per batch, so skip it.
        if cfg["scheduler"] == "plateau":
            scheduler.step(selection_score(val_m, cfg["select_metric"]))    # feed it this epoch's validation score
        elif cfg["scheduler"] == "cosine":
            scheduler.step()                                                # advance one point along the cosine curve

        # Fold this epoch's weights into the running average. Done before the
        # early-stop check so a run that stops early still contributes.
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)     # blends the current weights into the running average
            swa_n += 1                              # count this epoch as having contributed

        score = selection_score(val_m, cfg["select_metric"])   # single number this run selects the "best" epoch on
        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],       # current learning rate, read off the optimizer
            "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_m.items()},              # every train metric, prefixed "train_"
            **{f"val_{k}": v for k, v in val_m.items()},                  # every val metric, prefixed "val_"
            "val_select_score": score,
        })

        # Keep the best weights seen so far. Small datasets give noisy
        # validation curves, so the final epoch is very often not the best one.
        improved = score < best_score - cfg["early_stop_min_delta"]    # strictly better by more than the noise floor
        if improved:
            best_score, best_epoch, epochs_since_best = score, epoch, 0    # record the new best and reset the stall counter
            # .cpu() so a checkpoint trained on a GPU box loads on a laptop
            # with no CUDA at all.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # deep copy of every weight tensor
        else:
            epochs_since_best += 1     # one more epoch without an improvement

        if epoch % cfg["print_every"] == 0 or improved or epoch == cfg["epochs"] - 1:   # print on a fixed cadence, on any improvement, and on the final epoch
            marker = " *" if improved else "  "
            print(f"  e{epoch:3d}{marker} "
                  + format_metrics("train", train_m) + "   |   "
                  + format_metrics("val", val_m))

        # Stop once validation has not improved for a while - the old runs
        # spent 100+ epochs past this point fitting nothing but training noise.
        if cfg["early_stop_patience"] and epochs_since_best >= cfg["early_stop_patience"]:   # patience=0 disables early stopping entirely
            print(f"\n  early stop: no improvement for {cfg['early_stop_patience']} "
                  f"epochs (best was epoch {best_epoch})")
            break                                # exits the epoch loop early

    elapsed = time.time() - start    # total training wall-clock time in seconds
    print(f"\nTrained {len(history)} epochs in {elapsed:.0f}s. "
          f"Best epoch {best_epoch} ({cfg['select_metric']} = {best_score:.4f})")

    # ---- final evaluation on the held-out test set --------------------------
    # Restore the best checkpoint first: evaluating the LAST epoch would report
    # a model we already decided was worse.
    model.load_state_dict(best_state)    # restore the weights from the best epoch before scoring anything further
    pred_rows = []                        # per-crystal prediction DataFrames, one appended per split below

    # If SWA collected anything, score it against the best single checkpoint on
    # VALIDATION and keep whichever wins. Deciding on validation (never test)
    # keeps the test set honest, and means enabling SWA can only help.
    if swa_model is not None and swa_n > 0:
        print(f"\n  SWA: averaged {swa_n} epochs; recomputing BatchNorm stats...")
        update_bn_stats(train_loader, swa_model.module, device)    # swa_model.module is the underlying averaged network
        _, swa_val_m, *_rest = run_epoch(val_loader, swa_model.module, criterion,   # score the SWA-averaged weights on validation
                                     normalizer, cfg, device=device)
        swa_score = selection_score(swa_val_m, cfg["select_metric"])
        print(f"  SWA  val {cfg['select_metric']} = {swa_score:.4f}"
              f"   vs best checkpoint {best_score:.4f}")
        if swa_score < best_score:
            print("  -> SWA wins, keeping the averaged weights")
            best_state = {k: v.detach().cpu().clone()
                          for k, v in swa_model.module.state_dict().items()}    # overwrite best_state with the SWA weights
            best_score = swa_score
            model.load_state_dict(best_state)     # load the SWA weights into `model` for everything that follows
        else:
            print("  -> best single checkpoint wins, discarding the average")

    results = {}                                       # split name -> metrics dict, e.g. results["test"]["kappa_mae_total"]
    for split_name, loader in (("train", train_loader),
                               ("val", val_loader),
                               ("test", test_loader)):
        _, m, pK, pG, tK, tG, ids = run_epoch(          # re-run each split once more with the FINAL chosen weights
            loader, model, criterion, normalizer, cfg, device=device)
        results[split_name] = m
        # Keep the per-crystal numbers, not just the aggregate. They are what
        # an ensemble is built from (32_ensemble_joint.py averages them in log
        # space and uses the member spread as a per-crystal uncertainty), and
        # a summary JSON cannot be un-aggregated after the fact.
        if split_name == "test":
            # Stratified by TRUE kappa. Average MAE cannot tell you whether a
            # change helped the low-kappa tail, which is the only part a
            # screen uses - see kappa_quintile_breakdown for the evidence.
            strat = kappa_quintile_breakdown(tK, tG, pK, pG)   # dict of per-quintile stats, keyed "Q1".."Q5" plus recall_at_10pct
        pred_rows.append(pd.DataFrame({                        # one row per crystal in this split
            "material_id": ids, "split": split_name,
            "true_log10_K": tK, "pred_log10_K": pK,
            "true_log10_G": tG, "pred_log10_G": pG}))

    print()
    print("=" * 78)
    print("  FINAL - best checkpoint")
    print("=" * 78)
    for split_name in ("train", "val", "test"):
        print("  " + format_metrics(f"{split_name:>5}", results[split_name]))

    print()
    print("  BY TRUE KAPPA QUINTILE  (Q1 = lowest kappa = what a screen selects on)")
    print(f"    {'bin':<5}{'n':>6}{'kappa MAE':>11}{'MAE K':>9}{'MAE G':>9}{'median G':>11}")
    for qb in [f"Q{i}" for i in range(1, 6)]:    # "Q1".."Q5", Q1 = lowest true kappa
        if qb not in strat:
            continue                              # a quintile can be empty on a very small test set
        d = strat[qb]                             # this quintile's stats dict
        print(f"    {qb:<5}{d['n']:6d}{d['kappa_mae']:11.4f}{d['mae_log_K']:9.4f}"
              f"{d['mae_log_G']:9.4f}{d['median_true_G_GPa']:10.1f} GPa")
    print(f"    recall@10% (true bottom decile picked) = "
          f"{100 * strat['recall_at_10pct']:.1f}%   "
          f"[separate-model baseline: 70.1%]")

    t = results["test"]
    print()
    print("  kappa_L error budget on the test set (rho and V cancel exactly):")
    print(f"    total  MAE log10(kappa)     = {t['kappa_mae_total']:.4f}"
          f"   ({100 * (10 ** t['kappa_mae_total'] - 1):.0f}% typical error)")
    print(f"      from prefactor G * v_s    = {t['kappa_mae_prefactor']:.4f}")
    print(f"      from exp(-gamma)          = {t['kappa_mae_anharmonic']:.4f}")
    print(f"    gamma MAE                   = {t['gamma_mae']:.4f}")
    print()
    # Baselines measured on this exact test set (split_seed 42) from the
    # existing separate-model predictions_*.csv. ONE joint model should be
    # compared against ONE separate model per target - the ensemble column is
    # the bar for a 3-seed joint ensemble, not for this single run.
    print("  vs SEPARATE models on the same 1,648 crystals:")
    print(f"                    this run    1 model each    3-model ens.")
    print(f"    MAE log10(K)     {t['mae_log_K']:.4f}        0.0696          0.0630")
    print(f"    MAE log10(G)     {t['mae_log_G']:.4f}        0.0836          0.0781")
    print(f"    MAE log10(K/G)   {t['mae_log_ratio']:.4f}        0.0993          0.0887"
          f"   <- the target")
    print(f"    residual corr   {t['residual_corr']:+.3f}       +0.263         +0.297"
          f"   <- the mechanism")
    print(f"    kappa MAE        {t['kappa_mae_total']:.4f}        0.2065          0.1897")

    # ---- save ---------------------------------------------------------------
    # Every output filename below is stamped "_31_" (this script's number) so
    # that when this project's other numbered scripts write into the same
    # results/ directory, the filename alone says which script produced which
    # file - see scripts/cgcnn/comment_and_rename_spec (script-number-in-
    # result-names rule). 32_ensemble_joint.py reads the checkpoint and
    # predictions files this block writes, so it must use this exact same
    # "model_31_<tag>.pth" / "predictions_31_<tag>.csv" naming.
    tag = cfg["tag"]                                            # this run's identifying suffix, from --tag
    ckpt_path = os.path.join(out_dir, f"model_31_{tag}.pth")    # was model_joint_{tag}.pth
    torch.save({"state_dict": best_state,                        # the model weights to reload later
                "normalizer": normalizer.state_dict(),            # target mean/std, needed to denormalise future predictions
                "config": cfg,                                    # the full CONFIG+overrides used for this run
                "best_epoch": best_epoch,
                "best_score": best_score,
                "results": results,                               # train/val/test metrics dicts
                "split": {"train": train_idx, "val": val_idx, "test": test_idx},   # exact crystal indices in each split
                "ids": ids},                                       # material ids in the same order run_epoch last returned them
               ckpt_path)

    pred_path = os.path.join(out_dir, f"predictions_31_{tag}.csv")   # was predictions_joint_{tag}.csv
    pd.concat(pred_rows, ignore_index=True).to_csv(pred_path, index=False)   # stack train+val+test rows into one file

    hist_path = os.path.join(out_dir, f"history_31_{tag}.csv")       # was history_joint_{tag}.csv
    pd.DataFrame(history).to_csv(hist_path, index=False)              # one row per epoch, for plotting learning curves

    summary_path = os.path.join(out_dir, f"summary_31_{tag}.json")   # was summary_joint_{tag}.json
    with open(summary_path, "w") as fh:
        json.dump({"tag": tag, "config": cfg, "n_params": n_params,
                   "n_total": len(dataset), "n_train": len(train_idx),
                   "n_val": len(val_idx), "n_test": len(test_idx),
                   "best_epoch": best_epoch, "epochs_run": len(history),
                   "seconds": elapsed, "results": results,
                   "test_by_kappa_quintile": strat},
                  fh, indent=2)

    print()
    print(f"  saved  {os.path.relpath(ckpt_path, PROJECT_ROOT)}")
    print(f"         {os.path.relpath(pred_path, PROJECT_ROOT)}")
    print(f"         {os.path.relpath(hist_path, PROJECT_ROOT)}")
    print(f"         {os.path.relpath(summary_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
