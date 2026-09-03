"""
Joint two-head CGCNN: predict BOTH elastic moduli from ONE shared trunk.
=========================================================================

WHY THIS FILE EXISTS
--------------------
The original pipeline trains two completely separate networks - one for the
bulk modulus K, one for the shear modulus G - and then feeds both into the
Slack physics to get kappa_L. That looks harmless. It is not.

Look at what the physics actually does with K and G:

    x^2   = (K + 4/3 G) / G  =  K/G + 4/3      <- ONLY depends on the RATIO K/G
    nu    = (x^2 - 2) / (2 x^2 - 2)
    gamma = 3(1 + nu) / (2(2 - 3 nu))
    kappa = G * v_s * V^(1/3) / (N * 300) * exp(-gamma)

The Grueneisen parameter gamma - and therefore the whole exp(-gamma)
anharmonic factor - is a function of K/G and nothing else. In log space,

    log10(K/G) = log10(K) - log10(G)

which is a DIFFERENCE of two numbers. If two independent models each make an
error, that difference inherits BOTH errors. If instead the two errors move
together (both predictions too high by a similar amount), the errors cancel in
the difference and gamma comes out right anyway.

MEASURED ON THE 1,648-CRYSTAL TEST SET, WITH THE OLD SEPARATE MODELS:

    true log10(K) vs log10(G) correlation : 0.884   <- almost the same signal
    RESIDUAL correlation (err_K vs err_G) : 0.297   <- but errors are ~independent

    MAE log10(K)   = 0.0630
    MAE log10(G)   = 0.0781
    MAE log10(K/G) = 0.0887   <- BIGGER than either individual error

    kappa_L error budget (rho and V cancel exactly, so this is exact):
        TOTAL MAE log10(kappa)    = 0.1897
          from prefactor G * v_s  = 0.1160
          from exp(-gamma)        = 0.0911
    Counterfactual, same modulus error but perfectly correlated:
        TOTAL MAE log10(kappa)    = 0.0882   <- 54% of the error disappears

So roughly half of the kappa_L error is not accuracy at all. It is the fact
that two separate networks make uncorrelated mistakes.

WHAT THIS FILE DOES ABOUT IT
----------------------------
Two changes, both aimed at forcing the errors to correlate:

  1. ONE SHARED TRUNK. Both outputs are read off the same crystal embedding,
     so whatever the trunk gets wrong about a crystal, it gets wrong for both
     outputs at once. That is exactly the correlated error we want.

  2. PREDICT THE RATIO DIRECTLY. Instead of two heads emitting log10(K) and
     log10(G), the heads emit log10(G) and log10(K/G). The ratio - the thing
     gamma actually depends on - becomes a quantity the network is explicitly
     trained and scored on, rather than an accident of subtracting two
     independent guesses. K is then recovered exactly as:

         log10(K) = log10(G) + log10(K/G)

     This is a lossless reparameterisation: no information is thrown away,
     the same two numbers come out, but the error budget is reshaped.

Both choices are switchable (see TARGET_MODE in the training script) so the
whole hypothesis can be A/B tested rather than assumed.

RELATION TO THE LITERATURE
--------------------------
Wu et al. 2026 (KappaFormer) also emits B and G from one shared "harmonic
branch" before feeding a Slack-style formula - but the paper never states why
a shared trunk matters, and never measures the error correlation. The
numbers above are the missing measurement.
"""

from __future__ import print_function, division

import os

import numpy as np
import torch
import torch.nn as nn

# Reuse the gated message-passing layer from the original implementation
# verbatim. The convolution is not what we are changing here - only what sits
# on top of it - so importing keeps the two models genuinely comparable.
from cgcnn_scratch.model import ConvLayer


# =============================================================================
#  EDGE DROPOUT
# =============================================================================
class EdgeDropoutConvLayer(ConvLayer):
    """ConvLayer that randomly drops a fraction of each atom's neighbours.

    WHY
    ---
    The graphs are pre-built once into graphs.pt and never change, so across
    150 epochs the network sees the byte-identical input for a given crystal
    every single time. There is nothing to stop it memorising, and the measured
    train/test MAE gap of 2.0-2.4x says it does exactly that.

    Standard image augmentation (crops, flips) has no crystal analogue, but
    graphs have their own: hide some of the edges. Each forward pass the atom
    then has to predict from a random subset of its coordination shell, so it
    cannot lean on any one neighbour being present. This is DropEdge, well
    established for molecular GNNs.

    IMPORTANT - this is the CHEAP augmentation, not the best one. Perturbing
    the atomic POSITIONS (thermal jitter of ~0.01-0.05 A) is better motivated
    for this particular problem: Srivastava et al. 2024 found that ML models
    trained on thermally-populated configurations recovered cubic anharmonic
    force constants an order of magnitude more accurately than models trained
    on finite-difference displacements, and Ojih et al. 2023 use thermal mean
    squared displacement as a kappa descriptor outright. Jitter needs the raw
    bond distances, which the cache does not keep - it stores the already
    Gaussian-expanded features - so it needs a cache format change. Edge
    dropout needs nothing.

    WHAT IS AND IS NOT DROPPED
    --------------------------
    The mask is per (atom, neighbour) pair and resampled every forward pass.
    Surviving messages are rescaled by 1/(1-p) so the expected value of the
    neighbour sum is unchanged - without that, activations shrink at training
    time and the network silently relearns the scale at eval time.

    Note the graphs already contain padding edges: data.py pads atoms with
    fewer than max_num_nbr neighbours using index 0 and distance radius+1.
    Dropping some of those too is harmless - they carry no real information.

    Inference is untouched: nn.Module.eval() sets self.training False and the
    mask is skipped entirely, so predictions stay deterministic.
    """

    def __init__(self, atom_fea_len, nbr_fea_len, edge_dropout=0.0):
        super(EdgeDropoutConvLayer, self).__init__(atom_fea_len, nbr_fea_len)
        self.edge_dropout = edge_dropout

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """Mirrors ConvLayer.forward exactly, with one masking step inserted.

        The parent's body is repeated rather than called because the mask has
        to go between the gating and the sum over neighbours, which is halfway
        through it. Keeping the two side by side makes the single difference
        obvious; see cgcnn_scratch/model.py for the annotated original.
        """
        N, M = nbr_fea_idx.shape

        # Gather each atom's M neighbours -> (N, M, atom_fea_len).
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # Build the (self, neighbour, bond) triple for every edge.
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea,
             nbr_fea], dim=2)

        # One shared linear layer over every edge - this is the convolution.
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        # Split into the sigmoid gate and the softplus message.
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        messages = nbr_filter * nbr_core

        # ---- THE ONLY DIFFERENCE FROM ConvLayer ---------------------------
        if self.training and self.edge_dropout > 0:
            keep = 1.0 - self.edge_dropout
            # (N, M, 1) so the same decision applies to a neighbour's whole
            # feature vector - dropping individual channels would be ordinary
            # dropout, not edge dropout.
            mask = torch.bernoulli(
                torch.full((N, M, 1), keep, device=messages.device,
                           dtype=messages.dtype))
            # Divide by `keep` so E[sum] is unchanged (inverted dropout).
            messages = messages * mask / keep
        # -------------------------------------------------------------------

        nbr_sumed = torch.sum(messages, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        # Residual update, as in the original.
        return self.softplus2(atom_in_fea + nbr_sumed)


# =============================================================================
#  TARGET PARAMETERISATIONS
# =============================================================================
# Both modes predict two numbers per crystal. They differ only in WHICH two.
#
#   "G_and_ratio"  head 0 -> log10(G),  head 1 -> log10(K/G)      [the new idea]
#   "K_and_G"      head 0 -> log10(K),  head 1 -> log10(G)        [the baseline]
#
# Keeping the baseline available in the same code path matters: it isolates
# the reparameterisation from every other difference (shared trunk, loss,
# regularisation), so an A/B run measures the one thing we care about.
TARGET_MODES = ("G_and_ratio", "K_and_G")


def targets_from_KG(log_K, log_G, mode):
    """Convert (log10 K, log10 G) into whatever pair `mode` asks the net to learn.

    Parameters
    ----------
    log_K, log_G : array-like or torch.Tensor
        Base-10 logs of the two moduli, in GPa.
    mode : str
        One of TARGET_MODES.

    Returns
    -------
    (t0, t1) - the two training targets, same type as the inputs.
    """
    if mode == "G_and_ratio":
        # head 0 = log10(G); head 1 = log10(K) - log10(G) = log10(K/G)
        return log_G, log_K - log_G
    if mode == "K_and_G":
        return log_K, log_G
    raise ValueError(f"unknown target mode {mode!r}, expected one of {TARGET_MODES}")


def KG_from_targets(t0, t1, mode):
    """Inverse of targets_from_KG: turn the network's two outputs back into (K, G).

    Every evaluation goes through here, so the two modes are always scored on
    the SAME physical quantities (log10 K and log10 G) no matter which
    parameterisation was trained. Without this the two runs would not be
    comparable.
    """
    if mode == "G_and_ratio":
        log_G = t0
        log_K = t0 + t1          # log10(G) + log10(K/G) = log10(K)
        return log_K, log_G
    if mode == "K_and_G":
        return t0, t1
    raise ValueError(f"unknown target mode {mode!r}, expected one of {TARGET_MODES}")


# =============================================================================
#  NORMALIZER (per-column)
# =============================================================================
class VectorNormalizer(object):
    """Standardise EACH target column to zero mean and unit variance separately.

    The original Normalizer in data.py takes a scalar mean/std because it only
    ever saw one target. Here the two targets live on wildly different scales:

        log10(G)    spans roughly 0.5 .. 2.6   (3 to 400 GPa)
        log10(K/G)  spans roughly 0.0 .. 0.8   (K/G of 1 to ~6)

    Normalising them jointly with one scalar std would make the ratio head's
    gradients tiny compared to the G head's, and the network would effectively
    ignore the ratio - which is the exact quantity we built this model to get
    right. Per-column statistics put both heads on an equal footing.

    As in the original: statistics come from the TRAINING SPLIT ONLY. Computing
    them over the whole dataset leaks test information into training.
    """

    def __init__(self, tensor):
        # tensor is (N, 2). dim=0 -> one mean and one std per column.
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)

    def norm(self, tensor):
        """Physical units -> normalised units (what the network sees)."""
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """Normalised units -> physical log10 units (what we report)."""
        return normed_tensor * self.std + self.mean

    def to(self, device):
        """Move the statistics to wherever the model lives, or torch complains."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self):
        # Always serialise on CPU so a GPU-trained checkpoint loads on a laptop.
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


# =============================================================================
#  DATASET
# =============================================================================
class JointGraphCacheData(torch.utils.data.Dataset):
    """Serve pre-built crystal graphs with TWO targets each.

    Identical in spirit to GraphCacheData in data.py, except __getitem__
    returns a length-2 target tensor instead of length-1. collate_pool in
    data.py already uses torch.stack on the targets, so it handles the extra
    column with no changes - the batch target simply comes out (B, 2)
    instead of (B, 1).
    """

    def __init__(self, cache_path, targets, ids=None, weights=None):
        """
        Parameters
        ----------
        cache_path : str
            Path to graphs.pt written by 01b_prepare_full_dataset.py.
        targets : dict
            Maps crystal id -> (t0, t1), already in the chosen parameterisation.
        ids : list of str, optional
            Restrict to (and order by) these ids. Defaults to every id in the
            cache that we also have a label for, in the cache's own order.
        """
        # weights_only=False explicitly: torch 2.6 flipped this default, and a
        # cache written under an older torch fails to unpickle otherwise. This
        # is a file we wrote ourselves, so there is no untrusted-pickle risk.
        blob = torch.load(cache_path, weights_only=False)
        self.graphs = dict(zip(blob["ids"], blob["graphs"]))

        if ids is None:
            ids = [i for i in blob["ids"] if i in targets]
        missing = [i for i in ids if i not in self.graphs]
        assert not missing, f"{len(missing)} ids missing from cache, e.g. {missing[:3]}"

        self.ids = list(ids)
        self.targets = targets
        self.weights = weights or {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        # Any number of targets: 2 for the (G, K/G) model, 3 when a gamma
        # head is present. The weight is always the LAST column, so the
        # trainer slices [:, :n_heads] for targets and [:, n_heads] for it.
        values = tuple(self.targets[item_id])
        # Shape (n_targets + 1,): the targets plus the per-crystal loss WEIGHT.
        # The weight rides along inside the target tensor rather than as a
        # fourth return value so that collate_pool in data.py works unchanged -
        # it does torch.stack on whatever width it is handed. run_epoch splits
        # columns 0:2 (targets) from column 2 (weight).
        w = self.weights.get(item_id, 1.0) if self.weights else 1.0
        return (self.graphs[item_id],
                torch.Tensor([float(v) for v in values] + [float(w)]),
                item_id)


def kappa_sample_weights(log_K, log_G, alpha):
    """Per-crystal loss weights that upweight the LOW-kappa tail.

    WHY
    ---
    Stratifying the test set by true kappa shows the model is far worse exactly
    where the screen operates:

        quintile    MAE K   MAE G   median true G
        Q1 (low)   0.1017  0.1340      10.5 GPa
        Q4         0.0404  0.0565      58.5 GPa

    That is a 2.5x gap, and it is NOT an artefact of the Slack formula
    amplifying errors at low kappa - the moduli themselves are predicted worse.
    Low-kappa crystals are SOFT (median G of 10.5 GPa against 102 GPa at the
    stiff end), soft crystals are rare in matbench, and an unweighted MSE/Huber
    loss is therefore dominated by the stiff majority. Classic imbalanced
    regression.

    The fix is the one Guo et al. 2023 used for phonon scattering rates, where
    the rare high-rate processes dominate the physics but not the loss: weight
    each sample by a power of its own target value. There, w = Gamma^0.4. Here,
    w scales as kappa^(-alpha), so soft low-kappa crystals pull harder on the
    gradient.

    Parameters
    ----------
    log_K, log_G : arrays of base-10 log moduli (GPa).
    alpha : float
        0.0 disables weighting entirely and every weight is exactly 1.0 -
        the behaviour of every run before this was added. 0.3-0.5 is a
        reasonable first sweep. Larger values chase the tail harder and start
        to cost accuracy on the bulk of the data.

    Returns
    -------
    Array of weights, mean 1.0, clipped to [0.25, 4.0].
    """
    if alpha <= 0:
        return np.ones(len(log_K))

    # kappa here is the rho- and V-independent part, computed from the TRUE
    # moduli. It is a deterministic function of the labels we already have, so
    # this leaks nothing: a crystal's weight depends only on its own label, and
    # test-set weights are never used in training.
    prefactor, anharmonic, gamma = slack_factors(np.asarray(log_K, dtype=float),
                                                 np.asarray(log_G, dtype=float))
    kappa = prefactor * anharmonic

    # Fall back to the median for any crystal where the physics is undefined,
    # so a handful of pathological K/G values cannot produce inf weights.
    bad = ~np.isfinite(kappa) | (kappa <= 0)
    if bad.any():
        kappa = kappa.copy()
        kappa[bad] = np.median(kappa[~bad])

    # Scale by the median first so alpha means the same thing regardless of the
    # arbitrary constant absorbed into kappa by dropping rho and V.
    w = (kappa / np.median(kappa)) ** (-alpha)
    # Clip before normalising: the kappa distribution has a long tail in both
    # directions and a single extreme crystal should not dominate an epoch.
    w = np.clip(w, 0.25, 4.0)
    return w / w.mean()


def load_joint_dataset(data_dir, mode, kappa_weight_alpha=0.0,
                       labels_file="labels.csv", cache_file="graphs.pt"):
    """Build the two-target Dataset from labels.csv + the graph cache.

    Both training and evaluation come through this one function so they can
    never disagree about dataset content or ordering - the split indices saved
    in a checkpoint are only meaningful if the dataset rebuilds identically.

    Returns
    -------
    (dataset, ids, log_K, log_G, sources) - the last four aligned to the
    dataset's own ordering. `sources` is what lets the trainer force augmented
    crystals into TRAIN so the held-out test set never changes.
    """
    import pandas as pd

    labels = pd.read_csv(os.path.join(data_dir, labels_file))

    # Drop rows where EITHER modulus is non-positive. A modulus <= 0 is
    # unphysical (a bad DFT entry) and log10 of it would hand the loss a -inf
    # and destroy the run. Note this is stricter than the single-target
    # clean_labels(), which only ever checked one column - so the joint model
    # trains on the intersection of the two single-target datasets.
    n_before = len(labels)
    labels = labels[(labels["K_VRH"] > 0) & (labels["G_VRH"] > 0)]
    n_dropped = n_before - len(labels)
    if n_dropped:
        print(f"  dropping {n_dropped} rows with non-positive K_VRH or G_VRH")

    cache_path = os.path.join(data_dir, cache_file)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No graph cache at {cache_path}. Build it first with:\n"
            f"    python scripts/cgcnn/01b_prepare_full_dataset.py")
    print(f"  using pre-built graph cache: {cache_path}")

    log_K = np.log10(labels["K_VRH"].to_numpy(dtype=float))
    log_G = np.log10(labels["G_VRH"].to_numpy(dtype=float))
    ids = labels["mb_id"].tolist()
    # A merged labels file marks where each crystal came from. Absent the
    # column (the plain matbench file), everything is matbench.
    sources = (labels["source"].to_numpy() if "source" in labels
               else np.array(["matbench"] * len(labels)))

    # Convert to whichever pair of targets this run is training on.
    t0, t1 = targets_from_KG(log_K, log_G, mode)
    targets = {i: (a, b) for i, a, b in zip(ids, t0, t1)}

    # Per-crystal loss weights. alpha=0 gives all-ones and changes nothing.
    w = kappa_sample_weights(log_K, log_G, kappa_weight_alpha)
    weights = dict(zip(ids, w))
    if kappa_weight_alpha > 0:
        print(f"  kappa weighting alpha={kappa_weight_alpha}: "
              f"weights {w.min():.2f} to {w.max():.2f}, mean {w.mean():.2f}")

    dataset = JointGraphCacheData(cache_path, targets, weights=weights)
    # Re-align the raw logs to the dataset's own id order. The cache order and
    # the CSV order need not match, and every downstream analysis indexes by
    # dataset position, so this alignment is not optional.
    pos = {i: k for k, i in enumerate(ids)}
    order = [pos[i] for i in dataset.ids]
    return dataset, dataset.ids, log_K[order], log_G[order], sources[order]


# =============================================================================
#  MODEL
# =============================================================================
class JointCrystalGraphConvNet(nn.Module):
    """CGCNN with one shared trunk and two independent output heads.

    Layout:

        atom features
              |  embedding (Linear)
              v
        [ ConvLayer ] x n_conv          <-- SHARED: identical to the original
              |
              |  mean-pool over atoms   <-- SHARED: keeps output intensive
              v
        crystal vector (atom_fea_len)
              |  conv_to_fc + softplus
              |  [ shared FC ] x n_shared_fc
              v
        shared crystal embedding (h_fea_len)
             /                        \\
        head 0                      head 1
        [FC] x n_head_fc            [FC] x n_head_fc
        -> 1 scalar                 -> 1 scalar

    Everything above the fork is shared. That is the whole point: a crystal
    the trunk misreads is misread for BOTH outputs simultaneously, which makes
    the two errors correlate and lets them cancel in log10(K/G).

    The heads are deliberately kept SMALL (one hidden layer by default). Large
    heads would let each branch specialise and re-decorrelate the errors,
    undoing the benefit of sharing. n_head_fc is exposed so this can be tested
    rather than trusted.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128,
                 n_shared_fc=1, n_head_fc=1, head_fea_len=64,
                 dropout=0.0, edge_dropout=0.0, n_heads=2):
        """
        Parameters
        ----------
        orig_atom_fea_len : int
            Width of the raw element vector from atom_init.json (92).
        nbr_fea_len : int
            Width of the Gaussian-expanded bond vector (41 at step 0.2, 0-8 A).
        atom_fea_len : int
            Hidden width inside the convolutions.
        n_conv : int
            Message-passing rounds. Each round widens the neighbourhood an atom
            can see by one bond hop; too many over-smooth the graph.
        h_fea_len : int
            Width of the shared fully-connected stack after pooling.
        n_shared_fc : int
            Extra shared FC layers after conv_to_fc. 0 means the heads fork
            straight off conv_to_fc's output.
        n_head_fc : int
            Hidden FC layers inside EACH head. 0 makes a head a bare linear
            readout - maximally shared, minimum capacity to decorrelate.
        head_fea_len : int
            Width of the hidden layers inside each head.
        dropout : float
            Dropout probability applied to the shared embedding before the
            heads. NOTE: the original CrystalGraphConvNet builds a Dropout
            layer but only ever applies it when classification=True, so the
            regression path has had NO dropout at all. That is one reason the
            old models show a 2.0-2.4x train/test MAE gap. Here it is always
            available; set 0.0 to disable.
        edge_dropout : float
            Fraction of each atom's neighbour messages randomly dropped every
            forward pass during training - graph augmentation against the
            memorisation the fixed graph cache invites. See
            EdgeDropoutConvLayer for the full rationale. 0.0 disables it and
            reproduces the original ConvLayer exactly.
        """
        super(JointCrystalGraphConvNet, self).__init__()

        # ---- shared trunk -------------------------------------------------
        # Project the fixed 92-dim element descriptors into the hidden space.
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # A stack of gated message-passing layers, each with its own weights.
        # EdgeDropoutConvLayer with edge_dropout=0.0 behaves EXACTLY like the
        # original ConvLayer (the mask branch is skipped), so using it
        # unconditionally costs nothing and keeps one code path.
        self.convs = nn.ModuleList([
            EdgeDropoutConvLayer(atom_fea_len=atom_fea_len,
                                 nbr_fea_len=nbr_fea_len,
                                 edge_dropout=edge_dropout)
            for _ in range(n_conv)
        ])

        # Widen from the per-atom width to the crystal-level width.
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        # Optional extra shared layers. ModuleList (not Sequential) so the
        # activation stays explicit and readable in forward().
        self.shared_fcs = nn.ModuleList(
            [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_shared_fc)])
        self.shared_acts = nn.ModuleList(
            [nn.Softplus() for _ in range(n_shared_fc)])

        # Dropout sits at the fork: it perturbs the SHARED embedding, so both
        # heads see the same perturbation. That is regularisation that does not
        # decorrelate the two outputs.
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ---- N heads ------------------------------------------------------
        # n_heads=2 is the (log10 G, log10 K/G) model of rounds 1-7. n_heads=3
        # adds a gamma head, which replaces the empirical Poisson relation
        # that derives gamma from K/G. That derivation is a poor stand-in:
        # against AFLOW's own tabulated gamma it scores MAE 0.4326 with a
        # correlation of only 0.4205, and substituting the real value cuts the
        # error against AFLOW's own kappa by 25% (0.2395 -> 0.1806).
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            self._build_head(h_fea_len, head_fea_len, n_head_fc)
            for _ in range(n_heads)])
        # Kept so checkpoints from the two-head rounds still load by name.
        self.head0 = self.heads[0]
        self.head1 = self.heads[1]

    @staticmethod
    def _build_head(in_len, hidden_len, n_hidden):
        """A small MLP: in_len -> (hidden_len, softplus) x n_hidden -> 1."""
        layers = []
        width = in_len
        for _ in range(n_hidden):
            layers.append(nn.Linear(width, hidden_len))
            layers.append(nn.Softplus())
            width = hidden_len
        layers.append(nn.Linear(width, 1))   # scalar output
        return nn.Sequential(*layers)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Shapes:
            N  = total atoms across the whole batch (all crystals concatenated)
            M  = neighbours kept per atom
            N0 = number of crystals in the batch

        Returns
        -------
        (N0, 2) tensor: column 0 is head 0's output, column 1 is head 1's.
        What those two columns MEAN depends on TARGET_MODE; use
        KG_from_targets() to turn them back into (log10 K, log10 G).
        """
        # 1. embed raw element descriptors into hidden space
        atom_fea = self.embedding(atom_fea)

        # 2. message passing - each pass lets an atom see one hop further
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # 3. pool atoms -> one vector per crystal (mean keeps it intensive:
        #    doubling the unit cell must not change a modulus)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # 4. shared MLP. The doubled softplus mirrors the original
        #    implementation exactly so the trunk stays comparable.
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        for fc, act in zip(self.shared_fcs, self.shared_acts):
            crys_fea = act(fc(crys_fea))

        # 5. regularise the shared embedding, then fork
        crys_fea = self.dropout(crys_fea)

        # 6. every head, concatenated into one (N0, n_heads) output
        return torch.cat([head(crys_fea) for head in self.heads], dim=1)

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        """Average atom vectors within each crystal.

        The batch stacks every crystal's atoms into one long (N, F) tensor, so
        crystal_atom_idx is what tells us which rows belong together. MEAN, not
        sum, so the prediction does not change if the cell is written with a
        different number of atoms.
        """
        # Every atom must be claimed by exactly one crystal.
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.shape[0]
        return torch.cat(
            [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
             for idx_map in crystal_atom_idx], dim=0)


# =============================================================================
#  PHYSICS + ERROR BUDGET
# =============================================================================
# These reproduce the Slack pipeline from 07_predict_kappa.py, but factored so
# that density (rho) and cell volume (V) cancel. That matters: rho and V come
# from the CIF, not from the model, so they are IDENTICAL for the true and
# predicted kappa. Their ratio is therefore exactly 1 and we can compute the
# model's share of the kappa error WITHOUT needing either quantity.

def slack_factors(log_K, log_G):
    """Return the rho/V-independent pieces of kappa_L from log10 moduli.

    kappa_L  =  [G * v_s]  *  [exp(-gamma)]  *  [V^(1/3) / (N * 300)]
                 ^prefactor    ^anharmonic       ^comes from the CIF, cancels

    Both v_long and v_trans carry the same 1/sqrt(rho), and v_s is a
    homogeneous function of them, so v_s also scales as 1/sqrt(rho). Dropping
    rho therefore rescales the prefactor by a constant that cancels in any
    predicted/true ratio - which is all we ever take.

    Returns
    -------
    prefactor : G * v_s      (up to the cancelling 1/sqrt(rho))
    anharmonic: exp(-gamma)
    gamma     : the Grueneisen parameter itself
    """
    K = 10.0 ** log_K
    G = 10.0 ** log_G

    v_long = np.sqrt(K + 4.0 / 3.0 * G)   # missing 1/sqrt(rho) - cancels
    v_trans = np.sqrt(G)                  # missing the same 1/sqrt(rho)
    v_sound = ((1.0 / v_long ** 3 + 2.0 / v_trans ** 3) / 3.0) ** (-1.0 / 3.0)

    x2 = (v_long / v_trans) ** 2          # = K/G + 4/3, the ratio-only quantity
    nu = (x2 - 2.0) / (2.0 * x2 - 2.0)    # Poisson ratio
    gamma = 3.0 * (1.0 + nu) / (2.0 * (2.0 - 3.0 * nu))

    return G * v_sound, np.exp(-gamma), gamma


def kappa_quintile_breakdown(true_log_K, true_log_G, pred_log_K, pred_log_G,
                             n_bins=5):
    """Error broken out by TRUE kappa quintile - the metric that actually matters.

    WHY THIS EXISTS
    ---------------
    Five rounds of architecture work were scored on average kappa MAE, and the
    winner turned out to be no better at the only job the pipeline has. Ranking
    the test set by true kappa and checking recall of the true bottom decile:

        separate baseline   recall@10% = 115/164 = 70.1%
        joint ensemble      recall@10% = 115/164 = 70.1%

    Identical. The joint model's gains sat in Q3-Q5 while the screen operates
    entirely in Q1. Average MAE hid that completely.

    It also exposes the far larger problem the architecture work never touched:

        Q1 (lowest kappa)  MAE log10(kappa) 0.3230
        Q4                 MAE log10(kappa) 0.1400
                                            2.3x

    Q1 crystals are soft (median shear modulus 10.5 GPa against 102 GPa in Q5)
    and rare in matbench, so an unweighted loss underfits them.

    Report this alongside the aggregate for anything intended to screen for low
    kappa. A change that improves the average and not Q1 is not an improvement.
    """
    prefactor_t, anharmonic_t, gamma_t = slack_factors(true_log_K, true_log_G)
    prefactor_p, anharmonic_p, gamma_p = slack_factors(pred_log_K, pred_log_G)
    kappa_t = prefactor_t * anharmonic_t
    kappa_p = prefactor_p * anharmonic_p

    ok = (np.isfinite(kappa_t) & np.isfinite(kappa_p)
          & (kappa_t > 0) & (kappa_p > 0))
    kt, kp = kappa_t[ok], kappa_p[ok]
    err = np.abs(np.log10(kp) - np.log10(kt))
    res_K = (pred_log_K - true_log_K)[ok]
    res_G = (pred_log_G - true_log_G)[ok]

    # Quintile edges come from the TRUE kappa values, which are fixed by the
    # split - so every run bins on identical boundaries and the numbers are
    # comparable across runs without any extra bookkeeping.
    edges = np.quantile(kt, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    which = np.digitize(kt, edges[1:-1])

    out = {}
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        out[f"Q{b + 1}"] = {
            "n": int(m.sum()),
            "kappa_mae": float(err[m].mean()),
            "mae_log_K": float(np.abs(res_K[m]).mean()),
            "mae_log_G": float(np.abs(res_G[m]).mean()),
            "median_true_G_GPa": float(np.median(10.0 ** true_log_G[ok][m])),
        }

    # Recall of the true bottom decile by PREDICTED kappa: the screening task,
    # stated directly. Insensitive to calibration, sensitive to ranking.
    n10 = max(1, int(0.10 * len(kt)))
    truth = set(np.argsort(kt)[:n10])
    picked = set(np.argsort(kp)[:n10])
    out["recall_at_10pct"] = len(truth & picked) / n10
    out["n_scored"] = int(ok.sum())
    return out


def kappa_error_budget(true_log_K, true_log_G, pred_log_K, pred_log_G):
    """Split the model's kappa_L error into 'prefactor' and 'gamma' parts.

    This is the headline metric for this whole line of work. Improving
    log10(K) or log10(G) in isolation barely moves kappa if the ratio error
    stays put - so this is what actually has to go down.

    Returns a dict of MAEs in log10(kappa) units, plus the diagnostics that
    explain them (residual correlation and the ratio error).
    """
    pre_t, anh_t, gam_t = slack_factors(true_log_K, true_log_G)
    pre_p, anh_p, gam_p = slack_factors(pred_log_K, pred_log_G)

    # Guard against the handful of crystals where an extreme K/G drives the
    # Poisson ratio out of its physical range and gamma blows up.
    ok = (np.isfinite(gam_t) & np.isfinite(gam_p)
          & (anh_t > 0) & (anh_p > 0) & (pre_t > 0) & (pre_p > 0))

    def mae_log(a, b):
        return float(np.abs(np.log10(a[ok]) - np.log10(b[ok])).mean())

    res_K = pred_log_K - true_log_K
    res_G = pred_log_G - true_log_G

    return {
        "n": int(ok.sum()),
        "mae_log_K": float(np.abs(res_K).mean()),
        "mae_log_G": float(np.abs(res_G).mean()),
        # The ratio error - what gamma is a function of. THE number to watch.
        "mae_log_ratio": float(np.abs(res_K - res_G).mean()),
        # How correlated the two errors are. Higher is better; 0.297 is the
        # separate-models baseline this model exists to beat.
        "residual_corr": float(np.corrcoef(res_K, res_G)[0, 1]),
        "kappa_mae_total": mae_log(pre_p * anh_p, pre_t * anh_t),
        "kappa_mae_prefactor": mae_log(pre_p, pre_t),
        "kappa_mae_anharmonic": mae_log(anh_p, anh_t),
        "gamma_mae": float(np.abs(gam_p[ok] - gam_t[ok]).mean()),
    }
