"""
The Crystal Graph Convolutional Neural Network (CGCNN) architecture.
====================================================================

This is our own heavily-annotated implementation of the architecture from:

    Xie, T. & Grossman, J. C.
    "Crystal Graph Convolutional Neural Networks for an Accurate and
     Interpretable Prediction of Material Properties"
    Phys. Rev. Lett. 120, 145301 (2018)

THE BIG IDEA
------------
A crystal is represented as a GRAPH:

    * every ATOM in the unit cell is a NODE
    * every atom is connected to its M nearest NEIGHBOURS by an EDGE
    * each node carries a feature vector describing "what element am I"
      (electronegativity, group number, radius, ...)
    * each edge carries a feature vector describing "how far away is my
      neighbour" (a Gaussian-expanded distance)

The network then does three things, in order:

    1. EMBED    each atom's raw element features into a hidden vector.
    2. CONVOLVE those hidden vectors several times. Each convolution lets an
                atom look at its neighbours and update itself, so after
                n_conv rounds every atom "knows about" its local chemical
                environment out to n_conv hops.
    3. POOL     all the atom vectors into ONE vector for the whole crystal,
                then push that through a small MLP to predict a number
                (here: log10 of the bulk or shear modulus).

Why pooling by MEAN matters: a crystal is periodic and the unit cell can be
written down with different numbers of atoms. Averaging (rather than summing)
makes the prediction INTENSIVE - it does not change if you double the cell.
That is exactly what we want for a modulus, which is a per-material property,
not a per-atom one.
"""

from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """One round of message passing on the crystal graph.

    In words: each atom builds a message from every (self, neighbour, bond)
    triple, gates those messages, sums them, and adds the result back onto
    itself as a residual update.

    The "gating" is the clever part. For each neighbour we produce TWO
    vectors from the same linear layer:

        * a FILTER, squashed by a sigmoid into [0, 1] - this decides
          *how much* of that neighbour's message should get through
        * a CORE, passed through a softplus - this is *what* the message says

    Multiplying them means the network can learn to ignore irrelevant
    neighbours (filter near 0) and attend to important ones (filter near 1).
    This is the same trick used in LSTM/GRU gates.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Parameters
        ----------
        atom_fea_len : int
            Length of the hidden feature vector carried by each atom.
        nbr_fea_len : int
            Length of the bond (edge) feature vector.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        # The input to this layer is the concatenation of, per (atom, neighbour) pair:
        #   [ my own features | my neighbour's features | the bond features ]
        # which is 2*atom_fea_len + nbr_fea_len numbers wide.
        # We map that to 2*atom_fea_len so we can split the output in half
        # into the "filter" and the "core" described above.
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)

        self.sigmoid = nn.Sigmoid()      # squashes the filter into [0, 1]
        self.softplus1 = nn.Softplus()   # smooth ReLU for the message core
        self.softplus2 = nn.Softplus()   # smooth ReLU for the residual output

        # Two batch-norm layers keep activations well-scaled so training stays
        # stable. bn1 normalises the raw gated features, bn2 normalises the
        # summed neighbour message before it is added back to the atom.
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Shapes use these symbols throughout:
            N = total number of atoms in the whole batch (all crystals stacked)
            M = number of neighbours kept per atom (fixed, e.g. 12)

        Parameters
        ----------
        atom_in_fea : (N, atom_fea_len)   hidden features going in
        nbr_fea     : (N, M, nbr_fea_len) bond features to each neighbour
        nbr_fea_idx : (N, M) long tensor  row index of each neighbour

        Returns
        -------
        (N, atom_fea_len) updated atom features
        """
        N, M = nbr_fea_idx.shape

        # Gather the feature vector of every neighbour. Fancy-indexing a
        # (N, atom_fea_len) tensor with an (N, M) index tensor gives
        # (N, M, atom_fea_len) - i.e. for each atom, the features of its M
        # neighbours, stacked.
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # Build the (self, neighbour, bond) triple for every edge.
        # expand() repeats the atom's own vector M times WITHOUT copying memory.
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea,
             nbr_fea], dim=2)

        # One shared linear layer processes every edge identically. Sharing
        # weights across edges is what makes this a *convolution*: the same
        # local rule applies everywhere in the graph.
        total_gated_fea = self.fc_full(total_nbr_fea)

        # BatchNorm1d expects 2D input, so flatten the (N, M) edge axis,
        # normalise, then restore the shape.
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        # Split into the two halves and apply the gate.
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # Sum the gated messages over the M neighbours -> one message per atom.
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        # RESIDUAL connection: add the message onto the original features
        # rather than replacing them. This lets gradients flow straight through
        # many conv layers and stops early layers from being washed out.
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """The full network: embed -> convolve x n_conv -> pool -> MLP -> scalar."""

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Parameters
        ----------
        orig_atom_fea_len : int
            Width of the RAW element feature vector read from atom_init.json
            (92 for the standard CGCNN element embedding).
        nbr_fea_len : int
            Width of the Gaussian-expanded bond feature vector.
        atom_fea_len : int
            Width of the HIDDEN atom vectors inside the conv layers.
        n_conv : int
            How many message-passing rounds. More rounds = each atom sees a
            larger neighbourhood, but training gets slower and can over-smooth.
        h_fea_len : int
            Width of the fully-connected layers after pooling.
        n_h : int
            Total number of fully-connected layers after pooling.
        classification : bool
            False -> regression, predict one number (our case).
            True  -> two-class classification with log-softmax output.
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification

        # Step 1: project raw element descriptors into the hidden space.
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Step 2: a stack of independent conv layers (each has its own weights).
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        # Step 3: after pooling, widen from atom_fea_len to h_fea_len.
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        # Optional extra hidden layers (only built when n_h > 1).
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])

        # Output head: 1 number for regression, 2 logits for classification.
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Shapes:
            N  = total atoms in the batch
            M  = neighbours per atom
            N0 = number of crystals in the batch

        Parameters
        ----------
        atom_fea        : (N, orig_atom_fea_len) raw element features
        nbr_fea         : (N, M, nbr_fea_len)    bond features
        nbr_fea_idx     : (N, M)                 neighbour row indices
        crystal_atom_idx: list of N0 LongTensors, each holding the row indices
                          belonging to one crystal. This is how we know which
                          atoms to average together in the pooling step.

        Returns
        -------
        (N0, 1) predictions for regression, or (N0, 2) log-probabilities.
        """
        # 1. embed
        atom_fea = self.embedding(atom_fea)

        # 2. convolve - note each layer overwrites atom_fea, so information
        #    propagates one extra hop per iteration
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # 3. pool atoms -> one vector per crystal
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # 4. MLP head.
        #    NOTE: the softplus is applied before AND after conv_to_fc here.
        #    This looks odd but is faithful to the reference implementation,
        #    and the pre-trained AI4Kappa checkpoints were trained this way -
        #    changing it would make those weights incompatible.
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """Average the atom vectors within each crystal.

        The batch stacks every crystal's atoms into one long (N, F) tensor,
        so we need crystal_atom_idx to tell us where each crystal starts and
        stops. Taking the MEAN (not the sum) keeps the output independent of
        how many atoms happen to be in the unit cell - see the module docstring.
        """
        # Sanity check: the index lists must account for every atom exactly once.
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
            atom_fea.data.shape[0]

        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
