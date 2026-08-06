"""
Turning CIF files into crystal graphs the network can eat.
==========================================================

A neural network cannot read a .cif file. This module does the translation:

    CIF file  ->  pymatgen Structure  ->  (node features, edge features, edges)

There are three pieces of machinery here:

    GaussianDistance      expands a single bond length into a smooth vector
    AtomFeaturiser        looks up a fixed feature vector for each element
    structure_to_graph    the actual Structure -> graph conversion
    GraphCacheData        a torch Dataset that reads pre-built graphs from disk
    collate_pool          batches variable-sized graphs into flat tensors

WHY GRAPHS ARE CACHED
---------------------
Turning a crystal into a graph means a periodic neighbour search for every
atom, which is the slowest step in the whole pipeline. Doing it lazily would
cost the better part of an hour for 11,000 crystals, and it would have to be
paid again in every new process. So `scripts/01b_prepare_full_dataset.py` runs
the conversion ONCE and pickles the tensors; GraphCacheData just loads them.

Training and prediction both go through `structure_to_graph`, which matters
more than it sounds: if the training graphs and the prediction graphs were
built by two slightly different code paths, the model would silently be fed
features it was never trained on.

WHY EXPAND DISTANCES INTO A VECTOR?
-----------------------------------
If we fed the raw bond length (say 2.31 A) straight in as one number, the
network would have to learn a highly non-linear response to it from scratch.
Instead we place a row of Gaussian "bumps" along the distance axis and record
how strongly the bond activates each bump. A bond at 2.31 A lights up the
bumps centred near 2.3 and barely touches the one at 5.0. This is the same
idea as a radial basis function expansion, and it makes the distance
dependence much easier for the network to learn.

WHY A FIXED ATOM EMBEDDING?
---------------------------
atom_init.json maps each atomic number to a 92-long binary vector built from
tabulated element properties (group, period, electronegativity, covalent
radius, valence electrons, ...), each one-hot binned. These are NOT learned -
they are the fixed starting description of "what element is this". The
network's first Linear layer then learns how to project them into its own
hidden space.
"""

from __future__ import print_function, division

# torch MUST be imported before numpy. In a conda environment MKL loads its own
# OpenMP runtime first, and the duplicate libiomp5 aborts the process with
# "OMP: Error #15". Every script here orders its imports this way, but the
# package has to be safe to import on its own too - `import cgcnn_scratch`
# should not crash. Do not "tidy" these into alphabetical order.
import torch
from torch.utils.data import Dataset

import json
import os

import numpy as np


class GaussianDistance(object):
    """Expand a scalar distance into a vector of Gaussian basis activations."""

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin, dmax : float
            Range of distances to cover, in Angstrom.
        step : float
            Spacing between neighbouring Gaussian centres. Smaller step means
            a longer, finer-grained feature vector.
        var : float, optional
            Width of each Gaussian. Defaults to `step`, which makes adjacent
            bumps overlap by about the right amount - wide enough that the
            encoding varies smoothly with distance, narrow enough to stay
            informative.
        """
        assert dmin < dmax
        assert dmax - dmin > step
        # e.g. dmin=0, dmax=8, step=0.2 -> 41 centres at 0.0, 0.2, ... 8.0
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        """
        Parameters
        ----------
        distances : np.ndarray of any shape

        Returns
        -------
        np.ndarray with one extra trailing axis of length len(self.filter).
        """
        # Broadcasting does the work: (..., 1) - (n_filters,) -> (..., n_filters)
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomFeaturiser(object):
    """Look up the fixed feature vector for an element, given its atomic number."""

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        # JSON keys are strings; we want int atomic numbers.
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        self._embedding = {key: np.array(value, dtype=float)
                           for key, value in elem_embedding.items()}
        self.atom_types = set(self._embedding.keys())

    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]

    def __len__(self):
        return len(self._embedding)


def structure_to_graph(crystal, ari, gdf, max_num_nbr=12, radius=8):
    """Convert one pymatgen Structure into (atom_fea, nbr_fea, nbr_fea_idx).

    This is the single source of truth for how a crystal becomes a graph. The
    training-set builder and the prediction script both go through here, so the
    features can never drift apart between the two.

    Parameters
    ----------
    crystal : pymatgen Structure
    ari : AtomFeaturiser
    gdf : GaussianDistance
    max_num_nbr : int
        Neighbours kept per atom. Fixed so every atom yields the same shape.
    radius : float
        Neighbour search cutoff in Angstrom.

    Returns
    -------
    (atom_fea, nbr_fea, nbr_fea_idx) as torch tensors.
    """
    # --- NODE FEATURES -------------------------------------------------------
    # One row per atom, looked up by atomic number.
    atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number)
                          for i in range(len(crystal))])
    atom_fea = torch.Tensor(atom_fea)

    # --- EDGES AND EDGE FEATURES --------------------------------------------
    # get_all_neighbors respects periodic boundary conditions, so atoms near a
    # cell face correctly see their periodic images.
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    # Sort each atom's neighbours by distance so "nearest M" is well defined.
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            # Not enough neighbours within the cutoff. Pad the index list with 0
            # and the distance list with radius + 1. The padded distance sits
            # outside every Gaussian, so its expanded feature is ~0 and the
            # padded neighbour contributes almost nothing.
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(
                list(map(lambda x: x[1], nbr)) +
                [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            # Plenty of neighbours - just take the closest max_num_nbr.
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea_idx = np.array(nbr_fea_idx)
    nbr_fea = np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)  # raw distances -> Gaussian vectors

    return atom_fea, torch.Tensor(nbr_fea), torch.LongTensor(nbr_fea_idx)


class GraphCacheData(Dataset):
    """Serve graphs that were already built and pickled to disk.

    The cache file is whatever `scripts/01b_prepare_full_dataset.py` wrote: a
    dict with a list of (atom_fea, nbr_fea, nbr_fea_idx) tuples and the ids that
    go with them. Targets are supplied separately, because the same cache is
    reused for both the bulk and the shear run - only the label column changes.

    Parameters
    ----------
    cache_path : str
        Path to the .pt written by the preparation script.
    targets : dict
        Maps id -> target value (already log10'd by the caller).
    ids : list of str, optional
        Restrict to (and order by) these ids. Defaults to every id in the cache
        that also appears in `targets`.
    """

    def __init__(self, cache_path, targets, ids=None):
        # weights_only=False explicitly. torch 2.6 flipped this default to True,
        # so a cache written under an older torch (or read on a newer one, e.g.
        # Colab) can fail to unpickle. This file is one we wrote ourselves, so
        # there is no untrusted-pickle concern - but the default must not be
        # left to vary with whatever torch the machine happens to have.
        blob = torch.load(cache_path, weights_only=False)
        self.graphs = dict(zip(blob["ids"], blob["graphs"]))

        if ids is None:
            # Keep only ids we have a label for, in the cache's own order.
            ids = [i for i in blob["ids"] if i in targets]
        missing = [i for i in ids if i not in self.graphs]
        assert not missing, f"{len(missing)} ids missing from cache, e.g. {missing[:3]}"

        self.ids = list(ids)
        self.targets = targets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        return (self.graphs[item_id],
                torch.Tensor([float(self.targets[item_id])]),
                item_id)


def collate_pool(dataset_list):
    """Stack a list of variable-sized crystal graphs into flat batch tensors.

    Different crystals have different numbers of atoms, so we cannot simply
    torch.stack them. Instead we concatenate every crystal's atoms into one
    long list and keep a bookkeeping list, crystal_atom_idx, recording which
    rows belong to which crystal. The model's pooling step uses that to average
    the right atoms back together.

    The one subtlety is neighbour indices. Each crystal's nbr_fea_idx refers to
    ITS OWN atom numbering starting at 0. Once we concatenate, crystal #2's
    atoms no longer start at row 0, so every index must be shifted by the
    number of atoms already placed - that is what base_idx tracks.

    Returns
    -------
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), targets, batch_cif_ids
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0

    for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id in dataset_list:
        n_i = atom_fea.shape[0]  # atoms in THIS crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)  # shift into batch space
        # Record the rows this crystal occupies in the concatenated tensor.
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
        torch.stack(batch_target, dim=0), \
        batch_cif_ids


def clean_labels(data_dir, target):
    """Read labels.csv from a data dir and drop rows we cannot take a log of.

    A modulus of zero or below is unphysical and indicates a bad DFT entry;
    left in, log10 would hand the loss a -inf and destroy the whole run.
    """
    import pandas as pd

    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    bad = labels[labels[target] <= 0]
    if len(bad):
        print(f'  dropping {len(bad)} rows with non-positive {target}')
        labels = labels[labels[target] > 0]
    return labels


def load_dataset_for(data_dir, target):
    """Build the Dataset for `target` from the pre-built graph cache.

    Training and evaluation both come through here so they can never disagree
    about what the dataset is or what order it is in - the split indices saved
    in a checkpoint are only meaningful if the dataset is rebuilt identically.
    """
    import numpy as np

    labels = clean_labels(data_dir, target)
    cache_path = os.path.join(data_dir, 'graphs.pt')
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'No graph cache at {cache_path}. Build it first with:\n'
            f'    python scripts/01b_prepare_full_dataset.py')

    print(f'Using pre-built graph cache: {cache_path}')
    targets = dict(zip(labels.mb_id, np.log10(labels[target])))
    return GraphCacheData(cache_path, targets)


class Normalizer(object):
    """Standardise targets to zero mean and unit variance, and undo it later.

    Bulk moduli span roughly 1-400 GPa. Feeding those raw into an MSE loss
    makes the gradients enormous and the learning rate impossible to tune.
    We instead train on z = (y - mean) / std, and convert predictions back with
    denorm() whenever we want a physical number.

    The mean/std MUST come from the training split only - computing them over
    the whole dataset would leak test information into training.
    """

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def to(self, device):
        """Move the statistics onto `device`.

        norm/denorm combine these with the model's output, so they have to live
        wherever the model does or torch raises a device mismatch.
        """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self):
        # Always serialise on CPU. A checkpoint written on a GPU box has to
        # load on a laptop that has no CUDA at all.
        return {'mean': self.mean.cpu(), 'std': self.std.cpu()}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
