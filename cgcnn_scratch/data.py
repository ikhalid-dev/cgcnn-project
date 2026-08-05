"""
Turning CIF files into crystal graphs the network can eat.
==========================================================

A neural network cannot read a .cif file. This module does the translation:

    CIF file  ->  pymatgen Structure  ->  (node features, edge features, edges)

There are three pieces of machinery here:

    GaussianDistance      expands a single bond length into a smooth vector
    AtomFeaturiser        looks up a fixed feature vector for each element
    CIFData               a torch Dataset that ties it all together
    collate_pool          batches variable-sized graphs into flat tensors

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

import csv
import functools
import json
import os
import random

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


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


class CIFData(Dataset):
    """A torch Dataset that yields (graph, target, id) for each CIF.

    Expects a directory laid out like:

        root_dir/
            atom_init.json      <- the element embedding table
            id_prop.csv         <- two columns: filename, target value
            mp-1234.cif
            mp-5678.cif
            ...

    IMPORTANT - this differs from the original Xie CGCNN in two ways, chosen to
    match the AI4Kappa fork so that its pre-trained checkpoints stay usable:

        * id_prop.csv HAS a header row, which we skip.
        * the first column is the FULL FILENAME including the .cif suffix,
          not a bare id.

    Getting either wrong fails quietly - a missing header silently eats your
    first structure - so we validate explicitly below.
    """

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        """
        Parameters
        ----------
        root_dir : str
            Directory laid out as described above.
        max_num_nbr : int
            How many nearest neighbours to keep per atom. Fixed so that every
            atom produces the same shaped tensor and batching is easy.
        radius : float
            Search cutoff in Angstrom. Neighbours further than this are never
            considered, even if an atom has fewer than max_num_nbr of them.
        dmin, step : float
            Passed to GaussianDistance for the bond feature expansion.
        random_seed : int
            Seeds the shuffle of the id/property list so runs are reproducible.
        """
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), f'root_dir does not exist: {root_dir}'

        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            next(reader)  # discard the header row
            self.id_prop_data = [row for row in reader]

        # Shuffle once, deterministically. Downstream code slices this list into
        # train/val/test, so shuffling here removes any ordering bias that may
        # be baked into the CSV (e.g. sorted by formula or by id).
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomFeaturiser(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    # Cache parsed graphs in RAM. Building a graph means a full neighbour
    # search, which is the single slowest step in the whole pipeline - and every
    # epoch asks for the same structures again. With a few hundred crystals the
    # entire dataset fits in memory comfortably.
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id))

        # --- NODE FEATURES ---------------------------------------------------
        # One row per atom, looked up by atomic number.
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        # --- EDGES AND EDGE FEATURES ----------------------------------------
        # get_all_neighbors respects periodic boundary conditions, so atoms
        # near a cell face correctly see their periodic images.
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        # Sort each atom's neighbours by distance so "nearest M" is well defined.
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                # Not enough neighbours within the cutoff. Pad the index list
                # with 0 and the distance list with radius + 1. The padded
                # distance sits outside every Gaussian, so its expanded feature
                # is ~0 and the padded neighbour contributes almost nothing.
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                # Plenty of neighbours - just take the closest max_num_nbr.
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)  # raw distances -> Gaussian vectors

        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


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

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
