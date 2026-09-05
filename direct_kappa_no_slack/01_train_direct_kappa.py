#!/usr/bin/env python3
"""
DIRECT KAPPA, NO SLACK - train a CGCNN to predict log10(kappa_L) from the graph.
================================================================================

    python direct_kappa_no_slack/01_train_direct_kappa.py --seed 42 --tag s42

WHAT THIS IS
--------------
Every other training script in this repository predicts elastic moduli and then
pushes them through Slack's equation to obtain a lattice thermal conductivity.
This one does not. It has a single head, its target is log10 of AFLOW's
tabulated kappa, and the Slack formula appears nowhere in this file.

The rationale, the pre-registered outcomes and the honest limits are in this
directory's README.md. Read that before interpreting anything this script
prints.

THE ONE VARIABLE
------------------
Held identical to scripts/cgcnn/37_train_gamma.py: the same cached AFLOW
graphs, the same 70/15/15 split at split_seed=42, the same trunk widths, the
same huber loss, epochs, batch size, optimiser, schedule and early stopping.
The only thing that differs is the head target and the absence of the formula.
If any CONFIG value below drifts from 37's, the comparison in 02 stops being a
one-variable experiment - so they are grouped and labelled for checking.

WHY THE TRUNK IS RE-IMPLEMENTED HERE
--------------------------------------
`JointCrystalGraphConvNet` hardcodes `self.head1 = self.heads[1]` in its
constructor and therefore cannot be built with one head. Rather than modify a
module that six other training scripts depend on, the trunk is rebuilt here,
layer for layer, from the SAME `EdgeDropoutConvLayer` primitive. The layer
list, widths, activation placement and pooling are copied from
cgcnn_scratch/joint.py so the two models differ only where they are meant to.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- data ---------------------------------------------------------------
    "data_dir": "data_full",
    "cache_file": "gamma_graphs.pt",       # cached AFLOW crystal graphs (5,563)
    "labels_file": "gamma_labels.csv",     # carries kappa_agl, the target
    "target_column": "kappa_agl",          # what this model predicts, in log10
    "out_dir": "direct_kappa_no_slack/models",
    "pred_dir": "direct_kappa_no_slack/results/csv",
    "tag": "s42",                          # names the checkpoint/summary/predictions

    # ---- architecture -- MUST MATCH 37_train_gamma.py --------------------
    "atom_fea_len": 64,
    "n_conv": 3,
    "h_fea_len": 128,
    "n_shared_fc": 1,
    "n_head_fc": 1,
    "head_fea_len": 64,
    "dropout": 0.10,

    # ---- optimisation -- MUST MATCH 37_train_gamma.py --------------------
    "loss_fn": "huber",
    "huber_delta": 1.0,
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 0.01,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "cosine",
    "early_stop_patience": 80,

    # ---- split -- MUST MATCH 37_train_gamma.py ---------------------------
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,                      # which crystals are held out
    "init_seed": 42,                       # weight init / shuffling; --seed overrides THIS

    # ---- runtime ------------------------------------------------------------
    "device": "auto",                      # "auto" | "cpu" | "cuda"
    "num_workers": 0,
    "print_every": 25,                     # epochs between progress lines
}
# =============================================================================

import argparse             # --seed / --tag and one flag per CONFIG key
import json                 # the summary file
import os                   # paths and directory creation
import sys                  # sys.path manipulation
import time                 # wall-clock timing

# torch first: MKL loads its own OpenMP runtime and a duplicate libiomp5 aborts
# the process if numpy/pandas get in first. Project-wide rule, not optional.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

# direct_kappa_no_slack/ -> project root, so `cgcnn_scratch` imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

from cgcnn_scratch.data import collate_pool          # noqa: E402
from cgcnn_scratch.joint import EdgeDropoutConvLayer  # noqa: E402

from importlib import import_module                   # noqa: E402
_gamma = import_module("37_train_gamma")              # split_indices ONLY - no Slack import


# -----------------------------------------------------------------------------
#  The model: the joint trunk, one head, no formula
# -----------------------------------------------------------------------------
class DirectKappaCGCNN(nn.Module):
    """CGCNN trunk with a single scalar head predicting log10(kappa).

    Layer for layer identical to JointCrystalGraphConvNet's trunk - same
    embedding, same EdgeDropoutConvLayer stack, same mean pooling, same
    doubled-softplus conv_to_fc, same shared FC block, same dropout at the
    fork - so that any difference measured in 02 is attributable to the head
    target and the missing Slack step, not to a different network.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128,
                 n_shared_fc=1, n_head_fc=1, head_fea_len=64, dropout=0.0):
        super().__init__()
        # Project the fixed 92-dim element descriptors into the hidden space.
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # edge_dropout=0.0 makes EdgeDropoutConvLayer behave exactly like the
        # original ConvLayer (the mask branch is skipped), which is what 37 uses.
        self.convs = nn.ModuleList([
            EdgeDropoutConvLayer(atom_fea_len=atom_fea_len,
                                 nbr_fea_len=nbr_fea_len, edge_dropout=0.0)
            for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.shared_fcs = nn.ModuleList(
            [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_shared_fc)])
        self.shared_acts = nn.ModuleList(
            [nn.Softplus() for _ in range(n_shared_fc)])
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # The one head. Same small-MLP shape the joint model builds per head.
        layers, width = [], h_fea_len
        for _ in range(n_head_fc):
            layers += [nn.Linear(width, head_fea_len), nn.Softplus()]
            width = head_fea_len
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        """Mean over each crystal's atoms - keeps the output intensive."""
        assert sum(len(m) for m in crystal_atom_idx) == atom_fea.shape[0]
        return torch.cat([torch.mean(atom_fea[m], dim=0, keepdim=True)
                          for m in crystal_atom_idx], dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        crys = self.pooling(atom_fea, crystal_atom_idx)
        # The doubled softplus mirrors the original implementation exactly.
        crys = self.conv_to_fc(self.conv_to_fc_softplus(crys))
        crys = self.conv_to_fc_softplus(crys)
        for fc, act in zip(self.shared_fcs, self.shared_acts):
            crys = act(fc(crys))
        crys = self.dropout(crys)
        return self.head(crys)               # (N0, 1)


class ScalarDataset(torch.utils.data.Dataset):
    """Cached graphs paired with ONE scalar target, in the cache's own id order.

    JointGraphCacheData exists but returns a target vector plus a weight column
    and assumes >= 2 heads. A single-target wrapper is three lines and keeps
    the batch shape unambiguous, so it is written out rather than worked around.
    """

    def __init__(self, cache_path, targets):
        # weights_only=False explicitly: torch 2.6 flipped the default and a
        # cache written under an older torch will not unpickle otherwise. This
        # is a file this project wrote itself, so there is no untrusted risk.
        blob = torch.load(cache_path, weights_only=False)
        graphs = dict(zip(blob["ids"], blob["graphs"]))
        self.ids = [i for i in blob["ids"] if i in targets]
        self.graphs = [graphs[i] for i in self.ids]
        self.y = torch.tensor([[targets[i]] for i in self.ids], dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.graphs[i], self.y[i], self.ids[i]


def parse_args():
    """--seed and --tag, plus a flag for every CONFIG key.

    Flags are appended AFTER any common flags by the caller, because argparse
    keeps the LAST occurrence of a repeated option - the ordering trap that
    silently reverted four of six sweeps in round 2.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", type=int, default=None,
                   help="overrides init_seed (weight init); split_seed is left alone "
                        "so every seed is held out on the SAME crystals")
    for k, v in CONFIG.items():
        if k == "init_seed":
            continue
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    return p.parse_args()


def build_loss(cfg):
    """Same three options 37 exposes; huber is round 3's winner on this project."""
    if cfg["loss_fn"] == "huber":
        return nn.HuberLoss(delta=cfg["huber_delta"])
    if cfg["loss_fn"] == "l1":
        return nn.L1Loss()
    return nn.MSELoss()


def pick_device(requested):
    """'auto' resolves to cuda only if it is present AND this torch supports it.

    torch.cuda.is_available() returns True on an incompatible GPU (the P100
    sm_60 trap that killed a full 12-run grid on Kaggle), so the architecture
    list is checked too.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list():
            return torch.device("cuda")
        print(f"  GPU sm_{cap[0]}{cap[1]} unsupported by this torch build - using CPU")
    return torch.device("cpu")


def run_epoch(loader, model, criterion, mean, std, optimizer=None):
    """One pass. Returns (mean loss, predictions, truths, ids) in DENORMALISED log10."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total, n, preds, trues, ids = 0.0, 0, [], [], []

    with torch.set_grad_enabled(training):
        for inputs, target, bid in loader:
            atom_fea, nbr_fea, nbr_idx, crys_idx = inputs
            device = mean.device
            out = model(atom_fea.to(device), nbr_fea.to(device),
                        nbr_idx.to(device), [c.to(device) for c in crys_idx])
            target = target.to(device)
            loss = criterion(out, (target - mean) / std)     # z-scored target space

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * target.shape[0]
            n += target.shape[0]
            preds.append((out.detach() * std + mean).cpu().numpy())
            trues.append(target.detach().cpu().numpy())
            ids.extend(bid)

    return total / max(n, 1), np.vstack(preds).ravel(), np.vstack(trues).ravel(), ids


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    for k in cfg:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.seed is not None:
        cfg["init_seed"] = args.seed          # ONLY the init seed; the split is fixed

    torch.manual_seed(cfg["init_seed"])
    np.random.seed(cfg["init_seed"])

    device = pick_device(cfg["device"])
    out_dir = os.path.join(PROJECT_ROOT, cfg["out_dir"])
    pred_dir = os.path.join(PROJECT_ROOT, cfg["pred_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print("=" * 78)
    print(f"  DIRECT KAPPA (no Slack)  tag={cfg['tag']}  seed={cfg['init_seed']}  "
          f"device={device}")
    print("=" * 78)

    # ---- data ---------------------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])
    meta = pd.read_csv(os.path.join(data_dir, cfg["labels_file"])).set_index("gid")
    kappa = meta[cfg["target_column"]].astype(float)
    if (kappa <= 0).any():
        # log10 of a non-positive kappa is undefined. AFLOW's export has none,
        # but a silent NaN here would poison the normalizer and every metric
        # downstream, so it is checked rather than assumed.
        sys.exit(f"ERROR: {int((kappa <= 0).sum())} crystals have "
                 f"{cfg['target_column']} <= 0; cannot take log10.")
    targets = {i: float(np.log10(v)) for i, v in kappa.items()}

    dataset = ScalarDataset(os.path.join(data_dir, cfg["cache_file"]), targets)
    tr, va, te = _gamma.split_indices(len(dataset), cfg["train_ratio"],
                                      cfg["val_ratio"], cfg["split_seed"])
    print(f"  {len(dataset)} crystals -> {len(tr)} train / {len(va)} val / "
          f"{len(te)} test  (split_seed={cfg['split_seed']})")

    lk = dict(batch_size=cfg["batch_size"], collate_fn=collate_pool,
              num_workers=cfg["num_workers"])
    loaders = {
        "train": DataLoader(dataset, sampler=SubsetRandomSampler(tr), **lk),
        "val": DataLoader(dataset, sampler=SubsetRandomSampler(va), **lk),
    }
    # The test loader is SEQUENTIAL over a Subset so predictions come back in a
    # known crystal order and can be written to CSV with their ids attached.
    test_subset = torch.utils.data.Subset(dataset, list(te))
    loaders["test"] = DataLoader(test_subset, sampler=SequentialSampler(test_subset), **lk)

    # Normaliser fitted on TRAIN ONLY - using all rows would leak the held-out
    # distribution into the target scaling.
    train_y = dataset.y[list(tr)]
    mean = train_y.mean().to(device)
    std = train_y.std().to(device)
    print(f"  target log10({cfg['target_column']}): train mean {mean.item():.3f} "
          f"std {std.item():.3f}")

    # ---- model --------------------------------------------------------------
    (sample_g, _, _) = dataset[0]
    model = DirectKappaCGCNN(
        sample_g[0].shape[-1], sample_g[1].shape[-1],
        atom_fea_len=cfg["atom_fea_len"], n_conv=cfg["n_conv"],
        h_fea_len=cfg["h_fea_len"], n_shared_fc=cfg["n_shared_fc"],
        n_head_fc=cfg["n_head_fc"], head_fea_len=cfg["head_fea_len"],
        dropout=cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DirectKappaCGCNN: {n_params:,} parameters, 1 head, no Slack")

    criterion = build_loss(cfg)
    opt_cls = optim.AdamW if cfg["optimizer"] == "adamw" else optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["learning_rate"],
                        weight_decay=cfg["weight_decay"])
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] == "cosine" else None)

    # ---- train --------------------------------------------------------------
    best_val, best_state, best_epoch, since = float("inf"), None, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        tr_loss, _, _, _ = run_epoch(loaders["train"], model, criterion,
                                     mean, std, optimizer)
        va_loss, _, _, _ = run_epoch(loaders["val"], model, criterion, mean, std)
        if scheduler:
            scheduler.step()
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})

        if va_loss < best_val:
            best_val, best_epoch, since = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg["early_stop_patience"]:
                print(f"  early stop at epoch {epoch} "
                      f"(no val improvement for {cfg['early_stop_patience']})")
                break
        if epoch % cfg["print_every"] == 0:
            print(f"    epoch {epoch:>3}  train {tr_loss:.4f}  val {va_loss:.4f}")

    model.load_state_dict(best_state)
    seconds = time.time() - t0
    print(f"  best epoch {best_epoch}, val loss {best_val:.4f}, {seconds:.0f}s")

    # ---- score every split with the restored best weights -------------------
    results = {}
    for split in ("train", "val", "test"):
        _, pred, true, ids = run_epoch(loaders[split], model, criterion, mean, std)
        err = np.abs(pred - true)
        results[split] = {
            "n": int(len(true)),
            # Mean AND median: this project's error distribution is skewed
            # enough that quoting one alone has misled before.
            "mae_log10": float(np.mean(err)),
            "median_ae_log10": float(np.median(err)),
            "p95_ae_log10": float(np.percentile(err, 95)),
            "rmse_log10": float(np.sqrt(np.mean(err ** 2))),
        }
        if split == "test":
            pd.DataFrame({"gid": ids,
                          "true_log10_kappa": true,
                          "pred_log10_kappa": pred}).to_csv(
                os.path.join(pred_dir, f"01_direct_kappa_test_{cfg['tag']}.csv"),
                index=False)

    # test/train ratio: this project's own diagnostic for underfitting. Round 9
    # sat at 1.08-1.23 on AFLOW, which is the band where the model never learned
    # rather than the band where it generalised. Printed so 02 can flag it.
    ratio = results["test"]["mae_log10"] / max(results["train"]["mae_log10"], 1e-9)
    print()
    print(f"  {'split':<8}{'n':>6}{'MAE':>9}{'median':>9}{'p95':>9}")
    for split in ("train", "val", "test"):
        r = results[split]
        print(f"  {split:<8}{r['n']:>6}{r['mae_log10']:>9.4f}"
              f"{r['median_ae_log10']:>9.4f}{r['p95_ae_log10']:>9.4f}")
    print(f"  test/train ratio {ratio:.2f}"
          + ("   <- UNDERFITTING BAND (cf. round 9's 1.08-1.23 on AFLOW)"
             if ratio < 1.35 else ""))

    # ---- persist ------------------------------------------------------------
    torch.save({"state_dict": model.state_dict(),
                "normalizer": {"mean": mean.cpu(), "std": std.cpu()},
                "config": cfg, "best_epoch": best_epoch,
                "split": {"train": list(map(int, tr)), "val": list(map(int, va)),
                          "test": list(map(int, te))}},
               os.path.join(out_dir, f"model_direct_kappa_{cfg['tag']}.pth"))
    with open(os.path.join(out_dir, f"summary_direct_kappa_{cfg['tag']}.json"), "w") as fh:
        json.dump({"tag": cfg["tag"], "config": cfg, "best_epoch": best_epoch,
                   "seconds": seconds, "n_params": n_params,
                   "test_over_train": ratio, "results": results,
                   "history": history}, fh, indent=2)
    print()
    print(f"  wrote {cfg['out_dir']}/model_direct_kappa_{cfg['tag']}.pth")
    print(f"  wrote {cfg['pred_dir']}/01_direct_kappa_test_{cfg['tag']}.csv")


if __name__ == "__main__":
    main()
