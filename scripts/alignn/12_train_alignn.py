#!/usr/bin/env python3
"""
STEP 12 - Train ALIGNN on the same matbench split our CGCNN ensemble used.
================================================================================

    python scripts/12_train_alignn.py --target bulk_modulus_kv
    python scripts/12_train_alignn.py --target shear_modulus_gv

    # quick local smoke test before trusting a long/GPU run:
    python scripts/12_train_alignn.py --target bulk_modulus_kv \
        --n-train 200 --n-val 40 --n-test 40 --epochs 2 --out-dir /tmp/alignn_smoke

WHY keep_data_order=True
-------------------------
scripts/11_prepare_alignn_data.py already reordered the data into
train-then-val-then-test order using OUR OWN split_indices() (the exact
function that produced the CGCNN ensemble's split) - not ALIGNN's own
split_seed, which shuffles with a different RNG (Python's stdlib `random`,
not numpy) and would silently produce a different partition even given "the
same" seed number. Setting keep_data_order=True tells ALIGNN's own
get_train_val_loaders "do not shuffle again, the order you were given already
IS train/val/test" - the officially-supported way to hand this API a
pre-computed split rather than fighting its internal one.

WHY calculate_gradient=False
------------------------------
ALIGNNAtomWiseConfig defaults to calculate_gradient=True, which asks the model
to differentiate the graph-level output with respect to atomic positions -
meaningful for a force-field task (dE/dR), meaningless here: bulk/shear
modulus has no associated per-atom force or stress label to backprop against,
so leaving this on would compute a gradient nothing in this task supervises.

WHY model="alignn_atomwise_pure" AND neighbor_strategy="pure_torch"
-----------------------------------------------------------------------
This install has no working `dgl` (pip's own dependency tree confirms ALIGNN
does not require it - and installing it separately was tried and reverted: its
prebuilt wheel for this exact torch build fails to load, an ABI mismatch, a
different flavour of the same "two compiled extensions fighting over the
same runtime" problem this project has hit before with MKL/libiomp5). Despite
that, the DEFAULT code path (model="alignn_atomwise", default
neighbor_strategy="k-nearest") unconditionally calls literal `dgl.graph(...)`
deep inside alignn/graphs.py and crashes with dgl=None - confirmed directly by
running this exact smoke test before this comment was written, not assumed
from ALIGNN's release notes (which say the DGL dependency was dropped; the
installed package's default *code path* disagrees with its own release
notes, same category of doc-vs-code mismatch this project keeps finding
elsewhere). The fix, also confirmed by inspection of alignn/graphs.py: passing
neighbor_strategy="pure_torch" routes through alignn/torch_graph_builder.py
instead, which returns its own TorchGraph object directly whenever dgl is
None rather than converting to one - and alignn_atomwise_pure.py's
ALIGNNAtomWisePure model is the one written to consume TorchGraph natively.
Both must be set together; either alone leaves a DGL-shaped object meeting a
model (or vice versa) that does not expect it.

--resume: RECOVERING FROM A LOST SESSION WITHOUT STARTING OVER AT EPOCH 0
------------------------------------------------------------------------------
Motivated by a real incident: a Colab session was lost mid-run (the VM got
reclaimed - see colab/run_alignn_cli.py's own history), losing all progress
on a 150-epoch run. ALIGNN already writes current_model.pt (weights) and
current_state.pt (optimizer/scheduler/epoch/best_loss) after EVERY epoch -
alignn/train.py's own train_dgl() already knows how to resume from
current_state.pt when config.resume_checkpoint=True, it just was not wired
up here. --resume rebuilds the model from out_dir/config.json +
current_model.pt (the same two-step restore alignn's own train_alignn.py
uses for --restart_model_path, reused rather than reimplemented differently)
and sets resume_checkpoint=True so train_dgl() picks the optimizer/scheduler/
epoch back up from current_state.pt on its own. If out_dir has no checkpoint
yet (first-ever launch), --resume is a harmless no-op, not an error - there
is nothing to resume from, so it just starts fresh.

This only helps if current_model.pt/current_state.pt/config.json actually
survive whatever interrupted the run - i.e. they need to have been copied
somewhere OUTSIDE the lost VM before it was lost. That is
colab/run_alignn_cli.py's job (periodic --wait downloads), not this script's.
"""

import argparse
import os
import sys
import warnings
from importlib import import_module

import torch  # noqa: F401  (see cgcnn_scratch/data.py - import order matters)

import pickle

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# scripts/cgcnn/, not this file's own scripts/alignn/ - see 11's identical note.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
_train = import_module("02_train")  # reuse pick_device(), same as scripts/cgcnn/05


def load_split(cache_path, n_train_override, n_val_override, n_test_override):
    """Load the cached, pre-split dataset_array; optionally take a small
    prefix of each segment for a fast smoke test rather than the full run.
    """
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)
    data, n_train, n_val, n_test = (cache["dataset_array"], cache["n_train"],
                                    cache["n_val"], cache["n_test"])

    if n_train_override is None:
        return data, n_train, n_val, n_test

    n_tr, n_va, n_te = n_train_override, n_val_override, n_test_override
    subset = (data[:n_tr] + data[n_train:n_train + n_va]
             + data[n_train + n_val:n_train + n_val + n_te])
    return subset, n_tr, n_va, n_te


def load_resume_model(out_dir, device):
    """Rebuild the model and restore its weights from a previous run's
    checkpoint - the same two-step restore alignn's own train_alignn.py
    uses for --restart_model_path (config.json for architecture,
    current_model.pt for weights), reused rather than reimplemented
    differently. Returns None (not an error) if no checkpoint is there yet.

    Only restores WEIGHTS. Optimizer/scheduler/epoch are restored separately,
    inside alignn's own train_dgl(), from current_state.pt, triggered by
    config.resume_checkpoint=True - not duplicated here.
    """
    import json
    model_path = os.path.join(out_dir, "current_model.pt")
    config_path = os.path.join(out_dir, "config.json")
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        return None

    from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig

    with open(config_path) as fh:
        saved = json.load(fh)
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**saved["model"]))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Resuming model weights from {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True,
                        choices=["bulk_modulus_kv", "shear_modulus_gv"])
    parser.add_argument("--data-cache", default=os.path.join(PROJECT_ROOT, "data_full", "alignn_data.pkl"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alignn-layers", type=int, default=4)
    parser.add_argument("--gcn-layers", type=int, default=4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--embedding-features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-train", type=int, default=None,
                        help="override for a smoke test; default uses the full cached split")
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--n-early-stopping", type=int, default=None,
                        help="stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--resume", action="store_true",
                        help="resume from --out-dir's current_model.pt/current_state.pt "
                             "if present; harmless no-op if there's nothing to resume from")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", "alignn", f"alignn_{args.target}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Training ALIGNN on {args.target} ===\n")
    device = _train.pick_device(args.device)
    print(f"Device: {device}")

    data, n_train, n_val, n_test = load_split(
        args.data_cache, args.n_train, args.n_val, args.n_test)
    print(f"Data: {n_train} train / {n_val} val / {n_test} test "
         f"({len(data)} total, from {args.data_cache})")

    from alignn.config import TrainingConfig
    from alignn.data import get_train_val_loaders
    from alignn.train import train_dgl

    config = TrainingConfig(
        target=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_train=n_train, n_val=n_val, n_test=n_test,
        keep_data_order=True,
        id_tag="jid",
        criterion="mse",
        optimizer="adamw",
        scheduler="onecycle",
        random_seed=args.seed,
        num_workers=0,
        pin_memory=(device == "cuda"),
        write_predictions=True,
        save_dataloader=False,
        output_dir=out_dir,
        n_early_stopping=args.n_early_stopping,
        model={
            "name": "alignn_atomwise_pure",
            "alignn_layers": args.alignn_layers,
            "gcn_layers": args.gcn_layers,
            "hidden_features": args.hidden_features,
            "embedding_features": args.embedding_features,
            # get_train_val_loaders' default atom_features="cgcnn" is a
            # 92-element one-hot table (elements 1-92, the same atom_init.json
            # convention our own CGCNN uses) - the model's atom_input_features
            # must match that width or the first linear layer's shapes don't
            # line up. Confirmed by hitting exactly that shape-mismatch error
            # with the config's own atom_input_features=1 default.
            "atom_input_features": 92,
            "output_features": 1,
            "calculate_gradient": False,
            "classification": False,
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
        },
        resume_checkpoint=args.resume,
    )

    resume_model = None
    if args.resume:
        resume_model = load_resume_model(out_dir, device)
        if resume_model is None:
            print(f"--resume given but no checkpoint found in {out_dir} - "
                 f"starting fresh (this is expected on a first-ever launch).")
            config.resume_checkpoint = False

    print("\nBuilding dataloaders (line-graph construction - this is the slow part)...")
    # use_pure_torch=True picks the non-lmdb-C-library storage backend;
    # neighbor_strategy="pure_torch" is the separate, also-necessary flag that
    # avoids literal dgl.graph() construction inside Graph.atom_dgl_multigraph
    # (see the module docstring - both are required, confirmed by testing).
    (train_loader, val_loader, test_loader, prepare_batch) = get_train_val_loaders(
        dataset_array=data,
        target=args.target,
        id_tag="jid",
        batch_size=args.batch_size,
        n_train=n_train, n_val=n_val, n_test=n_test,
        keep_data_order=True,
        output_dir=out_dir,
        workers=0,
        pin_memory=(device == "cuda"),
        use_pure_torch=True,
        neighbor_strategy="pure_torch",
        # Without this, the LMDB caches land at ./train_data, ./val_data,
        # ./test_data (relative to wherever the script was launched from,
        # not out_dir) - harmless but messy, and it means two targets run
        # from the same cwd redundantly rebuild each other's cache. Found by
        # noticing exactly those three directories appear at the repo root
        # after running the smoke test.
        filename=os.path.join(out_dir, ""),
    )

    print(f"\nTraining for {args.epochs} epochs -> {out_dir}\n")
    train_dgl(config, model=resume_model,
             train_val_test_loaders=(train_loader, val_loader, test_loader, prepare_batch))

    print(f"\nDone. Checkpoint, config, and test-set predictions written to {out_dir}/")


if __name__ == "__main__":
    main()
