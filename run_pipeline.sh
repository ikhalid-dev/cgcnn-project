#!/usr/bin/env bash
#
# Reproduce every result in this repository, from nothing.
#
#   ./run_pipeline.sh              # full run: build data, train 6 models, predict
#   ./run_pipeline.sh --quick      # 5-epoch smoke test, ~10 min, proves the wiring
#   ./run_pipeline.sh --predict    # skip training, use the committed checkpoints
#
# RUNTIME. On a GPU this is about 45 minutes end to end. On a CPU it is closer
# to NINE HOURS, because training dominates and the model is small enough that
# a laptop CPU gains nothing from vectorisation. If you have no GPU locally,
# use colab/PINK_CGCNN.ipynb - it runs these same scripts on a free Colab T4
# and hands back the trained checkpoints.
#
# WHAT IS AND IS NOT COMMITTED. The six trained checkpoints ARE committed
# (~380 KB each), so --predict reproduces the headline table and every figure
# in about two minutes without training anything. The 240 MB graph cache is NOT
# committed; step 1 rebuilds it from matbench in ~6 minutes.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-full}"
EPOCHS=200
if [ "$MODE" = "--quick" ]; then
    EPOCHS=5
    echo ">>> QUICK MODE: $EPOCHS epochs. This checks the pipeline runs; the"
    echo ">>> resulting numbers are meaningless. Do not report them."
fi

# Pick a device rather than assuming. 02_train.py does this itself with
# --device auto; this is only so the banner tells the truth.
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
echo "=================================================================="
echo " PINK / CGCNN - elastic moduli from crystal structure"
echo " mode: $MODE   device: $DEVICE   epochs: $EPOCHS"
echo "=================================================================="

# --- Step 1: the training set ---------------------------------------------
# Skipped if the cache is already there, since it is deterministic and slow.
if [ ! -f data_full/graphs.pt ]; then
    echo; echo ">>> [1/4] Building the 10,987-crystal training set"
    python scripts/01b_prepare_full_dataset.py
else
    echo; echo ">>> [1/4] data_full/graphs.pt exists, skipping the build"
fi

# --- Step 2: train ---------------------------------------------------------
# Three members per target. --seed varies weight initialisation; --split-seed
# is PINNED at 42 so all members share one train/val/test split - otherwise the
# ensemble's test score would be measured partly on data some members trained
# on. See scripts/05_ensemble.py, which refuses to combine mismatched members.
if [ "$MODE" != "--predict" ]; then
    echo; echo ">>> [2/4] Training six models"
    for TARGET in K_VRH G_VRH; do
        for SPEC in "${TARGET}_full 42 3" "${TARGET}_s1 1 4" "${TARGET}_s2 2 3"; do
            set -- $SPEC
            echo "--- $1 (seed $2, n_conv $3) ---"
            python -u scripts/02_train.py \
                --target "$TARGET" --tag "$1" --seed "$2" --n-conv "$3" \
                --data-dir data_full --epochs "$EPOCHS" --batch-size 128 \
                --lr 0.01 --atom-fea-len 64 --h-fea-len 128 --n-h 1 \
                --split-seed 42 --scheduler cosine --device auto
        done
    done
else
    echo; echo ">>> [2/4] --predict: using the committed checkpoints"
fi

# --- Step 3: score ---------------------------------------------------------
echo; echo ">>> [3/4] Scoring each target, alone and as an ensemble"
for TARGET in K_VRH G_VRH; do
    python scripts/03_evaluate.py --target "$TARGET" --data-dir data_full \
        --tag "${TARGET}_full"
    python scripts/05_ensemble.py --target "$TARGET" --data-dir data_full \
        --tags "${TARGET}_full,${TARGET}_s1,${TARGET}_s2"
done
python scripts/06_summarise.py

# --- Step 4: the deliverable ----------------------------------------------
# K and G for all 1,213 PINK crystals - the input the kappa_L stage consumes.
echo; echo ">>> [4/4] Predicting moduli for the 1,213 PINK crystals"
python scripts/04_predict_moduli.py \
    --k-tag K_VRH_full,K_VRH_s1,K_VRH_s2 \
    --g-tag G_VRH_full,G_VRH_s1,G_VRH_s2

echo
echo "=================================================================="
echo " Done."
echo "   results/RESULTS.md                  headline numbers"
echo "   results/metrics_summary.csv         every model, every split"
echo "   results/pink_moduli_predictions.csv K and G for 1,213 crystals"
echo "   results/parity_*_ens.png            predicted vs DFT"
echo "=================================================================="
