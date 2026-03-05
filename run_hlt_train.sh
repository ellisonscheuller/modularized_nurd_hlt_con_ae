#!/bin/bash
# Run HLT contrastive training with checkpoints saved to EOS.
# Usage: bash run_hlt_train.sh [extra args...]
# Example: bash run_hlt_train.sh --epochs 50 --batch_size 512

set -e

source /eos/user/e/escheull/con_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── EOS output directory ────────────────────────────────────────────────────
EOS_CHECKPOINTS="/eos/user/e/escheull/nurd_hlt_checkpoints"
mkdir -p "$EOS_CHECKPOINTS"

# Redirect the hardcoded relative `checkpoints/` dir to EOS via symlink
if [ -d checkpoints ] && [ ! -L checkpoints ]; then
    echo "Moving existing checkpoints/ to EOS..."
    cp -r checkpoints/* "$EOS_CHECKPOINTS/" 2>/dev/null || true
    rm -rf checkpoints
fi
if [ ! -L checkpoints ]; then
    ln -s "$EOS_CHECKPOINTS" checkpoints
fi

# ── Data paths ──────────────────────────────────────────────────────────────
SMCOCKTAIL_TRAIN="/eos/user/e/escheull/smcocktail_1M/hlt_smcocktail_train.pt"
SMCOCKTAIL_TEST="/eos/user/e/escheull/smcocktail_1M/hlt_smcocktail_test.pt"
AE_SCORES_TRAIN="/eos/user/e/escheull/ae_outputs_axis1/ae_scores_bkg_train.pt"
AE_SCORES_TEST="/eos/user/e/escheull/ae_outputs_axis1/ae_scores_bkg_test.pt"

# ── Run ─────────────────────────────────────────────────────────────────────
python train.py \
    --smcocktail_train "$SMCOCKTAIL_TRAIN" \
    --smcocktail_test  "$SMCOCKTAIL_TEST"  \
    --ae_scores_train  "$AE_SCORES_TRAIN"  \
    --ae_scores_test   "$AE_SCORES_TEST"   \
    --model_arch       hlt_contrastive     \
    --in_dataset       hlt                 \
    --project_name     hlt_con             \
    --epochs           30                  \
    --batch_size       256                 \
    --lr               1e-4               \
    --local_testing    0                   \
    "$@"

# Set --local_testing 1 to disable wandb (e.g. for quick debugging runs).
# Any extra args you pass to this script are forwarded to train.py.
