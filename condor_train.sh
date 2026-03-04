#!/bin/bash
set -e

echo "==== Job started: $(date) ===="
echo "Host: $(hostname)"

# Get CVMFS Python 3.11.9 (available on all batch nodes)
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Activate venv — must unset PYTHONHOME/PYTHONPATH so the venv takes precedence
unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "Torch CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda, "| CUDA available:", torch.cuda.is_available())')"

# Move to code directory
cd /afs/cern.ch/user/e/escheull/hlt_con/nurd-hlt

# Set up checkpoints symlink to EOS
EOS_CHECKPOINTS="/eos/user/e/escheull/nurd_hlt_checkpoints"
mkdir -p "$EOS_CHECKPOINTS"
if [ -d checkpoints ] && [ ! -L checkpoints ]; then
    cp -r checkpoints/* "$EOS_CHECKPOINTS/" 2>/dev/null || true
    rm -rf checkpoints
fi
if [ ! -L checkpoints ]; then
    ln -s "$EOS_CHECKPOINTS" checkpoints
fi

# wandb API key (stored in .netrc but set explicitly for batch nodes)
export WANDB_API_KEY=24d1d60ce26563c74d290d7b487cb104fc251271

$PYTHON train.py \
    --smcocktail_train /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \
    --smcocktail_test  /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt  \
    --ae_scores_train  /eos/user/e/escheull/ae_outputs_noZB/ae_scores_bkg_train.pt \
    --ae_scores_test   /eos/user/e/escheull/ae_outputs_noZB/ae_scores_bkg_test.pt  \
    --model_arch       hlt_contrastive \
    --in_dataset       hlt             \
    --project_name     hlt_con         \
    --epochs           30              \
    --batch_size       512             \
    --lr               1e-4            \
    --joint_indep      1               \
    --critic_epochs    5               \
    --nurd_lambda      1.0             \
    --grad_clip        1.0             \
    --local_testing    0

echo "==== Job finished: $(date) ===="
