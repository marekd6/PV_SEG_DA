#!/bin/bash
#SBATCH -J pv_ftsS3
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END
#SBATCH --time 24:00:00

SCRATCH_DIR=$TASK_USER_WORK
CACHE_DIR="$SCRATCH_DIR/.cache"

mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/torch"

export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_API_KEY=""
export WANDB_DIR="/users/project1/pt01299/synt/wandb_logs"

SWEEP_ID=""

module load trytonp/apptainer/1.3.0

singularity exec --nv \
  -B "$SCRATCH_DIR:$SCRATCH_DIR" \
  docker://pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime \
  bash -c "pip install --user monai==1.5.0 pandas==2.3.1 scikit_learn==1.7.0 segmentation_models_pytorch==0.5.0 timm==1.0.17 transformers==4.53.2 wandb==0.21.0 && python3 /users/project1/pt01299/synt/solar_scope_md2/src/synt_ftS3.py --sweep_id $SWEEP_ID"
