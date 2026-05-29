#!/bin/bash
#SBATCH -J pv_da
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
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

module load trytonp/apptainer/1.3.0

singularity exec --nv \
  -B "$SCRATCH_DIR:$SCRATCH_DIR" \
  docker://nvcr.io/nvidia/pytorch:21.11-py3 \
  bash -c "pip install --upgrade 'pydantic>=2.0' && pip install --user transformers monai wandb && python3 /users/project1/pt01299/synt/solar_scope_md2/src/simple_ft_pnl_splt.py"
