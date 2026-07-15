#!/bin/bash
#SBATCH -J pv_ftsGDA
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mail-type=END
#SBATCH --time 24:00:00

SCRATCH_DIR=$TASK_USER_WORK
CACHE_DIR="$SCRATCH_DIR/.cache"

mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/torch"

export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_API_KEY=""
export WANDB_DIR="/users/project1/pt01299/synt/wandb_logs2"

SWEEP_ID3=""

module load trytonp/apptainer/1.3.0

for i in 0 1 2 3; do
  apptainer exec --nv -B "$SCRATCH_DIR:$SCRATCH_DIR" /users/project1/pt01299/synt/pytorch_env.sif \
    bash -c "pip install --user monai==1.5.0 pandas==2.3.1 scikit_learn==1.7.0 segmentation_models_pytorch==0.5.0 timm==1.0.17 transformers==4.53.2 wandb==0.21.0 && CUDA_VISIBLE_DEVICES=$i python3 /users/project1/pt01299/synt/solar_scope_md2/src/synt_ft_phgda.py" &
done

wait

echo "All sweeps have completed successfully."
