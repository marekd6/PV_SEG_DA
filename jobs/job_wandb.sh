#!/bin/bash
#SBATCH -J wandb
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:09:50

module load trytonp/python3/3.13.0

export WANDB_API_KEY=""
export WANDB_DIR="/users/project1/pt01299/synt/wandb_logs"

#pip install wandb
python3 /users/project1/pt01299/synt/solar_scope_md2/src/synt_ftS5wandb.py
