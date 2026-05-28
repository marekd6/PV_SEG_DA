#!/bin/bash
#SBATCH -J cuda_check
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:01:00

nvidia-smi
