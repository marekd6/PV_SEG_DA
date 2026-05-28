#!/bin/bash
#SBATCH -J dck_img
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:10:10

cd /users/project1/pt01299/synt/
module load trytonp/apptainer/1.3.0
apptainer pull pytorch_env.sif docker://pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
