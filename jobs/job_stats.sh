#!/bin/bash
#SBATCH -J ft_stats
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:09:10

python3 /users/project1/pt01299/synt/extract_best_epochs.py
