#!/bin/bash
#SBATCH -J xls
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:10:10

python3 dir_to_xls.py
