#!/bin/bash
#SBATCH -J osf
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 00:20:10

cd /users/project1/pt01299/synt
unzip segformer_dataset255_all.zip
unzip dataset_split_all.zip
