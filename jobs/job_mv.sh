#!/bin/bash
#SBATCH -J mv_cache
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 10:10:10

mv ~/.local $TASK_USER_WORK/
ln -s $TASK_USER_WORK/.local ~/.local
