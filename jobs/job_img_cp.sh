#!/bin/bash
#SBATCH -J img_cp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 10:50:10

cp -R /users/project1/${project_id}/synt/dataset_split_all/test/images/. /users/project1/${project_id}/synt/segformer_dataset255_all/test/images/
echo "test"
cp -R /users/project1/${project_id}/synt/dataset_split_all/val/images/. /users/project1/${project_id}/synt/segformer_dataset255_all/val/images/
echo "val"
cp -R /users/project1/${project_id}/synt/dataset_split_all/train/images/. /users/project1/${project_id}/synt/segformer_dataset255_all/train/images/
echo "train"
# du -h --max-depth=1 $TASK_USER_WORK | sort -hr
# echo "done"
