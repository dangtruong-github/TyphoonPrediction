#!/bin/bash

#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -J data
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G

cd /N/u/tnn3/BigRed200/truongchu
module load miniconda
conda activate truongchuenv
module load python/gpu/3.10.10

# Run B.sh
bash run.sh preprocess train eval