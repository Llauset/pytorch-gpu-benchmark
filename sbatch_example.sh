#!/bin/bash
#SBATCH -J bench_IA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300000
#SBATCH --time=04:00:00

module load conda
conda activate pytorch-1.8.0

# Change this variable to selecte the number of GPUs
NB_GPUS=4

python benchmark_models.py -g ${NB_GPUS}
