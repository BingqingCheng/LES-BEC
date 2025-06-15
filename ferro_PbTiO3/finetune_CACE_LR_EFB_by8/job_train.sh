#!/bin/bash 
#SBATCH --account=drxkp
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=%j-%x.log

source ~/.bashrc
conda activate cace

rm CACE_NNP*
rm avge0.pkl

python train_CACE_LR.py
