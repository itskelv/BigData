#!/bin/bash
#SBATCH --job-name=bigdata
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH--cpus-per-task=8
#SBATCH --gres=gpu:1

module load cuda/12.1
conda activate bigdataenv
python train_xlmRoBERTa.py
python train_xlmRemBERT.py