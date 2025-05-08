#!/usr/bin/env bash
#SBATCH -A cs552
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH -C A30  
#SBATCH --mem 16g 
#SBATCH --job-name="preds" 

source activate kalshi
python train.py