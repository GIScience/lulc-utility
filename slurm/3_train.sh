#!/bin/bash -l

#SBATCH --job-name train
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --mail-user danielle.gatland@heigit.org
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --output=slurm/logs/%x.%j.out

# TODO: not sure how the config should look for multiple GPUs

cd ~/lulc-utility

uv run --env-file .env lulc/train.py