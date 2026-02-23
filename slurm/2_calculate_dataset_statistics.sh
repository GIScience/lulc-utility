#!/bin/bash -l

#SBATCH --job-name calculate_dataset_statistics
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --time=00:20:00
#SBATCH --mem=32gb
#SBATCH --mail-user danielle.gatland@heigit.org
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --output=slurm/logs/%x.%j.out

# Used 16.5GB memory, 1xA40, 8 min (with 4 workers and batch size of 8)

cd ~/lulc-utility

uv run --env-file .env lulc/calculate_dataset_statistics.py