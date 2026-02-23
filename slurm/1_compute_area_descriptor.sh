#!/bin/bash -l

#SBATCH --job-name compute_area_descriptor
#SBATCH --partition=cpu-single
#SBATCH --time=00:5:00
#SBATCH --mem=8gb
#SBATCH --mail-user danielle.gatland@heigit.org
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --output=slurm/logs/%x.%j.out

cd ~/lulc-utility

uv run --env-file .env lulc/compute_area_descriptor.py