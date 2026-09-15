#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=13_20_MED
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# source ../.venv/bin/activate

python -u experiments/exp_13/exp_13.py \
    --n 10 \
    --hopping 0.0 \
    --mu 0.0 \
    --delta 1.0 \
    --alpha 0.0 \
    --q-values 0.0909090909 0.5 \
    --tune-q \
    --estimator median_of_means \
    --n-samples 10 30 100 300 1000 3000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
    --num-repeats 20 \
    --seed 0 \
    --version 13_20_MED
