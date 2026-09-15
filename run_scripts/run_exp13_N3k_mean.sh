#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=N3k_m
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# regime: hopping=mu=delta=1, alpha=1 -- the model's own "natural" physics
# defaults. Companion to the N_max=100000 run but capped at N_max=30000 with
# 27 n-samples checkpoints (denser than the 100k runs' 25) and
# num_repeats=250 (vs 80) -- more averaging, since the lower N_max frees up
# a lot of the 72h budget. predicted ~48h wall clock (mean estimator),
# comfortable margin under the 72h limit.

# source ../.venv/bin/activate

python -u experiments/exp_13/exp_13.py \
    --n 10 \
    --hopping 1.0 \
    --mu 1.0 \
    --delta 1.0 \
    --alpha 1.0 \
    --q-values 0.0909090909 0.5 \
    --tune-q \
    --estimator mean \
    --n-samples 10 15 20 30 50 70 100 150 200 300 500 700 1000 1500 2000 3000 5000 7000 10000 12500 15000 17500 20000 22500 25000 27500 30000 \
    --num-repeats 250 \
    --seed 0 \
    --version N3k_m
