#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=sparse_m_2
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# source ../.venv/bin/activate

python -u experiments/exp_10/exp_10.py \
    --n 10 \
    --m 2 \
    --estimator mean \
    --num-observables 100 \
    --n-samples 10 30 100 300 1000 3000 5000 10000 15000 20000 25000 30000 \
    --num-state-repeats 20 \
    --num-observable-repeats 20 \
    --states haar_random \
    --seed 0 \
    --version sparse_m_2
