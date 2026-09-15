#!/bin/bash

#SBATCH --job-name=EXP7_n5m3_random
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=168:00:00

#SBATCH --array=0-0

source ../.venv/bin/activate

python experiments/exp_7/exp_7.py \
    --n 5 \
    --m 3 \
    --observable-type exact \
    --rest-mode random \
    --num-observables 50 \
    --n-samples 10 50 100 200 400 600 800 1000 \
    --num-state-repeats 10 \
    --num-observable-repeats 50 \
    --states haar_random \
    --version n5_m3_exact_random
