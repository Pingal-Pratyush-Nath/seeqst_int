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
    --num-observables 5 \
    --n-samples 1 2 3 4 5 6 7 8 9 10 \
    --num-state-repeats 5 \
    --num-observable-repeats 5 \
    --states haar_random \
    --version n5_m3_exact_random
