#!/bin/bash

#SBATCH --job-name=EXP7_n7m5_z
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=168:00:00

#SBATCH --array=0-0

source ../.venv/bin/activate

python experiments/exp_7/exp_7.py \
    --n 7 \
    --m 5 \
    --observable-type exact \
    --rest-mode z \
    --num-observables 100 \
    --n-samples 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --num-state-repeats 10 \
    --num-observable-repeats 50 \
    --states haar_random \
    --version n7_m5_exact_z
