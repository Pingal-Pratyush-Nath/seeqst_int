#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=N10k_m
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# regime: hopping=mu=delta=1, alpha=1 -- the model's own "natural" physics
# defaults (see jw_hamiltonian.jw_long_range_kitaev's docstring); local
# terms dominate here and Pauli is expected to win. Same n/repeats/n-samples
# as the alpha=0 companion run for a direct, apples-to-apples comparison.
# num_repeats=80, N_max=100000 -- predicted ~51h wall clock, comfortable
# margin under the 72h limit.

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
    --n-samples 10 20 30 50 70 100 200 300 500 700 1000 2000 3000 5000 7000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
    --num-repeats 80 \
    --seed 0 \
    --version N10k_m
