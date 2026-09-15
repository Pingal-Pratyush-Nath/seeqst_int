#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=T10k_M
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# regime: hopping=mu=0, alpha=0 (fully uniform all-to-all pairing) -- the
# non-physical corner where SEEQST/Clifford beat Pauli, see exp_13.py's
# module docstring. num_repeats=80, N_max=100000, 25 n-samples checkpoints (9 more
# than the original 16, added 2026-09-14 for finer log+linear
# resolution) -- predicted ~62-65h wall clock (mean-estimator cost
# plus median_of_means' checkpoint overhead, now slightly higher
# with more checkpoints), still under the 72h limit but with less
# margin than before -- worth watching the actual elapsed time on
# this run.

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
    --n-samples 10 20 30 50 70 100 200 300 500 700 1000 2000 3000 5000 7000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
    --num-repeats 80 \
    --seed 0 \
    --version T10k_M
