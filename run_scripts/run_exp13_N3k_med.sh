#!/bin/bash
#SBATCH --account=naiss2026-4-398-cpu
#SBATCH --job-name=N3k_M
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=0-0

# regime: hopping=mu=delta=1, alpha=1 -- the model's own "natural" physics
# defaults. Same N_max=30000 / 27-checkpoint / num_repeats=250 config as the
# mean-estimator companion. predicted ~55-60h wall clock (mean-estimator
# cost plus median_of_means' checkpoint overhead -- this estimate carries
# more uncertainty than usual since it's extrapolated to a checkpoint
# density not measured before; worth watching the actual elapsed time on
# this run and trimming num_repeats on a resubmit if it runs hotter than
# expected).

# source ../.venv/bin/activate

python -u experiments/exp_13/exp_13.py \
    --n 10 \
    --hopping 1.0 \
    --mu 1.0 \
    --delta 1.0 \
    --alpha 1.0 \
    --q-values 0.0909090909 0.5 \
    --tune-q \
    --estimator median_of_means \
    --n-samples 10 15 20 30 50 70 100 150 200 300 500 700 1000 1500 2000 3000 5000 7000 10000 12500 15000 17500 20000 22500 25000 27500 30000 \
    --num-repeats 250 \
    --seed 0 \
    --version N3k_M
