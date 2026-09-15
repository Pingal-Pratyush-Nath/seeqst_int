#!/usr/bin/env python3
"""
================================================================================
 exp_1 -- Error scaling of classical-shadow estimators vs. number of shots
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
For a chosen system size n, we compare the three measurement ensembles
(Pauli / Clifford / SEEQST) on all three test-state families (Haar-random
pure states, GHZ states, random stabilizer states) by looking at how the
error of the classical-shadow MEAN estimator shrinks as we increase the
number of shadow snapshots N_sample used to form it.

Two layers of randomization (this is the key design point of this script):

  OUTER layer -- random state: draw a fresh random state (of each requested
  type) for the system size n.

  INNER layer -- random observable set, for that SAME state: draw a fresh
  set of M random Pauli observables O_1, ..., O_M (each qubit independently
  I/X/Y/Z, conditioned on non-identity -- i.e. random weight, unless --k
  fixes the locality), and evaluate all three error metrics for that
  (state, observable-set) pair. This inner draw is repeated
  --num-observable-repeats times for the SAME outer state before moving on
  to the next outer state.

IMPORTANT -- snapshots are measured ONCE per state, not once per observable
set: right after each outer-layer state is drawn, we immediately draw ONE
stream of max(n_samples) classical-shadow snapshots per ensemble for it
(``ShadowEnsemble.sample_snapshots``), BEFORE looking at any observables at
all. That SAME stream of snapshots is then reused for every inner
observable-set draw (``ShadowEnsemble.evaluate_snapshots``) -- we never
re-measure the state just because we're about to look at a different set of
observables. This mirrors the actual physical protocol (and the whole point
of classical shadows): you measure a state once, then that data can be
mined for as many different observables as you like, with no new
experiments required. An earlier version of this script incorrectly drew a
fresh batch of snapshots for every inner observable-set draw, which
conflated "new measurement noise" with "new observable choice" and did not
reflect this reuse property.

For a given (state, ensemble), we draw one stream of max(n_samples)
independent single-shot shadow estimates o_hat_i^(1), ..., o_hat_i^(N_max)
per observable O_i (i.e. per inner observable-set draw, evaluated against
the shared snapshot stream above), and at every checkpoint N_sample in
--n-samples we form the running-mean estimator
o_hat_i(N_sample) = mean of the first N_sample shots, and report, across the
M observables:

    RMSE(N_sample)          = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MeanVariance(N_sample)  = mean_i Var[o_hat_i^(1)], estimated as the
                              sample variance of the first N_sample RAW
                              (pre-averaging) single-shot values
    MaxError(N_sample)      = max_i |o_hat_i(N_sample) - o_i|

where o_i = tr(O_i rho) is computed exactly from the statevector.
MeanVariance is a direct empirical estimate of the quantity bounded by
Lemma S1 of Huang, Kueng & Preskill (arXiv:2002.08953): the single-shot
shadow-norm-squared, ||O||^2_shadow >= Var[o_hat]. It is a fixed number
(depends on the observable/ensemble, not on N_sample), so it should flatten
out as N_sample grows rather than keep shrinking -- unlike RMSE and
MaxError, which decay because averaging N_sample i.i.d. shots divides
variance by N_sample: Var[o_hat(N_sample)] = Var[o_hat(1)]/N_sample, i.e.
RMSE(N_sample) ~ sqrt(MeanVariance / N_sample). See
``experiments/exp_0/exp_0.py`` for the closed-form values of this
same shadow-norm quantity (3^k for Pauli, 3*2^n for Clifford).

All three metrics are finally averaged over BOTH randomization layers
(every outer state x every inner observable-set draw, pooled together).

WHY A RUNNING (CUMULATIVE) MEAN INSTEAD OF REDRAWING AT EVERY N_sample
--------------------------------------------------------------------------------
We draw ONE stream of N_max shots per (state, ensemble) and take the running
mean at every checkpoint, rather than drawing a fresh batch of exactly
N_sample shots for every checkpoint. A prefix of length N_sample of an
i.i.d. sequence is itself a valid i.i.d. sample of that size, so this is
statistically equivalent to the "naive" approach but far cheaper (one pass
of N_max shots instead of one pass per checkpoint) -- and it's the same
N_max-shot stream that gets reused across every inner observable-set draw,
as described above.

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --num-observables         M, number of random Pauli observables per draw
  --n-samples               list of N_sample checkpoints to evaluate at
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state
  --k                       optional: restrict observables to EXACTLY weight k
                            (default: fully random weight, any k from 1 to n)
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ... -- see OUTPUT below)
  --quiet                   suppress the per-state-draw progress lines (see
                            PROGRESS OUTPUT below)

PROGRESS OUTPUT (the outer/state randomization layer)
--------------------------------------------------------------------------------
By default, every time the OUTER layer draws a new random state, a line is
printed to the terminal, e.g.:

    [outer layer | state draw 3/9] state_repeat=0 state_type='ghz' n=5 -> drawing 3 random observable set(s) for it

so you can watch that layer of randomization progress live as the run
executes (there are num_state_repeats x len(states) such draws in total).
Pass --quiet to suppress these lines.

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_1/v<N>/ so
that different configurations don't overwrite each other's data
(v<N> auto-increments by default -- the next unused v<N> under results/exp_1/
-- or set it explicitly with --version):
  hyperparameters.json                   all settings used for this run
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png           )
  mean_variance_vs_nsample_<state>.png  )  one figure per state type (3 states),
  max_error_vs_nsample_<state>.png      )  one metric per figure, 3 ensemble lines
i.e. 3 states x 3 metrics = 9 plots per run, each on a log-log scale. RMSE
and MaxError get a 1/sqrt(N_sample) reference line; MeanVariance gets a
flat reference line at its most-converged (largest-N_sample) value.

USAGE
--------------------------------------------------------------------------------
    python exp_1.py --quick
    python exp_1.py --n 5 --num-observables 10 \\
        --num-state-repeats 3 --num-observable-repeats 3 \\
        --n-samples 10 30 100 300 1000 3000 10000
    python exp_1.py --n 6 --k 2 --num-observables 20 --states haar_random ghz
    python exp_1.py --n 5 --version my_run_name   # explicit output folder name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives at shadow_benchmark/experiments/exp_1/exp_1.py -- two
# levels below the shadow_benchmark root (common/, ensembles/, tasks/,
# results/ all live there). Python only auto-adds the SCRIPT's own directory
# to sys.path, so without this the "from common..." etc. imports below, and
# "python exp_1.py" run directly from this folder, would fail with
# ModuleNotFoundError. parents[2] = .../experiments/exp_1/exp_1.py ->
# parents[0]=exp_1/, parents[1]=experiments/, parents[2]=shadow_benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from tasks.exp_1_error_scaling import Exp1Config, run_exp1, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_1/results/) so this script keeps writing to the same
# place, and the same auto-incrementing v<N> history, regardless of which
# subfolder the script itself has been moved into.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_1"

# (mean_col, std_col, ylabel, filename_tag, reference_style)
# "inv_sqrt_n" -> dashed 1/sqrt(N_sample) guide anchored at the first data
#   point (RMSE / MaxError are expected to decay at this rate).
# "flat"       -> dashed horizontal guide at the most-converged (largest
#   N_sample) value (MeanVariance is expected to flatten out to this).
METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("mean_variance_mean", "mean_variance_std", "Mean variance  Var[o_hat]", "mean_variance", "flat"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
]
ENSEMBLE_STYLE = {"pauli": ("o", "tab:blue"), "clifford": ("s", "tab:orange"), "seeqst": ("^", "tab:green")}


def build_ensembles() -> dict:
    return {"pauli": PauliEnsemble(), "clifford": CliffordEnsemble(), "seeqst": SEEQSTEnsemble()}


def plot_state(summary, state_type: str, out_dir: Path) -> None:
    sub = summary[summary.state_type == state_type]
    if sub.empty:
        return
    for mean_col, std_col, ylabel, tag, ref_style in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ref_anchor = None
        for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
            se = sub[sub.ensemble == ens_name].sort_values("n_sample")
            if se.empty:
                continue
            ax.errorbar(
                se.n_sample, se[mean_col], yerr=se[std_col],
                marker=marker, color=color, label=ens_name, capsize=2, lw=1.5,
            )
            if ref_style == "inv_sqrt_n" and ref_anchor is None:
                ref_anchor = (se.n_sample.iloc[0], se[mean_col].iloc[0])
            elif ref_style == "flat" and ref_anchor is None:
                ref_anchor = se[mean_col].iloc[-1]  # most-converged (largest N_sample) value
        if ref_style == "inv_sqrt_n" and ref_anchor is not None:
            x0, y0 = ref_anchor
            xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
            ref = y0 * np.sqrt(x0 / xs)
            ax.plot(xs, ref, "k--", alpha=0.4, lw=1, label=r"$\propto 1/\sqrt{N_{sample}}$ (reference)")
        elif ref_style == "flat" and ref_anchor is not None:
            xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
            ax.plot(xs, np.full_like(xs, ref_anchor), "k--", alpha=0.4, lw=1,
                     label="converged value (reference)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. $N_{{sample}}$  --  state = {state_type}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_vs_nsample_{state_type}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="system size (number of qubits)")
    parser.add_argument("--num-observables", type=int, default=10, help="M, number of random Pauli observables")
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
        help="N_sample checkpoints to evaluate the running-mean estimator at",
    )
    parser.add_argument(
        "--num-state-repeats", type=int, default=3,
        help="outer layer: how many random states to draw",
    )
    parser.add_argument(
        "--num-observable-repeats", type=int, default=3,
        help="inner layer: how many random observable sets to draw per state",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="restrict observables to exactly weight k (default: fully random weight)",
    )
    parser.add_argument(
        "--states", nargs="+", default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_1/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-state-draw progress lines from the outer randomization layer",
    )
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.num_observables = 5
        args.n_samples = [10, 50, 200, 800]
        args.num_state_repeats = 2
        args.num_observable_repeats = 2

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp1Config(
        n=args.n,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(),
        state_types=args.states,
        k=args.k,
        seed=args.seed,
    )

    print(
        f"Running exp_1 -> {out_dir}\n"
        f"  n={args.n}, M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  k={'random' if args.k is None else args.k}, states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp1(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "num_observables": args.num_observables,
        "n_samples": sorted(set(args.n_samples)),
        "num_state_repeats": args.num_state_repeats,
        "num_observable_repeats": args.num_observable_repeats,
        "k": args.k,
        "states": args.states,
        "seed": args.seed,
        "version": out_dir.name,
        "elapsed_seconds": round(elapsed, 1),
        "num_raw_rows": len(raw),
    }
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print(f"Wrote raw.csv, summary.csv, hyperparameters.json to {out_dir}")

    for state_type in args.states:
        plot_state(summary, state_type, out_dir)
    print(f"Wrote {3 * len(args.states)} plots to {out_dir}")


if __name__ == "__main__":
    main()
