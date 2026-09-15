#!/usr/bin/env python3
"""
================================================================================
 exp_2 -- Derandomized SEEQST: error scaling under an observable-set-adapted,
          frequency-weighted measurement schedule
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_1 measures each state with a uniformly-random draw from the FULL SEEQST
ensemble, oblivious to which observables you'll eventually ask about -- that
obliviousness is the whole point of classical shadows (you don't need to know
your observables in advance). exp_2 asks the opposite question: if you DO
know your M observables in advance, how much better can you do by choosing
your measurement circuits specifically for them?

For a chosen system size n and a drawn set of M random Pauli observables
O_1, ..., O_M (same generating distribution as exp_1's inner layer):

  1. DERANDOMIZE each observable: ``derandomized_seeqst.circuit_for_pauli``
     gives the exact SEEQST circuit that measures that specific observable
     with certainty (see that module for the derivation). Distinct
     observables can map to the SAME circuit -- e.g. Z_1 and Z_1 Z_2 are
     both measured by "just measure everything in the computational basis"
     -- and a single shot with that circuit gives a deterministic +-1
     reading for EVERY observable that maps to it, simultaneously. This
     "measurement grouping" is where all of the efficiency gain over exp_1
     comes from.

  2. ALLOCATE the shot budget across the resulting DISTINCT circuits,
     proportional to observable demand: if C distinct circuits are needed
     and circuit S_j is the exact match for count_j of the M observables,
     each individual shot's circuit is drawn from the categorical
     distribution P(S_j) = count_j / M. (E.g. if S_1 is needed by 2 of 6
     observables and S_3 by 4 of 6, a given shot uses S_1 with probability
     2/6 and S_3 with probability 4/6.) This is a simple, greedy allocation
     -- shots go where the demand is -- not a variance-optimized one.

  3. ESTIMATE: every shot updates the running history of every observable
     that shares its circuit. At a checkpoint N_sample (meaning: after
     N_sample total shots drawn from the shared budget), observable O_i's
     estimate is the mean of however many of ITS OWN matching shots have
     occurred so far within those N_sample total shots -- a random, and for
     small N_sample often zero, count (see COVERAGE below).

TWO IMPORTANT DIFFERENCES FROM exp_1's PROTOCOL
--------------------------------------------------------------------------------
1. No shared snapshot stream across observable-set draws. exp_1 measures a
   state ONCE per outer-layer draw and reuses that same snapshot stream for
   every inner observable-set draw, because the (uniform) SEEQST sampling
   distribution doesn't depend on the observables. Here it does -- the
   whole point is to adapt the circuit-sampling policy to the observable
   set -- so a fresh batch of shots has to be measured from scratch for
   EVERY inner observable-set draw. There is no "measure once, mine many
   times" step in this protocol.

2. No beta rescaling. exp_1's single-shot estimator is
   ``beta(P) * <psi_pre|P|psi_pre>``, and it's unbiased only because P's
   measurement circuit is drawn from the SAME (uniform, full-ensemble)
   distribution that beta(P) = 1/alpha_P was computed under -- beta corrects
   for the fact that, on average, only a 1/beta fraction of draws "hit"
   (measure P exactly). Here, every shot used for O_i comes from O_i's own
   exactly-diagonalizing circuit BY CONSTRUCTION -- there is no "miss" left
   to correct for, so ``<psi_pre|P_i|psi_pre>`` (no beta factor!) is ALREADY
   an exactly unbiased single-shot estimate of Tr(P_i rho): this is ordinary
   basis-rotated projective measurement, not the shadow inverse-map trick.
   Multiplying by beta(P_i) here would silently inflate every estimate by a
   factor of 2 or 2^(n+1) -- verified NOT to do this (see
   tasks/exp_2_derandomized_scaling.py's ``run_exp2`` docstring and a
   throwaway numerical check confirming the no-beta estimator converges to
   the true expectation value, while the beta-scaled version does not).

COMPARING AGAINST RANDOM SEEQST (exp_1's PROTOCOL)
--------------------------------------------------------------------------------
By default (``--no-random-seeqst`` to turn it off), this script ALSO runs
exp_1's exact random-SEEQST protocol -- uniformly-random (subset, branch)
per shot, one shared snapshot stream per state reused across every
observable-set draw, beta(P)-rescaled hit-or-miss estimator -- on the SAME
drawn states and observable sets as the derandomized protocol above, in this
same run (rather than reading a separately-run exp_1's output file). This
guarantees a true apples-to-apples comparison -- identical states and
observables, not just the same n and seed -- and needs no prior exp_1 run to
exist. Every row in raw.csv/summary.csv is tagged with a ``method`` column
('derandomized_seeqst' or 'random_seeqst'), and both curves are drawn on
every plot.

METRICS
--------------------------------------------------------------------------------
For every n_sample checkpoint, across the M observables, for EACH method:

    RMSE(N_sample)          = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MeanVariance(N_sample)  = mean_i Var[o_hat_i^(1)], sample variance of
                              o_i's own matching-shot history so far
    MaxError(N_sample)      = max_i |o_hat_i(N_sample) - o_i|
    Coverage(N_sample)      = fraction of the M observables with >= 1
                              matching shot in the first N_sample total shots
    MeanMatchesPerObservable(N_sample) = mean_i (# of o_i's matching shots
                              within the first N_sample total shots)

For the derandomized_seeqst method, observables with ZERO matching shots so
far are EXCLUDED from RMSE / MeanVariance / MaxError for that checkpoint
(there's nothing to estimate with yet) -- Coverage tracks how many
observables that excludes, so you can tell when the other three metrics have
"warmed up" (coverage -> 1) versus when they're still based on a small,
possibly unrepresentative subset of the M observables. Expect Coverage to be
worst at small N_sample and for observables whose circuit is rare (low
count_j / M). For the random_seeqst method, every shot contributes (via
beta rescaling, not exclusion) to every observable's running mean, so
Coverage is fixed at 1.0 and MeanMatchesPerObservable = N_sample always.

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --num-observables         M, number of random Pauli observables per draw
  --n-samples               list of N_sample (total shot budget) checkpoints
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state (each gets its own fresh shots)
  --k                       optional: restrict observables to EXACTLY weight k
                            (default: fully random weight, any k from 1 to n)
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ... -- see OUTPUT below)
  --no-random-seeqst        skip the random-SEEQST comparison method (faster,
                            but plots then only show the derandomized curve)
  --quiet                   suppress the per-state / per-observable-set
                            progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_2/v<N>/ (same
auto-incrementing scheme as exp_1, see common/versioning.py):
  hyperparameters.json                   all settings used for this run
  raw.csv        one row per (method, state_repeat, obs_repeat, state_type, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png             )
  mean_variance_vs_nsample_<state>.png    )  one figure per state type,
  max_error_vs_nsample_<state>.png        )  one metric per figure, one line
  coverage_vs_nsample_<state>.png         )  per method (derandomized/random)
i.e. 3 states x 4 metrics = 12 plots per run. RMSE and MaxError get a
1/sqrt(N_sample) reference line (anchored to the derandomized curve);
MeanVariance gets a flat reference line at the derandomized curve's
most-converged value; Coverage gets a flat reference line at 1.0.

USAGE
--------------------------------------------------------------------------------
    python exp_2.py --quick
    python exp_2.py --n 5 --num-observables 10 \\
        --num-state-repeats 3 --num-observable-repeats 3 \\
        --n-samples 10 30 100 300 1000 3000 10000
    python exp_2.py --n 6 --k 2 --num-observables 20 --states haar_random ghz
    python exp_2.py --n 5 --version my_run_name   # explicit output folder name
    python exp_2.py --n 5 --no-random-seeqst       # skip the comparison method
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives at shadow_benchmark/experiments/exp_2/exp_2.py -- two
# levels below the shadow_benchmark root (common/, ensembles/, tasks/,
# results/ all live there). Python only auto-adds the SCRIPT's own directory
# to sys.path, so without this the "from common..." etc. imports below, and
# "python exp_2.py" run directly from this folder, would fail with
# ModuleNotFoundError. parents[2] = .../experiments/exp_2/exp_2.py ->
# parents[0]=exp_2/, parents[1]=experiments/, parents[2]=shadow_benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from tasks.exp_2_derandomized_scaling import Exp2Config, run_exp2, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_2/results/) so this script keeps writing to the same
# place, and the same auto-incrementing v<N> history, regardless of which
# subfolder the script itself has been moved into.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_2"

# (mean_col, std_col, ylabel, filename_tag, reference_style)
METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("mean_variance_mean", "mean_variance_std", "Mean variance  Var[o_hat]", "mean_variance", "flat"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
    ("coverage_mean", None, "Coverage (fraction of observables with >= 1 matching shot)", "coverage", "one"),
]
# method -> (marker, color, linestyle, legend label)
METHOD_STYLE = {
    "derandomized_seeqst": ("^", "tab:green", "-", "derandomized seeqst"),
    "random_seeqst": ("x", "tab:gray", "--", "random seeqst (exp_1 protocol)"),
}


def plot_state(summary, state_type: str, out_dir: Path) -> None:
    sub = summary[summary.state_type == state_type]
    if sub.empty:
        return
    for mean_col, std_col, ylabel, tag, ref_style in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ref_anchor = None
        for method, (marker, color, linestyle, label) in METHOD_STYLE.items():
            se = sub[sub.method == method].sort_values("n_sample")
            if se.empty:
                continue
            yerr = se[std_col] if std_col else None
            ax.errorbar(
                se.n_sample, se[mean_col], yerr=yerr,
                marker=marker, color=color, linestyle=linestyle, label=label, capsize=2, lw=1.5,
            )
            # Reference lines are anchored to the derandomized curve specifically
            # (the protocol this script is about) even though both are plotted.
            if method == "derandomized_seeqst":
                if ref_style == "inv_sqrt_n":
                    ref_anchor = (se.n_sample.iloc[0], se[mean_col].iloc[0])
                elif ref_style == "flat":
                    ref_anchor = se[mean_col].iloc[-1]  # most-converged (largest N_sample) value

        xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
        if ref_style == "inv_sqrt_n" and ref_anchor is not None:
            x0, y0 = ref_anchor
            ref = y0 * np.sqrt(x0 / xs)
            ax.plot(xs, ref, "k--", alpha=0.4, lw=1, label=r"$\propto 1/\sqrt{N_{sample}}$ (reference)")
        elif ref_style == "flat" and ref_anchor is not None:
            ax.plot(xs, np.full_like(xs, ref_anchor), "k--", alpha=0.4, lw=1,
                     label="converged value (reference)")
        elif ref_style == "one":
            ax.plot(xs, np.ones_like(xs), "k--", alpha=0.4, lw=1, label="full coverage (reference)")
        ax.set_xscale("log")
        if ref_style != "one":
            ax.set_yscale("log")
        ax.set_xlabel("total shots drawn  $N_{sample}$")
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
        help="N_sample checkpoints (total shots drawn) to evaluate the running-mean estimator at",
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
        help="output subfolder name under results/exp_2/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument(
        "--no-random-seeqst", action="store_true",
        help="skip the random-SEEQST comparison method (faster; plots then show only the derandomized curve)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-state / per-observable-set progress lines",
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

    config = Exp2Config(
        n=args.n,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        state_types=args.states,
        k=args.k,
        seed=args.seed,
        include_random_seeqst=not args.no_random_seeqst,
    )

    print(
        f"Running exp_2 -> {out_dir}\n"
        f"  n={args.n}, M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  k={'random' if args.k is None else args.k}, states={args.states}, seed={args.seed}\n"
        f"  include_random_seeqst={config.include_random_seeqst}"
    )
    t0 = time.time()
    raw = run_exp2(config, verbose=not args.quiet)
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
        "include_random_seeqst": config.include_random_seeqst,
        "elapsed_seconds": round(elapsed, 1),
        "num_raw_rows": len(raw),
    }
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print(f"Wrote raw.csv, summary.csv, hyperparameters.json to {out_dir}")

    for state_type in args.states:
        plot_state(summary, state_type, out_dir)
    print(f"Wrote {4 * len(args.states)} plots to {out_dir}")


if __name__ == "__main__":
    main()
