#!/usr/bin/env python3
"""
================================================================================
 exp_4 -- Derandomized SEEQST vs. Huang-Kueng-Preskill's derandomized Pauli
          shadows: two different derandomization strategies, head to head
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_2/exp_3 derandomize SEEQST: given M known observables, pick the exact
SEEQST (entangling GHZ-block) circuit that measures each one deterministically,
group observables that share a circuit, and allocate a shot budget across the
resulting distinct circuits proportional to demand.

Huang, Kueng & Preskill's own follow-up paper (arXiv:2103.07510,
``papers/derandomized_shadows.pdf``) derandomizes their ORIGINAL (local
random-Pauli / product-measurement) protocol instead: given the same kind of
M known observables, a deterministic greedy algorithm outputs a FIXED
schedule of single-qubit measurement bases (no entangling gates at all) that
tries to measure every observable exactly the target number of times, using
as few total measurement rounds as possible.

exp_4 runs BOTH derandomization strategies on the SAME drawn states and
observable sets and compares them on identical footing -- same metrics,
same n_sample checkpoints, same "state, then M random Pauli observables"
draws -- to see how much (if anything) SEEQST's larger, entangling
measurement ensemble buys you over a fully local, product-measurement
derandomization scheme.

THE TWO METHODS
--------------------------------------------------------------------------------
  derandomized_seeqst  (exp_2/exp_3's protocol, reused here as-is):
    1. Derandomize each observable via ``experiments.Archived.exp_3.exp_3``'s
       ``select_subset_and_branch`` -- the exact SEEQST (subset, branch)
       circuit that measures it with certainty. (exp_3 lives under
       ``Archived/`` for folder organization only -- it's a live dependency,
       not deprecated.)
    2. Group observables sharing a circuit; draw shots i.i.d. from a
       categorical distribution over the resulting distinct circuits,
       proportional to how many observables each one serves.
    3. Every shot updates every observable sharing its circuit.

  derandomized_pauli  (Huang-Kueng-Preskill's protocol, via
  ``codes/predicting-quantum-properties/data_acquisition_shadow.derandomized_classical_shadow``,
  called unmodified):
    1. A single call computes the ENTIRE measurement schedule up front: a
       deterministic greedy procedure (their Algorithm 1) that, round by
       round, picks the single-qubit Pauli basis for every qubit that best
       serves the observables not yet sufficiently measured, using a
       pessimistic-estimator cost function derived from a Chernoff-style
       tail bound. No entangling gates -- each round is a plain product
       measurement.
    2. Every round is simulated once; an observable "matches" a round iff
       every qubit it acts on was measured in exactly the basis it needs
       (unlike SEEQST's grouping, where a whole GROUP of observables always
       hits together -- here, in general, a different, changing subset of
       observables matches each round).
    3. The schedule itself has NO per-shot randomness at all (given the
       same observables/target/n, ``derandomized_classical_shadow`` always
       returns the identical schedule) -- the only randomness left is which
       state/observables were drawn, and the physical (Born-rule)
       measurement outcomes when that fixed schedule is actually executed.

Both methods use a hit-conditional (self-normalized) estimator: an
observable's estimate at checkpoint N_sample is the mean of however many of
ITS OWN matching shots occurred within the first N_sample total rounds --
no rescaling factor needed for either method, since a "hit" already gives an
exact, unbiased single-shot value (see ``experiments/exp_2/exp_2.md``'s "no
beta rescaling" section -- the same argument applies to derandomized_pauli).
Observables with zero matches so far are excluded from RMSE/MeanVariance/
MaxError at that checkpoint; ``coverage`` tracks how many that excludes.

ONE LONG SCHEDULE, MANY CHECKPOINTS -- AND ITS CAVEAT
--------------------------------------------------------------------------------
Both methods are run ONCE per (state, observable-set) trial with a shot
budget of max(n_samples), then evaluated at every smaller checkpoint by
looking only at a PREFIX of that one run -- the same trick exp_1/exp_2 use.
For derandomized_seeqst this is exact (shots are i.i.d., so a prefix of a
long run is statistically identical to a fresh short run). For
derandomized_pauli it is only an APPROXIMATION: Huang's greedy algorithm's
round-by-round choices depend on the target ``num_of_measurements_per_observable``
it's told to aim for (later rounds get more "urgent" as that target
approaches), so a schedule calibrated for max(n_samples) is not bit-for-bit
identical to one calibrated exactly for a smaller checkpoint. Recomputing a
fresh schedule at every checkpoint would be more faithful but requires
re-running the (nontrivial-cost) derandomization call once per checkpoint
instead of once per trial -- not done here. Keep this in mind for the
smallest checkpoints especially; see PERFORMANCE below for why this
trade-off was made.

PERFORMANCE
--------------------------------------------------------------------------------
``derandomized_classical_shadow`` is unmodified, pure-Python, and its cost
scales roughly as O(rounds x n x M) (rounds = the schedule length it settles
on, which is >= max(n_samples) and grows with how incompatible the M
observables are with each other). This is comparable in order to exp_2's own
per-trial simulation cost, but with a larger constant -- expect
derandomized_pauli to be the slower of the two methods per trial, and budget
--n-samples / --num-observables accordingly (start with --quick).

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --num-observables         M, number of random Pauli observables per draw
  --n-samples               list of N_sample (total round budget) checkpoints
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state (each gets fresh measurements,
                            for both methods)
  --k                       optional: restrict observables to EXACTLY weight k
                            (default: fully random weight, any k from 1 to n)
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ... -- see OUTPUT below)
  --no-seeqst               skip the derandomized_seeqst arm
  --no-derandomized-pauli   skip the derandomized_pauli arm (usually the
                            slower of the two -- skip it for a quick SEEQST-only look)
  --quiet                   suppress the per-state / per-observable-set
                            progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_4/v<N>/ (same
auto-incrementing scheme as exp_1/exp_2, see common/versioning.py):
  hyperparameters.json                   all settings used for this run
  raw.csv        one row per (method, state_repeat, obs_repeat, state_type, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png             )
  mean_variance_vs_nsample_<state>.png    )  one figure per state type,
  max_error_vs_nsample_<state>.png        )  one metric per figure, one line
  coverage_vs_nsample_<state>.png         )  per method (seeqst/pauli)
i.e. 3 states x 4 metrics = 12 plots per run (fewer if a method is skipped
via --no-seeqst / --no-derandomized-pauli). RMSE and MaxError get a
1/sqrt(N_sample) reference line (anchored to derandomized_seeqst);
MeanVariance gets a flat reference line at derandomized_seeqst's
most-converged value; Coverage gets a flat reference line at 1.0.

USAGE
--------------------------------------------------------------------------------
    python experiments/exp_4/exp_4.py --quick
    python experiments/exp_4/exp_4.py --n 5 --num-observables 8 \\
        --num-state-repeats 3 --num-observable-repeats 3 \\
        --n-samples 10 30 100 300 1000
    python experiments/exp_4/exp_4.py --n 6 --k 2 --num-observables 15 --states haar_random ghz
    python experiments/exp_4/exp_4.py --n 5 --version my_run_name   # explicit output folder name
    python experiments/exp_4/exp_4.py --n 5 --no-derandomized-pauli  # SEEQST arm only, fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives at shadow_benchmark/experiments/exp_4/exp_4.py -- two
# levels below the shadow_benchmark root (common/, ensembles/, tasks/,
# results/ all live there). Python only auto-adds the SCRIPT's own directory
# to sys.path, so without this the "from common..." etc. imports below, and
# "python exp_4.py" run directly from this folder, would fail with
# ModuleNotFoundError. parents[2] = .../experiments/exp_4/exp_4.py ->
# parents[0]=exp_4/, parents[1]=experiments/, parents[2]=shadow_benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from tasks.exp_4_derandomized_comparison import Exp4Config, run_exp4, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_4/results/) so this script keeps writing to the same
# place, and the same auto-incrementing v<N> history, regardless of which
# subfolder the script itself has been moved into.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_4"

# (mean_col, std_col, ylabel, filename_tag, reference_style)
METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("mean_variance_mean", "mean_variance_std", "Mean variance  Var[o_hat]", "mean_variance", "flat"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
    ("coverage_mean", None, "Coverage (fraction of observables with >= 1 matching shot)", "coverage", "one"),
]
# method -> (marker, color, linestyle, legend label)
METHOD_STYLE = {
    "derandomized_seeqst": ("^", "tab:green", "-", "derandomized SEEQST (exp_2/exp_3)"),
    "derandomized_pauli": ("D", "tab:red", "-", "derandomized Pauli (Huang-Kueng-Preskill)"),
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
            # Reference lines are anchored to derandomized_seeqst specifically,
            # so the two methods' plots stay comparable across runs that skip one arm.
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
        ax.set_xlabel("total measurement rounds  $N_{sample}$")
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
        help="N_sample checkpoints (total measurement rounds) to evaluate the running-mean estimator at",
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
        help="output subfolder name under results/exp_4/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument(
        "--no-seeqst", action="store_true",
        help="skip the derandomized_seeqst arm",
    )
    parser.add_argument(
        "--no-derandomized-pauli", action="store_true",
        help="skip the derandomized_pauli (Huang-Kueng-Preskill) arm -- usually the slower of the two",
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

    if args.no_seeqst and args.no_derandomized_pauli:
        parser.error("--no-seeqst and --no-derandomized-pauli together would run nothing")

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp4Config(
        n=args.n,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        state_types=args.states,
        k=args.k,
        seed=args.seed,
        include_seeqst=not args.no_seeqst,
        include_derandomized_pauli=not args.no_derandomized_pauli,
    )

    print(
        f"Running exp_4 -> {out_dir}\n"
        f"  n={args.n}, M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  k={'random' if args.k is None else args.k}, states={args.states}, seed={args.seed}\n"
        f"  include_seeqst={config.include_seeqst}, include_derandomized_pauli={config.include_derandomized_pauli}"
    )
    t0 = time.time()
    raw = run_exp4(config, verbose=not args.quiet)
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
        "include_seeqst": config.include_seeqst,
        "include_derandomized_pauli": config.include_derandomized_pauli,
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
