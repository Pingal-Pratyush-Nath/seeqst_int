#!/usr/bin/env python3
"""
================================================================================
 exp_9 -- exp_7, trimmed to the core comparison + a selectable estimator
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_9 is exp_7's "exact-m X/Y-local" observable family (see
``tasks/exp_7_tuned_locality.py`` / ``experiments/exp_7/exp_7.py`` for the
full theoretical background -- Theorem 1 / Corollary 5 of
``notes/main_theorem/main_theorem.tex``), restricted to exactly what the
project is currently comparing:

    * ONLY the "exact" observable family (X/Y-weight exactly ``m``);
      exp_7's "at_least" family is dropped.
    * ONLY two ``--rest-mode`` choices for the non-XY qubits: "random" (iid
      Z/I per qubit) and "z" (all-Z tail, full weight n); exp_7's
      "identity" mode is dropped.
    * ONLY four ensembles --

        pauli                     PauliEnsemble()                  (beta = 3^weight(P))
        clifford                  CliffordEnsemble()                (beta = 2^n+1, exact)
        seeqst_uniform            SEEQSTEnsemble()                  (flat/uniform SEEQST, q=1/2)
        seeqst_binomial_tuned     SEEQSTBinomialEnsemble(q=m/n)     (Theorem 1's optimal q)

      exp_7's ``seeqst_binomial_untuned`` and ``seeqst_unifsize_tuned`` are
      both dropped -- they were apparatus specific to exp_7's
      floor-robustness ("at_least") study, which this experiment doesn't
      run.

NEW: SELECTABLE POINT ESTIMATOR (``--estimator``)
--------------------------------------------------------------------------------
exp_7 (and every experiment before it) always used a plain running mean.
exp_9 adds a second option:

    mean              running empirical mean over the first N_sample shots
                      (exp_7's estimator, unchanged).
    median_of_means   HKP's own estimator (Huang, Kueng & Preskill,
                      informal Thm. 1 / SI Thm. S1 -- see
                      ``notes/working/seeqst_sample_complexity.tex``): split
                      the first N_sample shots into K equal-size groups,
                      average within each group, then take the
                      per-observable MEDIAN of those K group-means. This is
                      the estimator the sample-complexity theorem is
                      actually stated for (median-of-means, not a plain
                      mean), and it makes the RMSE robust to the rare
                      huge-magnitude outliers a badly-tuned/heavy-tailed
                      ensemble can produce (see
                      ``results/exp_7/n6_m5_exact_random`` /
                      ``n6_m5_exact_zex`` for exactly that failure mode in
                      ``seeqst_binomial_untuned`` -- not run by exp_9, but
                      the reason this estimator option exists).

``K``, the number of groups, is FIXED for the whole run (``--mom-num-groups``,
same as HKP's own theorem: K depends only on M and delta, not on N_sample or
epsilon) and defaults to HKP's own formula ``K = 2*ceil(ln(2*M/delta))``
(``--mom-delta``, default 0.1) when not given explicitly. For any
``--n-samples`` checkpoint smaller than K, K is clamped down to that
checkpoint's own N_sample (groups of size 1 -- a plain per-shot median) so a
run never crashes on its own smallest checkpoint; see
``tasks/exp_9_tuned_locality_mom.py::_checkpoint_estimates``.

Both estimators are computed from the SAME underlying stream of shots per
(state, ensemble) -- the two-layer randomization design and per-ensemble
snapshot-stream reuse are otherwise identical to exp_7.

METRICS -- ONLY TWO (no mean_variance), same as exp_6/exp_7
--------------------------------------------------------------------------------
    RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
averaged over BOTH randomization layers (every outer state x every inner
observable-set draw, pooled together).

PLOTS: no 1/sqrt(N_sample) reference line (exp_7 drew one; exp_9 drops it
-- it's only a claim about the "mean" estimator's own convergence rate and
would be misleading plotted against a median_of_means run).

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --m                       exact X/Y-locality; must satisfy 1<=m<=n-1
                            (needed for the tuned q*=m/n to lie strictly in
                            (0,1))
  --rest-mode               "random" | "z" (default "random")
  --estimator               "mean" | "median_of_means" (default "mean")
  --mom-num-groups          K for median_of_means (default: HKP's own
                            K=2*ceil(ln(2M/delta)); ignored for --estimator=mean)
  --mom-delta               delta used ONLY to derive the default K above
                            when --mom-num-groups is not given (default 0.1)
  --num-observables         M, number of random observables per draw
  --n-samples               list of N_sample checkpoints to evaluate at
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ...)
  --quick                   small/fast configuration for smoke-testing
  --quiet                   suppress the per-state-draw progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_9/v<N>/:
  hyperparameters.json                  all settings used for this run
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png       )  one figure per state type,
  max_error_vs_nsample_<state>.png  )  one metric per figure, 4 ensemble lines

USAGE
--------------------------------------------------------------------------------
    python exp_9.py --quick
    python exp_9.py --n 6 --m 5 --rest-mode random
    python exp_9.py --n 6 --m 5 --rest-mode z --estimator median_of_means
    python exp_9.py --n 6 --m 5 --estimator median_of_means --mom-num-groups 5
    python exp_9.py --n 7 --m 3 --version my_run_name   # explicit output folder name
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# This script lives two levels below the shadow_benchmark root (same as
# every other exp_*.py), so we add that root to sys.path ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from tasks.exp_9_tuned_locality_mom import Exp9Config, run_exp9, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_9/results/), same convention as every other exp_*.py.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_9"

METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse"),
    ("max_error_mean", "max_error_std", "Max error", "max_error"),
]
ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
    "seeqst_binomial_tuned": ("D", "tab:red"),
}


def build_ensembles(n: int, m: int) -> dict:
    q_tuned = m / n
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
        "seeqst_binomial_tuned": SEEQSTBinomialEnsemble(q_tuned),
    }


def plot_state(summary, state_type: str, out_dir: Path) -> None:
    sub = summary[summary.state_type == state_type]
    if sub.empty:
        return
    for mean_col, std_col, ylabel, tag in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
            se = sub[sub.ensemble == ens_name].sort_values("n_sample")
            if se.empty:
                continue
            ax.errorbar(
                se.n_sample, se[mean_col], yerr=se[std_col],
                marker=marker, color=color, label=ens_name, capsize=2, lw=1.5,
            )
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
    parser.add_argument("--n", type=int, default=6, help="system size (number of qubits)")
    parser.add_argument(
        "--m", type=int, default=2,
        help="exact X/Y-locality; must satisfy 1<=m<=n-1",
    )
    parser.add_argument(
        "--rest-mode", type=str, default="random", choices=["random", "z"],
        help="how the n-m non-XY qubits are set ('random': iid Z/I per qubit, default. "
        "'z': all Z, i.e. full weight n)",
    )
    parser.add_argument(
        "--estimator", type=str, default="mean", choices=["mean", "median_of_means"],
        help="point-estimator used to turn the first N_sample shots into a prediction "
        "('mean': running empirical mean, exp_7's estimator. 'median_of_means': HKP's "
        "own estimator -- split into K groups, average within each, median across groups)",
    )
    parser.add_argument(
        "--mom-num-groups", type=int, default=None,
        help="K, number of groups for --estimator=median_of_means (default: HKP's own "
        "K=2*ceil(ln(2*num_observables/mom_delta)); ignored for --estimator=mean)",
    )
    parser.add_argument(
        "--mom-delta", type=float, default=0.1,
        help="delta used ONLY to derive the default --mom-num-groups above when it is "
        "not given explicitly (default 0.1)",
    )
    parser.add_argument("--num-observables", type=int, default=10, help="M, number of random observables")
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
        help="N_sample checkpoints to evaluate the estimator at",
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
        "--states", nargs="+", default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_9/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-state-draw progress lines from the outer randomization layer",
    )
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.m = 2
        args.num_observables = 5
        args.n_samples = [10, 50, 200, 800]
        args.num_state_repeats = 2
        args.num_observable_repeats = 2

    if not (1 <= args.m <= args.n - 1):
        parser.error(f"--m must satisfy 1 <= m <= n-1 (got m={args.m}, n={args.n})")

    n_samples_sorted = sorted(set(args.n_samples))

    if args.estimator == "median_of_means":
        if args.mom_num_groups is not None:
            mom_num_groups = args.mom_num_groups
            if mom_num_groups < 1:
                parser.error(f"--mom-num-groups must be >= 1 (got {mom_num_groups})")
        else:
            mom_num_groups = 2 * math.ceil(math.log(2 * args.num_observables / args.mom_delta))
        if n_samples_sorted[0] < mom_num_groups:
            print(
                f"Note: K={mom_num_groups} exceeds the smallest --n-samples checkpoint "
                f"({n_samples_sorted[0]}); that checkpoint (and any other below K) uses "
                f"K_eff=n_sample groups of size 1 instead -- see "
                f"tasks/exp_9_tuned_locality_mom.py::_checkpoint_estimates."
            )
    else:
        mom_num_groups = 1  # unused by the "mean" estimator; kept as a valid config value

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp9Config(
        n=args.n,
        m=args.m,
        rest_mode=args.rest_mode,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(args.n, args.m),
        estimator=args.estimator,
        mom_num_groups=mom_num_groups,
        state_types=args.states,
        seed=args.seed,
    )

    print(
        f"Running exp_9 -> {out_dir}\n"
        f"  n={args.n}, m={args.m}, rest_mode={args.rest_mode!r}, "
        f"q_tuned={args.m / args.n:.6g}\n"
        f"  estimator={args.estimator!r}"
        + (f", mom_num_groups={mom_num_groups}, mom_delta={args.mom_delta}" if args.estimator == "median_of_means" else "")
        + f"\n  M={args.num_observables}, n_samples={n_samples_sorted}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp9(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "m": args.m,
        "rest_mode": args.rest_mode,
        "q_tuned": args.m / args.n,
        "estimator": args.estimator,
        "mom_num_groups": mom_num_groups,
        "mom_delta": args.mom_delta,
        "num_observables": args.num_observables,
        "n_samples": n_samples_sorted,
        "num_state_repeats": args.num_state_repeats,
        "num_observable_repeats": args.num_observable_repeats,
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
    print(f"Wrote {len(METRICS) * len(args.states)} plots to {out_dir}")


if __name__ == "__main__":
    main()
