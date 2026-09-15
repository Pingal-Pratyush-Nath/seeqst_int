#!/usr/bin/env python3
"""
================================================================================
 exp_6 -- Bounded-X/Y-weight observables: Pauli vs. Clifford vs. 3 SEEQST variants
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_1 compares Pauli / Clifford / SEEQST on fully-random Pauli observables.
This experiment instead compares FIVE ensembles --

    pauli               PauliEnsemble()                    (beta = 3^k)
    clifford            CliffordEnsemble()                 (beta = 2^n+1, exact)
    seeqst_uniform      SEEQSTEnsemble()                    (flat/uniform SEEQST)
    seeqst_binomial     SEEQSTBinomialEnsemble(q=1/(n+1))   (per-qubit Bernoulli(q))
    seeqst_unifsize_l   SEEQSTUniformSizeEnsemble(l)        (capped uniform subset size)

-- on an observable family built to isolate exactly the tunable knob the
theory note (``QIP_notes/SEEQST_shadows_threshold.tex``) is about: the
number of X/Y (non-Z) factors in the Pauli string. Every observable here has
FULL weight n (every qubit is X, Y, or Z -- never identity), with the COUNT
of X/Y factors drawn uniformly from {1, ..., l} and every other qubit fixed
to Z (``pauli_utils.random_bounded_xy_spec``). ``l`` is a hyperparameter
(--l) capping how "Pauli-heavy" (vs. "Z-heavy") the observables are allowed
to be, and is also the parameter of the ``seeqst_unifsize_l`` ensemble
itself -- both the observable generator AND one of the two new ensembles
being benchmarked are governed by the same ``l``, by design (see exp_6.md
for why).

The two-layer randomization design, running-mean estimator, and per-ensemble
snapshot-stream reuse are otherwise IDENTICAL to exp_1 -- see that script's
module docstring for the full rationale, reproduced only in brief here:

  OUTER layer -- random state: draw a fresh random state (of each requested
  type) for the system size n, then immediately draw ONE stream of
  max(n_samples) classical-shadow snapshots per ensemble for it.

  INNER layer -- random observable set, for that SAME state and SAME
  snapshot stream: draw a fresh set of M random bounded-XY-weight
  observables, form the running-mean estimator at every checkpoint
  N_sample in --n-samples, and record error metrics. Repeated
  --num-observable-repeats times per outer state.

METRICS -- ONLY TWO (no mean_variance)
--------------------------------------------------------------------------------
Unlike exp_1, this experiment reports only:

    RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|

exp_1's MeanVariance metric (the empirical single-shot shadow-norm-squared)
is dropped entirely here -- not just omitted from the plots, but never
computed in ``tasks/exp_6_bounded_xy_weight.py::run_exp6`` in the first
place, since it isn't needed and computing it is wasted work otherwise. See
exp_0 / exp_1 if you want that quantity's closed-form / empirical values.

All metrics are averaged over BOTH randomization layers (every outer state x
every inner observable-set draw, pooled together).

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --l                       observable/ensemble hyperparameter: max number of
                            X/Y factors in each observable (drawn uniformly
                            from {1,...,l}); also the seeqst_unifsize_l
                            ensemble's own parameter. Must satisfy 1<=l<=n.
                            Unlike exp_1's optional --k, this is always a
                            concrete integer (default 2) -- it is the swept
                            hyperparameter this experiment exists to study,
                            not an optional restriction.
  --q-binomial              Bernoulli-per-qubit inclusion probability for
                            seeqst_binomial (default: 1/(n+1), computed after
                            --n is known)
  --num-observables         M, number of random bounded-XY-weight observables
                            per draw
  --n-samples               list of N_sample checkpoints to evaluate at
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ... -- see OUTPUT below)
  --quiet                   suppress the per-state-draw progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_6/v<N>/
(v<N> auto-increments by default, or set it explicitly with --version):
  hyperparameters.json                  all settings used for this run (incl.
                                         the actual q_binomial value used)
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png       )  one figure per state type (3 states),
  max_error_vs_nsample_<state>.png  )  one metric per figure, 5 ensemble lines
i.e. 3 states x 2 metrics = 6 plots per run, each on a log-log scale, with a
1/sqrt(N_sample) reference line.

To see how the metrics depend on ``l`` itself (holding n and n_sample
fixed), run this script several times at different --l and pass the
resulting results/exp_6/v<N>/ directories to
``experiments/exp_6/plot_l_dependence.py``.

USAGE
--------------------------------------------------------------------------------
    python exp_6.py --quick
    python exp_6.py --n 6 --l 2 --num-observables 10 \\
        --num-state-repeats 3 --num-observable-repeats 3 \\
        --n-samples 10 30 100 300 1000 3000 10000
    python exp_6.py --n 6 --l 4 --q-binomial 0.3 --states haar_random ghz
    python exp_6.py --n 5 --l 3 --version my_run_name   # explicit output folder name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# See exp_1.py's identical comment: this script lives two levels below the
# shadow_benchmark root, so we add that root to sys.path ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from ensembles.seeqst_unifsize_ensemble import SEEQSTUniformSizeEnsemble
from tasks.exp_6_bounded_xy_weight import Exp6Config, run_exp6, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_6/results/), same convention as every other exp_*.py.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_6"

# (mean_col, std_col, ylabel, filename_tag, reference_style) -- both metrics
# are expected to decay like 1/sqrt(N_sample), same as exp_1's rmse/max_error.
METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
]
ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
    "seeqst_binomial": ("D", "tab:red"),
    "seeqst_unifsize_l": ("v", "tab:purple"),
}


def build_ensembles(n: int, l: int, q_binomial: float) -> dict:
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
        "seeqst_binomial": SEEQSTBinomialEnsemble(q_binomial),
        "seeqst_unifsize_l": SEEQSTUniformSizeEnsemble(l),
    }


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
        if ref_style == "inv_sqrt_n" and ref_anchor is not None:
            x0, y0 = ref_anchor
            xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
            ref = y0 * np.sqrt(x0 / xs)
            ax.plot(xs, ref, "k--", alpha=0.4, lw=1, label=r"$\propto 1/\sqrt{N_{sample}}$ (reference)")
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
    parser.add_argument(
        "--l", type=int, default=2,
        help="max number of X/Y factors per observable (also the seeqst_unifsize_l "
        "ensemble's own parameter); must satisfy 1<=l<=n",
    )
    parser.add_argument(
        "--q-binomial", type=float, default=None,
        help="per-qubit Bernoulli inclusion probability for seeqst_binomial "
        "(default: 1/(n+1))",
    )
    parser.add_argument("--num-observables", type=int, default=10, help="M, number of random observables")
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
        "--states", nargs="+", default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_6/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-state-draw progress lines from the outer randomization layer",
    )
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.l = 2
        args.num_observables = 5
        args.n_samples = [10, 50, 200, 800]
        args.num_state_repeats = 2
        args.num_observable_repeats = 2

    if not (1 <= args.l <= args.n):
        parser.error(f"--l must satisfy 1 <= l <= n (got l={args.l}, n={args.n})")

    q_binomial = args.q_binomial if args.q_binomial is not None else 1.0 / (args.n + 1)

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp6Config(
        n=args.n,
        l=args.l,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(args.n, args.l, q_binomial),
        state_types=args.states,
        seed=args.seed,
    )

    print(
        f"Running exp_6 -> {out_dir}\n"
        f"  n={args.n}, l={args.l}, q_binomial={q_binomial:.6g}, "
        f"M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp6(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "l": args.l,
        "q_binomial": q_binomial,
        "num_observables": args.num_observables,
        "n_samples": sorted(set(args.n_samples)),
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
