#!/usr/bin/env python3
"""
================================================================================
 exp_11 -- exp_9's hyperparameters, estimating COMBINED S_m-sparse operators
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_11 uses exp_9's exact hyperparameter set (same four-ensemble roster,
same q=m/n tuning rule, same selectable point estimator, same RMSE/MaxError
metrics, same two-layer randomization design) but replaces exp_9's
observable with a genuinely different object: instead of
``num_observables`` separate single-Pauli-string observables, exp_11 draws
``num_observables`` independent random COMBINED ``S_m``-sparse operators --

    A = sum_{i=1}^{num_terms} c_i P_i,   each P_i with X/Y-weight in {min_xy,...,m}

via ``pauli_utils.random_sparse_operator`` (added specifically for this).
See that function's docstring for why any such superposition is guaranteed
``S_m``-sparse (Definition 5.3 of ``SEEQST_shadows4.pdf`` /
``notes/working/seeqst_sparse_tuning.tex``), and
``tasks/exp_11_sparse_operator_mom.py``'s module docstring for exactly how
a combined operator's shadow estimate is built from per-term shadow
estimates (short version: combine the per-shot estimates of the
``num_terms`` individual Pauli terms FIRST using the SAME real
coefficients the operator itself uses, THEN apply the point estimator
(mean or median-of-means) to that combined per-shot stream -- this is
exact for "mean" by linearity, and is the standard classical-shadows
treatment of a linear combination for "median_of_means").

ESTIMATOR AND METRICS: UNCHANGED FROM exp_9. ENSEMBLES: exp_9's FOUR, PLUS
ONE EXTRA (seeqst_binomial_untuned) FOR DIRECT q CONTROL
--------------------------------------------------------------------------------
    pauli                     PauliEnsemble()                        (beta = 3^weight(P))
    clifford                  CliffordEnsemble()                     (beta = 2^n+1, exact)
    seeqst_uniform            SEEQSTEnsemble()                       (flat/uniform SEEQST, q=1/2)
    seeqst_binomial_tuned     SEEQSTBinomialEnsemble(q=m/n)          (tuned to the ceiling m;
                                                                       q NOT independently settable)
    seeqst_binomial_untuned   SEEQSTBinomialEnsemble(q=q_untuned)    (q FREELY settable via
                                                                       --q-binomial-untuned, default
                                                                       1/(n+1) -- exp_8's own pattern)

    --estimator {mean, median_of_means}, with the same K-clamping behavior
    as exp_9/exp_10 -- see ``tasks/exp_9_tuned_locality_mom.py``.

    RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
    (i now indexes COMBINED OPERATORS, not individual Pauli strings)

A TUNING CAVEAT WORTH KNOWING BEFORE READING RESULTS
--------------------------------------------------------------------------------
``notes/working/seeqst_sparse_tuning.tex``'s q*=m/n theorem is about the
shadow-norm bound of the FULL S_m-basis sum (every eligible term, not a
random subset). exp_11's operators sum only ``num_terms`` randomly sampled
eligible terms, so exp_9/exp_10's q=m/n rule (reused here for consistency)
is a theory-motivated default, not a proven-optimal tuning for this exact
construction. See ``tasks/exp_11_sparse_operator_mom.py``'s module
docstring for the full explanation, and ``experiments/exp_8/exp_8.md`` for
the related (and, past m=n/2, DIFFERENT) two-regime result for the
list-of-separate-observables object exp_8/exp_10 track instead.

NEW HYPERPARAMETERS (none of these exist in exp_9, since exp_9 has no
"combine several terms into one operator" step)
--------------------------------------------------------------------------------
  --num-terms       number of Pauli terms summed into EACH combined
                     operator (default 5)
  --min-xy          floor on each term's X/Y-weight; default 1 (keeps
                     every term genuinely X/Y-bearing, matching exp_8/
                     exp_10's own generator convention)
  --rest-mode       "random" | "z" | "identity" -- how each term's non-XY
                     qubits are set (default "random"; unlike exp_9's
                     --rest-mode, which governed ONE exact-weight
                     observable, this now applies per-term inside the sum
                     -- see pauli_utils.random_capped_xy_spec)
  --coeff-dist      "normal" | "uniform_pm1" -- each operator's real term
                     coefficients (default "normal")
  --q-binomial-untuned  per-qubit Bernoulli inclusion probability q for the
                     EXTRA seeqst_binomial_untuned ensemble -- the direct
                     way to "select the binomial parameter" -- default
                     1/(n+1) (exp_6/exp_8's own default untuned q)

EVERYTHING ELSE IS EXP_9'S OWN CLI, UNCHANGED
--------------------------------------------------------------------------------
  --n, --m, --estimator, --mom-num-groups, --mom-delta, --num-observables,
  --n-samples, --num-state-repeats, --num-observable-repeats, --states,
  --seed, --version, --quick, --quiet

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_11/v<N>/:
  hyperparameters.json                  all settings used for this run
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png       )  one figure per state type,
  max_error_vs_nsample_<state>.png  )  one metric per figure, 5 ensemble lines

USAGE
--------------------------------------------------------------------------------
    python exp_11.py --quick
    python exp_11.py --n 6 --m 5
    python exp_11.py --n 6 --m 5 --num-terms 8 --coeff-dist uniform_pm1
    python exp_11.py --n 6 --m 5 --estimator median_of_means
    python exp_11.py --n 7 --m 3 --version my_run_name   # explicit output folder name
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
from tasks.exp_11_sparse_operator_mom import Exp11Config, run_exp11, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_11/results/), same convention as every other exp_*.py.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_11"

METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse"),
    ("max_error_mean", "max_error_std", "Max error", "max_error"),
]
ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
    "seeqst_binomial_tuned": ("D", "tab:red"),
    "seeqst_binomial_untuned": ("P", "tab:pink"),
}


def build_ensembles(n: int, m: int, q_untuned: float) -> dict:
    # seeqst_binomial_tuned is byte-for-byte exp_9's build_ensembles -- see
    # the module docstring's "A TUNING CAVEAT" section for what q=m/n does
    # and does not prove here. seeqst_binomial_untuned is exp_11's answer to
    # "how do I select the binomial parameter directly" -- a SECOND
    # SEEQSTBinomialEnsemble instance at a freely-chosen, independently-set
    # q (--q-binomial-untuned), exactly the same pattern exp_8 uses for its
    # own contrast ensemble (see exp_8.py::build_ensembles) -- the TUNED
    # ensemble's q stays algorithmic (m/n), not overridable, by design.
    q_tuned = m / n
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
        "seeqst_binomial_tuned": SEEQSTBinomialEnsemble(q_tuned),
        "seeqst_binomial_untuned": SEEQSTBinomialEnsemble(q_untuned),
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
        help="ceiling on each term's X/Y-locality; must satisfy 1<=m<=n-1",
    )
    parser.add_argument(
        "--num-terms", type=int, default=5,
        help="number of Pauli terms summed into EACH combined S_m-sparse operator",
    )
    parser.add_argument(
        "--min-xy", type=int, default=1,
        help="floor on each term's X/Y-weight (default 1, keeps every term X/Y-bearing)",
    )
    parser.add_argument(
        "--rest-mode", type=str, default="random", choices=["random", "z", "identity"],
        help="how each term's non-XY qubits are set ('random': iid Z/I per qubit, default. "
        "'z': all Z. 'identity': left as identity)",
    )
    parser.add_argument(
        "--coeff-dist", type=str, default="normal", choices=["normal", "uniform_pm1"],
        help="distribution for each combined operator's real term coefficients "
        "('normal': iid standard normal, default. 'uniform_pm1': iid Uniform{-1,+1})",
    )
    parser.add_argument(
        "--q-binomial-untuned", type=float, default=None,
        help="per-qubit Bernoulli inclusion probability q for the EXTRA "
        "seeqst_binomial_untuned ensemble -- freely settable, unlike "
        "seeqst_binomial_tuned's q, which is always exactly m/n (default: "
        "1/(n+1), exp_6/exp_8's own default untuned q)",
    )
    parser.add_argument(
        "--estimator", type=str, default="mean", choices=["mean", "median_of_means"],
        help="point-estimator used to turn the first N_sample shots into a prediction "
        "('mean': running empirical mean, exp_7/exp_9's estimator. 'median_of_means': HKP's "
        "own estimator -- split into K groups, average within each, median across groups -- "
        "applied to the COMBINED per-shot stream, see the module docstring)",
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
    parser.add_argument(
        "--num-observables", type=int, default=10,
        help="M, number of independent random COMBINED operators",
    )
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
        help="inner layer: how many random combined-operator sets to draw per state",
    )
    parser.add_argument(
        "--states", nargs="+", default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_11/ (default: auto-incrementing v0, v1, ...)",
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
        args.num_terms = 3
        args.num_observables = 5
        args.n_samples = [10, 50, 200, 800]
        args.num_state_repeats = 2
        args.num_observable_repeats = 2

    if not (1 <= args.m <= args.n - 1):
        parser.error(f"--m must satisfy 1 <= m <= n-1 (got m={args.m}, n={args.n})")
    if not (0 <= args.min_xy <= args.m):
        parser.error(f"--min-xy must satisfy 0 <= min-xy <= m (got min_xy={args.min_xy}, m={args.m})")
    if args.num_terms < 1:
        parser.error(f"--num-terms must be >= 1 (got {args.num_terms})")
    if args.q_binomial_untuned is not None and not (0.0 < args.q_binomial_untuned < 1.0):
        parser.error(
            f"--q-binomial-untuned must be strictly between 0 and 1 (got "
            f"{args.q_binomial_untuned}) -- q=0 or 1 makes any X/Y-containing "
            f"observable's inverse weight blow up (q**-k divides by zero for k>0)"
        )

    q_untuned = args.q_binomial_untuned if args.q_binomial_untuned is not None else 1.0 / (args.n + 1)

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
                f"tasks/exp_11_sparse_operator_mom.py::_checkpoint_estimates."
            )
    else:
        mom_num_groups = 1  # unused by the "mean" estimator; kept as a valid config value

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp11Config(
        n=args.n,
        m=args.m,
        num_terms=args.num_terms,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(args.n, args.m, q_untuned),
        estimator=args.estimator,
        mom_num_groups=mom_num_groups,
        min_xy=args.min_xy,
        rest_mode=args.rest_mode,
        coeff_dist=args.coeff_dist,
        state_types=args.states,
        seed=args.seed,
    )

    print(
        f"Running exp_11 -> {out_dir}\n"
        f"  n={args.n}, m={args.m} (ceiling: term weight ~ Uniform{{{args.min_xy},...,{args.m}}}), "
        f"q_tuned={args.m / args.n:.6g}\n"
        f"  num_terms={args.num_terms}, rest_mode={args.rest_mode!r}, coeff_dist={args.coeff_dist!r}\n"
        f"  q_untuned={q_untuned:.6g}\n"
        f"  estimator={args.estimator!r}"
        + (f", mom_num_groups={mom_num_groups}, mom_delta={args.mom_delta}" if args.estimator == "median_of_means" else "")
        + f"\n  M={args.num_observables}, n_samples={n_samples_sorted}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp11(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "m": args.m,
        "num_terms": args.num_terms,
        "min_xy": args.min_xy,
        "rest_mode": args.rest_mode,
        "coeff_dist": args.coeff_dist,
        "q_tuned": args.m / args.n,
        "q_tuned_is_minimax_optimal_full_basis": 2 * args.m <= args.n,
        "q_untuned": q_untuned,
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
