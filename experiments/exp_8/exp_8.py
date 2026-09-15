#!/usr/bin/env python3
"""
================================================================================
 exp_8 -- Ceiling-tuned SEEQST: sparse-m X/Y observables (weight <= m)
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_7's "exact"/"at_least" families tune SEEQST to a KNOWN LOWER value of
the X/Y-locality (exactly m, or at least m). This experiment tunes it to a
known UPPER bound instead: every observable has X/Y-weight drawn uniformly
from {1, ..., m} (never above m) and every other qubit is Z
(``pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)`` -- the SAME
generator exp_6 already uses for its own untuned family; exp_8 is exp_6's
observable family with the SEEQST side finally tuned to the cap instead of
left at a fixed q=1/(n+1)). This is the operational counterpart of the
m-SPARSE operators studied in ``notes/working/seeqst_sparse_tuning.tex``
(there: S_m = {s subseteq [n] : |s| <= m}, the set of X/Y-support patterns
allowed to appear) -- see "WHY q=m/n IS STILL RIGHT HERE" below for exactly
how that note's result does, and does not, carry over to this experiment.

SIX ensembles are compared, exactly exp_7's roster with "exact"/"at_least"
swapped for a single ceiling-tuned family:

    pauli                     PauliEnsemble()                       (beta = 3^weight(P))
    clifford                  CliffordEnsemble()                    (beta = 2^n+1, exact)
    seeqst_uniform            SEEQSTEnsemble()                      (flat/uniform SEEQST, q=1/2)
    seeqst_binomial_tuned     SEEQSTBinomialEnsemble(q=m/n)         (tuned to the CEILING m)
    seeqst_binomial_untuned   SEEQSTBinomialEnsemble(q=1/(n+1))     (exp_6's baseline q, for contrast)
    seeqst_unifsize_tuned     SEEQSTUniformSizeEnsemble(l=m)        (cap = m always; unlike exp_7's
                                                                     "at_least" mode, this NEVER needs
                                                                     the ValueError-avoiding widening --
                                                                     true weight is <= m by construction,
                                                                     so l=m is safe unconditionally)

WHY q=m/n IS RIGHT HERE *ONLY FOR m <= n/2* -- A TWO-REGIME RESULT, AND WHY
THIS IS *NOT* A RESTATEMENT OF ``seeqst_sparse_tuning.tex``'s THEOREM
--------------------------------------------------------------------------------
``notes/working/seeqst_sparse_tuning.tex`` proves that for a SINGLE combined
m-sparse operator A = sum_{s in S_m} c_s P_s (i.e. estimating one linear
combination whose Pauli terms all happen to have X/Y-support in S_m), the
Binomial family's worst-case shadow norm bound
||A||^2_sh <= 8 (sum_{s in S_m} 1/p(s)) ||A||_infty^2 is minimized, to
leading asymptotic order, at q* = m/n rather than the source material's
q=1/(n+1) -- via a genuinely combinatorial argument (Phi_m(q) = sum_{w=0}^m
C(n,w) q^-w (1-q)^-(n-w), dominant-term analysis as n->infinity).

exp_8 tests a DIFFERENT object: a LIST of M separately-tracked observables
(the max_i beta(P_i) sample-complexity criterion of Theorem 1 /
Huang-Kueng-Preskill Prop. S1), each individually having some X/Y-weight
k in {1, ..., m}, not a single sum. For this object the right argument is
the SIMPLE, single-string one already used by exp_7 -- but it needs BOTH
directions of q, not just q<1/2, and that turns out to matter:

  1. ln beta_q(k) = ln2 - k ln q - (n-k) ln(1-q) is LINEAR in k, with slope
     ln((1-q)/q). So on any fixed range k in {1,...,m}, beta_q(.) is
     monotonic in k -- INCREASING if q<1/2, DECREASING if q>1/2, and flat
     (constant = 2^(n+1)) at exactly q=1/2. Its max over k in {1,...,m} is
     therefore always at one of the two ENDPOINTS: beta_q(m) if q<=1/2,
     beta_q(1) if q>=1/2 -- never in the interior, and the two branches
     agree (both equal 2^(n+1)) at q=1/2.
  2. Minimizing branch q<=1/2 (i.e. beta_q(m)) is exactly Theorem 1 of
     ``notes/seeqst_sample_complexity.tex`` / ``main_theorem.tex``: the
     unconstrained minimizer is q=m/n. If m<=n/2 this lies inside the
     branch's own domain q<=1/2 -- self-consistent -- giving the value
     2^(1+n H2(m/n)). If m>n/2, the unconstrained minimizer q=m/n>1/2 falls
     OUTSIDE this branch's domain, so (beta_q(m) being convex-log in q,
     decreasing then increasing, with its true minimum past q=1/2) the
     CONSTRAINED minimum on q<=1/2 sits at the boundary q=1/2 itself,
     value 2^(n+1).
  3. Minimizing branch q>=1/2 (i.e. beta_q(1)) is the SAME theorem at k=1:
     unconstrained minimizer q=1/n, which is <1/2 for any n>2 -- always
     OUTSIDE this branch's domain. So this branch's constrained minimum is
     also always at its boundary q=1/2, value 2^(n+1).
  4. Taking the better of the two branches:

         min_q  max_{k=1}^{m} beta_q(k)
             =  2^(1 + n H2(m/n))   at  q* = m/n     if  m <= n/2
             =  2^(n+1)             at  q* = 1/2     if  m >  n/2

     -- i.e. q*=m/n is the correct, UNIQUE minimax tuning only in the
     genuinely "sparse" regime m<=n/2 (2^(1+nH2(m/n)) <= 2^(n+1) there,
     H2<=1). Past that point, ANY value of q other than 1/2 is actively
     worse in the worst case than not tuning at all -- the best a
     ceiling-tuned Binomial ensemble can do once the ceiling exceeds n/2 is
     fall back to being exactly ``seeqst_uniform``.

This two-branch structure -- not just the m<=n/2 half -- was checked
numerically before finalizing this experiment (``minimize_scalar`` over
max_{k=1}^m beta_q(k), swept over m=1..n-1 at several n, compared against
the piecewise closed form above: exact machine-precision agreement in
BOTH regimes, boundary at m=n/2 included). An earlier draft of this
docstring stated q*=m/n unconditionally, reasoning only from the q<1/2
half of point 1 above and never checking m>n/2 numerically -- wrong, and
caught precisely because ``tests/verify_exp8_observables.py`` checks the
full range 1<=m<=n-1, not just a few small-m examples.

Because of this, exp_8's default ``seeqst_binomial_tuned`` still uses the
literal q=m/n construction (matching exp_7's naming and the "naive m/n
tuning" this experiment is precisely built to stress-test), and exp_8.py
prints a warning whenever --m > n/2 explaining that the configured tuning
is no longer minimax-optimal there (see ``build_ensembles`` below) --
rather than silently substituting q=1/2, which would hide the very
breakdown this experiment exists to show.

The two arguments (this one and ``seeqst_sparse_tuning.tex``'s) happen to
reach the SAME q*=m/n in the regime where both are meaningful (m<=n/2,
which is also where "m-sparse" is a sensible description of the family in
the first place), but for structurally different reasons and different
objects (list-max vs. summed-operator bound) -- this experiment is a
direct empirical test of the argument above (exp_7-style: does
ceiling-tuned SEEQST-binomial beat untuned and beat Pauli on a weight-<=m
observable list, and does it stop mattering past m=n/2?), not of the
harder sparse-tuning note's summed-operator claim, and exp_8.md is
explicit about not conflating the two.

The two-layer randomization design, running-mean estimator, and per-ensemble
snapshot-stream reuse are IDENTICAL to exp_1/exp_6/exp_7 -- see exp_1.py's
module docstring for the full rationale.

METRICS -- ONLY TWO (no mean_variance), same as exp_6/exp_7
--------------------------------------------------------------------------------
    RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
averaged over BOTH randomization layers (every outer state x every inner
observable-set draw, pooled together).

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --m                       the swept hyperparameter: ceiling on X/Y-weight
                            (weight ~ Uniform{1,...,m}). Must satisfy
                            1<=m<=n-1 (needed for the tuned q*=m/n to lie
                            strictly in (0,1)).
  --q-binomial-untuned      seeqst_binomial_untuned's q (default: 1/(n+1),
                            exp_6's own default -- the tuned ensemble's q is
                            always exactly m/n, not independently settable)
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
Each run is saved to its own versioned subfolder results/exp_8/v<N>/:
  hyperparameters.json                  all settings used for this run
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png       )  one figure per state type,
  max_error_vs_nsample_<state>.png  )  one metric per figure, 6 ensemble lines

To see how the metrics depend on m itself (holding n and n_sample fixed),
run this script several times at different --m and pass the resulting
results/exp_8/v<N>/ directories to
``experiments/exp_8/plot_m_dependence.py``.

USAGE
--------------------------------------------------------------------------------
    python exp_8.py --quick
    python exp_8.py --n 6 --m 2
    python exp_8.py --n 6 --m 5
    python exp_8.py --n 7 --m 3 --version my_run_name   # explicit output folder name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives two levels below the shadow_benchmark root (same as
# every other exp_*.py), so we add that root to sys.path ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from ensembles.seeqst_unifsize_ensemble import SEEQSTUniformSizeEnsemble
from tasks.exp_8_sparse_locality import Exp8Config, run_exp8, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_8/results/), same convention as every other exp_*.py.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_8"

METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
]
ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
    "seeqst_binomial_tuned": ("D", "tab:red"),
    "seeqst_binomial_untuned": ("P", "tab:pink"),
    "seeqst_unifsize_tuned": ("v", "tab:purple"),
}


def build_ensembles(n: int, m: int, q_untuned: float) -> dict:
    """seeqst_binomial_tuned's q=m/n and seeqst_unifsize_tuned's l=m are both
    tuned to the CEILING m -- and, unlike exp_7's "at_least" mode,
    seeqst_unifsize_tuned needs no special-casing here: the observable
    generator (``random_bounded_xy_spec(n, l=m, rng, min_xy=1)``) can never
    produce a weight above m in the first place, so l=m never triggers
    SEEQSTUniformSizeEnsemble.inverse_weight's ValueError (see that class's
    docstring) the way it would if the true weight could exceed the tuned
    cap. This is a genuine, worth-noting structural difference from exp_7's
    "at_least" family, not an oversight.

    seeqst_binomial_tuned's q is the literal q=m/n construction UNCONDITIONALLY
    (even for m>n/2) -- see the module docstring's "WHY q=m/n IS RIGHT HERE
    *ONLY FOR m<=n/2*" section for the full two-regime derivation. Past
    m=n/2 this q is no longer minimax-optimal (q*=1/2, i.e. seeqst_uniform,
    is) -- ``main()`` prints a warning in that regime rather than silently
    changing what "tuned" means, since demonstrating that breakdown is part
    of the point of running this experiment past m=n/2."""
    q_tuned = m / n
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
        "seeqst_binomial_tuned": SEEQSTBinomialEnsemble(q_tuned),
        "seeqst_binomial_untuned": SEEQSTBinomialEnsemble(q_untuned),
        "seeqst_unifsize_tuned": SEEQSTUniformSizeEnsemble(m),
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
    parser.add_argument("--n", type=int, default=6, help="system size (number of qubits)")
    parser.add_argument(
        "--m", type=int, default=2,
        help="swept hyperparameter: ceiling on X/Y-weight (weight ~ Uniform{1,...,m}); "
        "must satisfy 1<=m<=n-1",
    )
    parser.add_argument(
        "--q-binomial-untuned", type=float, default=None,
        help="per-qubit Bernoulli inclusion probability for seeqst_binomial_untuned "
        "(default: 1/(n+1), exp_6's own default)",
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
        help="output subfolder name under results/exp_8/ (default: auto-incrementing v0, v1, ...)",
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

    if 2 * args.m > args.n:
        print(
            f"WARNING: --m={args.m} > n/2={args.n / 2:g}. q=m/n={args.m / args.n:.4f} is NOT "
            "minimax-optimal in this regime -- the module docstring's two-regime derivation "
            "('WHY q=m/n IS RIGHT HERE ONLY FOR m<=n/2') shows the true minimax tuning past "
            "m=n/2 is q*=1/2 (i.e. seeqst_binomial_tuned can do no better than seeqst_uniform "
            "here). seeqst_binomial_tuned is still run with the literal q=m/n construction "
            "below -- this run is a demonstration of THAT breakdown, not a bug."
        )

    q_untuned = args.q_binomial_untuned if args.q_binomial_untuned is not None else 1.0 / (args.n + 1)

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp8Config(
        n=args.n,
        m=args.m,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(args.n, args.m, q_untuned),
        state_types=args.states,
        seed=args.seed,
    )

    print(
        f"Running exp_8 -> {out_dir}\n"
        f"  n={args.n}, m={args.m} (ceiling: weight ~ Uniform{{1,...,{args.m}}})\n"
        f"  q_tuned={args.m / args.n:.6g}, q_untuned={q_untuned:.6g}, "
        f"M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp8(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "m": args.m,
        "q_tuned": args.m / args.n,
        "q_tuned_is_minimax_optimal": 2 * args.m <= args.n,
        "q_untuned": q_untuned,
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
