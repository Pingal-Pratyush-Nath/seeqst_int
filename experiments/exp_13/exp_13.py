#!/usr/bin/env python3
"""
================================================================================
 exp_13 -- ground-state energy of a long-range Kitaev chain, from shadows
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_13 is exp_12's exact pipeline (fixed state = exact ground state of a
fixed Hamiltonian, fixed observable = H itself, combine-then-estimate,
--num-repeats independent shadow draws for error bars -- see
``experiments/exp_12/exp_12.py``'s module docstring) pointed at a
DIFFERENT Hamiltonian: ``common.jw_hamiltonian.jw_long_range_kitaev``, the
open-boundary long-range p-wave pairing Kitaev chain (Vodola et al., PRL
113, 156402 (2014)):

    H = -hopping * sum_j (a_j^dagger a_{j+1} + h.c.)
        - mu * sum_j (n_j - 1/2)
        + (delta/2) * sum_{j<k} (a_j a_k + a_k^dagger a_j^dagger) / (k-j)^alpha

Unlike exp_12's ``demo_integrals`` (nearest-neighbour-only, Pauli weight
never exceeds 2 regardless of n), a pairing term spanning sites j and k
picks up a Jordan-Wigner Z-string over every site strictly between them, so
its Pauli weight is k-j+1 -- genuine weights up to n. This is the
Hamiltonian discussed for exp_12's "why doesn't SEEQST ever beat Pauli"
question -- see ``jw_hamiltonian.jw_long_range_kitaev``'s docstring for the
full parameter-regime caveat.

DEFAULTS ARE CHOSEN TO ACTUALLY SHOW THE EFFECT, NOT THE MODEL'S OWN
"NATURAL" PHYSICS DEFAULTS
--------------------------------------------------------------------------------
With this model's own naturally-quoted defaults (hopping=mu=delta=1,
alpha=1), Pauli still wins overall -- the O(1) hopping/onsite/short-range
terms dominate the coefficient-squared mass even though rare high-weight
terms exist. This script instead defaults to a regime that was checked
numerically to show SEEQST/Clifford beating Pauli: ``--hopping 0 --mu 0
--delta 1 --alpha 0 --n 10`` (fully uniform, all-to-all long-range
pairing, no competing local physics) -- at those settings, q=1/(n+1) beats
Pauli by roughly 5.7x and Clifford beats Pauli by roughly 2.9x in the
``sum c_i^2 beta(P_i)`` proxy. Override any of ``--hopping/--mu/--delta/
--alpha`` to explore other regimes (e.g. the model's own "natural"
defaults, to see Pauli win again).

ENSEMBLES AND q-VALUES
--------------------------------------------------------------------------------
Same ensemble roster as exp_12: pauli, clifford, seeqst_uniform, plus one
SEEQSTBinomialEnsemble per ``--q-values`` entry. Default q-values are
``1/(n+1)`` (the "generic"/untuned choice -- Equation 7 / Example 2 of
SEEQST_shadows4.pdf, the one that actually wins here) and ``0.5``. An
AGGRESSIVELY high-weight-tuned q (close to 1) was checked to be
dramatically WORSE than Pauli in this regime (q=0.8 was >400x worse at
n=10, alpha=0) -- don't assume "more tuned toward high weight" is better
here the way it was for a single fixed-weight observable in exp_9.

USAGE
--------------------------------------------------------------------------------
    python exp_13.py --quick
    python exp_13.py --n 10
    python exp_13.py --n 10 --q-values 0.0909090909 0.5 --num-repeats 30
    python exp_13.py --n 10 --hopping 1 --mu 1 --alpha 1   # the model's own "natural" defaults -- Pauli wins again
    python exp_13.py --n 12 --alpha 0.25

VALIDATION NOTE
--------------------------------------------------------------------------------
Same limitation as exp_12: qiskit is not installable in either sandbox
this was authored from, so the qiskit-touching pieces (Statevector,
evaluate_snapshots) could not be executed end-to-end from there. What WAS
verified with plain numpy: the Hamiltonian construction gives exactly real
coefficients and an exactly Hermitian matrix (checked at n=6 for several
parameter combinations, including the working regime); the
Pauli-weight-up-to-n structure was confirmed directly; and the
"sum c_i^2 beta(P_i)" proxy numbers quoted above and in
``jw_hamiltonian.jw_long_range_kitaev``'s docstring were computed with an
independent from-scratch reimplementation before this file was written.
Please run ``--quick`` first and sanity-check E_true before a full run.
Note also that exact diagonalization (for E_true and the ground state) is
dense 2**n x 2**n -- fine through n~10-12, impractical much beyond that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from common import jw_hamiltonian
from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from tasks.exp_13_kitaev_energy_mom import Exp13Config, run_exp13, summarize

BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_13"

BASE_ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
}
_Q_COLORS = ["tab:red", "tab:pink", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
_Q_MARKERS = ["D", "P", "X", "v", "*", "h"]


def build_ensembles(q_values: list[float]) -> tuple[dict, dict]:
    """Identical pattern to exp_12's ``build_ensembles`` -- one
    SEEQSTBinomialEnsemble per ``q_values`` entry, no q=m/n rule (H mixes
    many different weights at once, see module docstring)."""
    ensembles: dict = {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
    }
    style = dict(BASE_ENSEMBLE_STYLE)
    for i, q in enumerate(q_values):
        name = f"seeqst_binomial_q{q:.4g}"
        ensembles[name] = SEEQSTBinomialEnsemble(q)
        style[name] = (
            _Q_MARKERS[i % len(_Q_MARKERS)],
            _Q_COLORS[i % len(_Q_COLORS)],
        )
    return ensembles, style


def plot_energy_error(summary, ensemble_style: dict, e_true: float, num_terms: int, params: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ens_name, (marker, color) in ensemble_style.items():
        se = summary[summary.ensemble == ens_name].sort_values("n_sample")
        if se.empty:
            continue
        ax.errorbar(
            se.n_sample, se.abs_error_mean, yerr=se.abs_error_std,
            marker=marker, color=color, label=ens_name, capsize=2, lw=1.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
    ax.set_ylabel("|E_hat - E_true|  (ground-state energy error)")
    ax.set_title(
        f"Long-range Kitaev chain energy error vs. $N_{{sample}}$\n"
        f"(n={params['n']}, hopping={params['hopping']}, mu={params['mu']}, "
        f"delta={params['delta']}, alpha={params['alpha']}, E_true={e_true:.6g}, {num_terms} terms)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_error_vs_nsample.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="system size (number of qubits/fermionic modes)")
    parser.add_argument("--hopping", type=float, default=0.0, help="nearest-neighbour hopping amplitude w")
    parser.add_argument("--mu", type=float, default=0.0, help="chemical potential")
    parser.add_argument("--delta", type=float, default=1.0, help="long-range p-wave pairing amplitude")
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="power-law decay exponent for the pairing term (0 = fully uniform/"
        "all-to-all, no decay -- see module docstring for why the default "
        "parameter set here differs from the model's own 'natural' physics defaults)",
    )
    parser.add_argument(
        "--q-values", type=float, nargs="+", default=None,
        help="one SEEQSTBinomialEnsemble per value (default: 1/(n+1) and 0.5 -- "
        "see module docstring for why an aggressively high-weight-tuned q is "
        "NOT a good default here)",
    )
    parser.add_argument(
        "--tune-q", action=argparse.BooleanOptionalAction, default=True,
        help="ALSO add one extra SEEQSTBinomialEnsemble whose q is tuned "
        "directly to THIS Hamiltonian via jw_hamiltonian.optimal_q (argmin "
        "over q of sum_i coeffs[i]^2 * beta_q(specs[i]), a dependency-free "
        "grid search) -- shown separately in the plot as "
        "'seeqst_binomial_tuned'. On by default; pass --no-tune-q to skip. "
        "See jw_hamiltonian.optimal_q's docstring for why this recovers the "
        "textbook q=m/n rule when every term shares one X/Y-weight (as this "
        "chain's pairing terms do), and can differ from it otherwise.",
    )
    parser.add_argument("--estimator", type=str, default="mean", choices=["mean", "median_of_means"])
    parser.add_argument("--mom-num-groups", type=int, default=None)
    parser.add_argument("--mom-delta", type=float, default=0.1)
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
    )
    parser.add_argument(
        "--num-repeats", type=int, default=20,
        help="independent draws of the shadow snapshot stream (state and "
        "observable are both FIXED here -- see exp_12's module docstring)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n = 6
        args.hopping = 0.0
        args.mu = 0.0
        args.delta = 1.0
        args.alpha = 0.0
        args.q_values = None
        args.n_samples = [10, 50, 200, 800]
        args.num_repeats = 3

    q_values = args.q_values if args.q_values is not None else [1.0 / (args.n + 1), 0.5]
    if len(q_values) == 0 and not args.tune_q:
        parser.error("--q-values needs at least one value, or pass --tune-q")
    for q in q_values:
        if not (0.0 < q < 1.0):
            parser.error(f"--q-values entries must be strictly between 0 and 1 (got {q})")

    q_tuned = None
    if args.tune_q:
        coeffs, specs = jw_hamiltonian.jw_long_range_kitaev(
            args.n, args.hopping, args.mu, args.delta, args.alpha
        )
        q_tuned = jw_hamiltonian.optimal_q(coeffs, specs, args.n)
        print(
            f"--tune-q: optimal q for this exact Hamiltonian (n={args.n}, "
            f"hopping={args.hopping}, mu={args.mu}, delta={args.delta}, "
            f"alpha={args.alpha}) = {q_tuned:.6g} "
            f"(argmin of sum c_i^2 beta_q(P_i) -- see jw_hamiltonian.optimal_q)"
        )

    n_samples_sorted = sorted(set(args.n_samples))

    if args.estimator == "median_of_means":
        if args.mom_num_groups is not None:
            mom_num_groups = args.mom_num_groups
            if mom_num_groups < 1:
                parser.error(f"--mom-num-groups must be >= 1 (got {mom_num_groups})")
        else:
            mom_num_groups = 2 * math.ceil(math.log(2 * 1 / args.mom_delta))
        if n_samples_sorted[0] < mom_num_groups:
            print(
                f"Note: K={mom_num_groups} exceeds the smallest --n-samples checkpoint "
                f"({n_samples_sorted[0]}); that checkpoint (and any other below K) uses "
                f"K_eff=n_sample groups of size 1 instead."
            )
    else:
        mom_num_groups = 1

    ensembles, ensemble_style = build_ensembles(q_values)
    if q_tuned is not None:
        ensembles["seeqst_binomial_tuned"] = SEEQSTBinomialEnsemble(q_tuned)
        ensemble_style["seeqst_binomial_tuned"] = ("*", "black")

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp13Config(
        n=args.n,
        hopping=args.hopping,
        mu=args.mu,
        delta=args.delta,
        alpha=args.alpha,
        n_samples=n_samples_sorted,
        num_repeats=args.num_repeats,
        ensembles=ensembles,
        estimator=args.estimator,
        mom_num_groups=mom_num_groups,
        seed=args.seed,
    )

    print(
        f"Running exp_13 -> {out_dir}\n"
        f"  n={args.n}, hopping={args.hopping}, mu={args.mu}, delta={args.delta}, alpha={args.alpha}\n"
        f"  q_values={q_values}\n"
        f"  estimator={args.estimator!r}"
        + (f", mom_num_groups={mom_num_groups}, mom_delta={args.mom_delta}" if args.estimator == "median_of_means" else "")
        + f"\n  n_samples={n_samples_sorted}, num_repeats={args.num_repeats}, seed={args.seed}"
    )
    t0 = time.time()
    raw, e_true, num_terms = run_exp13(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows, E_true={e_true:.10g}, {num_terms} Pauli terms")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "hopping": args.hopping,
        "mu": args.mu,
        "delta": args.delta,
        "alpha": args.alpha,
        "q_values": q_values,
        "q_tuned": q_tuned,
        "estimator": args.estimator,
        "mom_num_groups": mom_num_groups,
        "mom_delta": args.mom_delta,
        "n_samples": n_samples_sorted,
        "num_repeats": args.num_repeats,
        "seed": args.seed,
        "version": out_dir.name,
        "elapsed_seconds": round(elapsed, 1),
        "num_raw_rows": len(raw),
        "e_true": e_true,
        "num_pauli_terms": num_terms,
    }
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print(f"Wrote raw.csv, summary.csv, hyperparameters.json to {out_dir}")

    plot_energy_error(
        summary, ensemble_style, e_true, num_terms,
        {"n": args.n, "hopping": args.hopping, "mu": args.mu, "delta": args.delta, "alpha": args.alpha},
        out_dir,
    )
    print(f"Wrote energy_error_vs_nsample.png to {out_dir}")


if __name__ == "__main__":
    main()
