#!/usr/bin/env python3
"""
================================================================================
 exp_12 -- estimating a real Hamiltonian's ground-state ENERGY from shadows
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
Unlike exp_9-exp_11 (many random synthetic observables against many random
states), exp_12 targets one fixed, physically meaningful problem: build a
concrete numeric Jordan-Wigner electronic Hamiltonian (via
``common.jw_hamiltonian`` -- the exact algebra verified against
``tests/JW_transformation.ipynb``, see that module's docstring), find its
EXACT ground state and ground energy ``E_true`` by direct diagonalization
(cheap at n<=6, a 64x64 dense matrix), and then ask: starting from ONLY
classical-shadow measurements of that ground state, how accurately does
each ensemble's shadow estimator recover ``E_true``?

The Hamiltonian used by default (``--integrals demo``) is
``jw_hamiltonian.demo_integrals``: a small, simple, HAND-PICKED (not a real
molecule -- no chemistry package used or required) n-site model with a
linear on-site energy ladder, nearest-neighbour hopping, and
nearest-neighbour density-density interaction. See that function's
docstring for the exact numbers.

H = sum_i coeffs[i] * P_i has O(100-600) Pauli terms at n=6 (see the
``JW_transformation.ipynb`` weight-distribution discussion). The energy
estimate is formed by combining every term's per-shot shadow estimate with
H's OWN coefficients before applying the point estimator -- exactly
exp_11's "combine-then-estimate" design; see
``tasks/exp_12_hamiltonian_energy_mom.py``'s module docstring.

ENSEMBLES: PAULI, CLIFFORD, SEEQST-UNIFORM, PLUS ONE SEEQSTBinomialEnsemble
PER --q-values ENTRY
--------------------------------------------------------------------------------
    pauli                  PauliEnsemble()                    (beta = 3^weight(P))
    clifford               CliffordEnsemble()                 (beta = 2^n+1, exact)
    seeqst_uniform         SEEQSTEnsemble()                   (flat/uniform SEEQST, q=1/2)
    seeqst_binomial_q<q>   SEEQSTBinomialEnsemble(q)          (one per --q-values entry,
                                                                freely chosen, e.g. 4/6, 2/6)

Unlike exp_9/exp_10/exp_11, there is no single "correct" m here (H mixes
many different X/Y-weights at once -- see the weight-distribution
discussion in ``JW_transformation.ipynb``), so exp_12 does NOT compute a
q=m/n "tuned" ensemble automatically. Instead, pass however many q-values
you want to compare directly via ``--q-values`` (default: 4/6 and 2/6, per
the original request).

    RMSE-style metric here collapses to a single scalar since there's only
    ONE observable (H itself): ``abs_error(N_sample) = |E_hat(N_sample) -
    E_true|``, averaged (with std-dev) across ``--num-repeats`` independent
    draws of the shadow snapshot stream.

USAGE
--------------------------------------------------------------------------------
    python exp_12.py --quick
    python exp_12.py --n 6 --q-values 0.667 0.333
    python exp_12.py --n 6 --q-values 0.6666666667 0.3333333333 --num-repeats 30
    python exp_12.py --n 6 --estimator median_of_means

VALIDATION NOTE
--------------------------------------------------------------------------------
qiskit is not installable in the environment this file was authored in
(neither the cloud sandbox nor the local device sandbox used to write and
py_compile-check it), so the qiskit-touching pieces (Statevector
construction, evaluate_snapshots) could not be executed end-to-end from
there. What WAS verified, with plain numpy, no qiskit/sympy required (see
``common/jw_hamiltonian.py``'s module docstring and the conversation around
``tests/JW_transformation.ipynb``):
  - the Jordan-Wigner algebra itself: canonical anticommutation relations,
    Hermiticity for symmetric integrals, and exact agreement with the
    analytic non-interacting spectrum;
  - the spec re-indexing into this codebase's convention, checked
    bit-for-bit against ``pauli_utils.spec_to_label``'s actual output;
  - ``demo_integrals``' Hermiticity (h symmetric, v's 4-fold chemist
    symmetry satisfied by construction).
Please run ``--quick`` first on your machine (where qiskit is installed)
before a full run, and sanity-check that the printed ``E_true`` looks like
a reasonable ground energy for the printed Hamiltonian size.
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
from tasks.exp_12_hamiltonian_energy_mom import Exp12Config, run_exp12, summarize

BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_12"

BASE_ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
}
_Q_COLORS = ["tab:red", "tab:pink", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
_Q_MARKERS = ["D", "P", "X", "v", "*", "h"]


def build_ensembles(q_values: list[float]) -> tuple[dict, dict]:
    """Returns (ensembles, ensemble_style). One SEEQSTBinomialEnsemble per
    entry of ``q_values`` (freely chosen -- no q=m/n rule here, see module
    docstring), named ``seeqst_binomial_q<q>`` with q formatted to 4
    significant figures so distinct-but-close q's stay distinguishable."""
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


def plot_energy_error(summary, ensemble_style: dict, e_true: float, num_terms: int, out_dir: Path) -> None:
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
    ax.set_title(f"Ground-state energy error vs. $N_{{sample}}$  (E_true={e_true:.6g}, {num_terms} Pauli terms)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_error_vs_nsample.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=6, help="system size (number of qubits/spin-orbitals)")
    parser.add_argument(
        "--q-values", type=float, nargs="+", default=[4.0 / 6.0, 2.0 / 6.0],
        help="one SEEQSTBinomialEnsemble per value (default: 4/6 and 2/6, per the "
        "original request) -- freely chosen, no q=m/n rule computed here (see "
        "module docstring for why)",
    )
    parser.add_argument(
        "--integrals", type=str, default="demo", choices=["demo"],
        help="which numeric one/two-electron integrals to build H from -- "
        "currently only 'demo' (jw_hamiltonian.demo_integrals), a simple "
        "hand-picked (not a real molecule) instance",
    )
    parser.add_argument(
        "--estimator", type=str, default="mean", choices=["mean", "median_of_means"],
    )
    parser.add_argument(
        "--mom-num-groups", type=int, default=None,
        help="K for --estimator=median_of_means (default: HKP's own "
        "K=2*ceil(ln(2/mom_delta)) -- num_observables=1 here, unlike exp_9-exp_11)",
    )
    parser.add_argument("--mom-delta", type=float, default=0.1)
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
    )
    parser.add_argument(
        "--num-repeats", type=int, default=20,
        help="independent draws of the shadow snapshot stream (state and "
        "observable are both FIXED here -- see module docstring -- so this "
        "replaces exp_9-exp_11's num_state_repeats * num_observable_repeats "
        "as the only source of error-bar statistics; needs to be larger "
        "than those individually were)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.q_values = [0.5, 0.25]
        args.n_samples = [10, 50, 200, 800]
        args.num_repeats = 3

    if len(args.q_values) == 0:
        parser.error("--q-values needs at least one value")
    for q in args.q_values:
        if not (0.0 < q < 1.0):
            parser.error(f"--q-values entries must be strictly between 0 and 1 (got {q})")

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

    h, v = jw_hamiltonian.demo_integrals(args.n)
    ensembles, ensemble_style = build_ensembles(args.q_values)

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp12Config(
        n=args.n,
        h=h,
        v=v,
        n_samples=n_samples_sorted,
        num_repeats=args.num_repeats,
        ensembles=ensembles,
        estimator=args.estimator,
        mom_num_groups=mom_num_groups,
        seed=args.seed,
    )

    print(
        f"Running exp_12 -> {out_dir}\n"
        f"  n={args.n}, integrals={args.integrals!r}, q_values={args.q_values}\n"
        f"  estimator={args.estimator!r}"
        + (f", mom_num_groups={mom_num_groups}, mom_delta={args.mom_delta}" if args.estimator == "median_of_means" else "")
        + f"\n  n_samples={n_samples_sorted}, num_repeats={args.num_repeats}, seed={args.seed}"
    )
    t0 = time.time()
    raw, e_true, num_terms = run_exp12(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows, E_true={e_true:.10g}, {num_terms} Pauli terms")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "integrals": args.integrals,
        "q_values": args.q_values,
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

    plot_energy_error(summary, ensemble_style, e_true, num_terms, out_dir)
    print(f"Wrote energy_error_vs_nsample.png to {out_dir}")


if __name__ == "__main__":
    main()
