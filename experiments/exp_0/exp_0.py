#!/usr/bin/env python3
"""
================================================================================
 exp_0 -- Pauli vs. Clifford vs. SEEQST on random k-local observables
          (closed-form theory-bound comparison)
================================================================================

The original benchmark script for this project (predates exp_1's empirical,
multi-checkpoint design and the versioned `results/exp_N/v<K>/` output
convention -- see exp_1.md for how that later design differs). Still fully
functional and still the only script here that plots against the paper's
closed-form theoretical variance bounds directly, rather than purely
empirically -- kept as exp_0, not superseded outright.

Compares Pauli / Clifford / SEEQST classical shadows on the "predict many
random k-local Pauli observables" task, framed as in Huang, Kueng &
Preskill's SI (see ``tasks/exp_0_pauli_observable_prediction.py`` for the
precise framing and references).

Usage:
    python experiments/exp_0/exp_0.py --quick
    python experiments/exp_0/exp_0.py --n-values 2 3 4 5 6 7 8 9 10 \
        --num-observables 15 --num-shots 1500

Outputs (written to results/exp_0/):
    pauli_observable_raw.csv       - one row per (state, n, k, ensemble, observable)
    pauli_observable_summary.csv   - aggregated over observables
    variance_vs_k_<state>.png      - variance vs. locality k, one line per ensemble, per n
    variance_vs_n_<state>.png      - variance vs. system size n, one line per ensemble, at fixed k
    sample_complexity_vs_n_<state>.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# This script lives at shadow_benchmark/experiments/exp_0/exp_0.py -- two
# levels below the shadow_benchmark root (common/, ensembles/, tasks/,
# results/ all live there). Python only auto-adds the SCRIPT's own directory
# to sys.path, so without this the "from ensembles..." etc. imports below,
# and "python exp_0.py" run directly from this folder, would fail with
# ModuleNotFoundError. parents[2] = .../experiments/exp_0/exp_0.py ->
# parents[0]=exp_0/, parents[1]=experiments/, parents[2]=shadow_benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from tasks.exp_0_pauli_observable_prediction import BenchmarkConfig, run_benchmark, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_0/results/) so this script keeps writing to the same
# place -- results/exp_0/pauli_observable_*.csv -- regardless of which
# subfolder the script itself has been moved into. Unlike exp_1/exp_2/exp_4,
# this script predates the versioned v<N>/ output convention and just
# overwrites these flat files on every run (see common/versioning.py for the
# newer convention used by the other experiments).
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_0"


def default_k_values_by_n(n_values: list[int]) -> dict[int, list[int]]:
    """Weight-1, weight-2, weight-3 (local) and full-weight-n (global)
    observables for every system size -- enough to see both the
    locality-dependent (Pauli) and dimension-dependent (Clifford, SEEQST)
    trends without an O(n) sweep of every possible k."""
    out = {}
    for n in n_values:
        ks = sorted({k for k in (1, 2, 3, n) if 1 <= k <= n})
        out[n] = ks
    return out


def build_ensembles() -> dict:
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst": SEEQSTEnsemble(),
    }


def plot_variance_vs_k(summary, state_type: str, out_dir: Path) -> None:
    sub = summary[summary.state_type == state_type]
    n_values = sorted(sub.n.unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 4), sharey=True)
    if len(n_values) == 1:
        axes = [axes]
    for ax, n in zip(axes, n_values):
        s = sub[sub.n == n]
        for ens, marker in [("pauli", "o"), ("clifford", "s"), ("seeqst", "^")]:
            se = s[s.ensemble == ens].sort_values("k")
            if se.empty:
                continue
            ax.plot(se.k, se.mean_empirical_var, marker=marker, label=f"{ens} (empirical)")
            if se.theory_var_bound.notna().any():
                ax.plot(
                    se.k,
                    se.theory_var_bound,
                    linestyle="--",
                    color=ax.lines[-1].get_color(),
                    alpha=0.6,
                    label=f"{ens} (theory bound)",
                )
        ax.set_yscale("log")
        ax.set_xlabel("Pauli weight k")
        ax.set_title(f"n = {n}")
        if ax is axes[0]:
            ax.set_ylabel("mean empirical Var[o_hat]  (log scale)")
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Variance vs. Pauli locality k  --  state = {state_type}")
    fig.tight_layout()
    fig.savefig(out_dir / f"variance_vs_k_{state_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vs_n(summary, state_type: str, out_dir: Path, k_fixed: int = 2) -> None:
    sub = summary[(summary.state_type == state_type) & (summary.k == k_fixed)]
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ens, marker in [("pauli", "o"), ("clifford", "s"), ("seeqst", "^")]:
        se = sub[sub.ensemble == ens].sort_values("n")
        if se.empty:
            continue
        ax1.plot(se.n, se.mean_empirical_var, marker=marker, label=ens)
        ax2.plot(se.n, se.mean_sample_complexity, marker=marker, label=ens)
    for ax, ylabel in [(ax1, "mean empirical Var[o_hat]"), (ax2, "mean N_eps = Var/eps^2")]:
        ax.set_yscale("log")
        ax.set_xlabel("number of qubits n")
        ax.set_ylabel(ylabel + "  (log scale)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Scaling with system size n at fixed locality k={k_fixed}  --  state = {state_type}")
    fig.tight_layout()
    fig.savefig(out_dir / f"variance_and_complexity_vs_n_{state_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--num-observables", type=int, default=15)
    parser.add_argument("--num-shots", type=int, default=1500)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--states",
        nargs="+",
        default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small/fast configuration for smoke-testing the pipeline.",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_values = [2, 3, 4]
        args.num_observables = 6
        args.num_shots = 300

    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    k_values_by_n = default_k_values_by_n(args.n_values)

    config = BenchmarkConfig(
        n_values=args.n_values,
        k_values_by_n=k_values_by_n,
        ensembles=build_ensembles(),
        state_types=args.states,
        num_observables=args.num_observables,
        num_shots=args.num_shots,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    print(f"Running exp_0 -> {BASE_RESULTS_DIR}\n"
          f"  n={args.n_values}, k_by_n={k_values_by_n}, "
          f"states={args.states}, M={args.num_observables}, shots={args.num_shots}")
    t0 = time.time()
    raw = run_benchmark(config)
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)

    raw.to_csv(BASE_RESULTS_DIR / "pauli_observable_raw.csv", index=False)
    summary.to_csv(BASE_RESULTS_DIR / "pauli_observable_summary.csv", index=False)
    print(f"Wrote {BASE_RESULTS_DIR / 'pauli_observable_raw.csv'}")
    print(f"Wrote {BASE_RESULTS_DIR / 'pauli_observable_summary.csv'}")

    for state_type in args.states:
        plot_variance_vs_k(summary, state_type, BASE_RESULTS_DIR)
        k_fixed = 2 if 2 in k_values_by_n.get(args.n_values[0], []) else k_values_by_n[args.n_values[0]][0]
        plot_vs_n(summary, state_type, BASE_RESULTS_DIR, k_fixed=k_fixed)
    print(f"Wrote plots to {BASE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
