"""Task: predicting many random k-local Pauli observables from classical
shadows, framed exactly as in Huang, Kueng & Preskill's SI (Sec. 3,
"Example 1" / "Example 2", and Theorem S1).

For a fixed state rho, a fixed locality k and a set of M independently
drawn random weight-k Pauli observables {P_1, ..., P_M}, we:

  1. Draw `num_shots` independent single-shot classical-shadow estimates
     o_hat_i = tr(P_i rho_hat) for every observable, for each ensemble.
  2. Compute the empirical variance of o_hat_i across shots (an empirical
     estimate of the shadow-norm-squared quantity that controls sample
     complexity in Lemma S1 / Theorem S1).
  3. Compare against the closed-form theoretical shadow-norm bounds where
     known:
       - local Pauli ensemble:   ||P||_shadow^2 = 3^k            (SI Eq. S17,
         exact for tensor-product observables)
       - global Clifford ensemble: ||P||_shadow^2 = 3 * tr(P^2) = 3 * 2^n
         (SI Eq. S16; note the explicit, unfavourable dependence on the full
         system dimension 2^n, regardless of how local P is)
       - SEEQST ensemble: no closed-form bound is derived in the draft, so
         we report the empirical value only.
  4. Convert variance into an (empirical) sample complexity
     N_eps = Var / eps^2, the number of single shadows needed for a
     mean-estimator to reach additive accuracy eps (Theorem S1, ignoring
     the log(M/delta) median-of-means overhead which is common to every
     ensemble and does not affect the comparison between ensembles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli, Statevector

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class BenchmarkConfig:
    n_values: list[int]
    k_values_by_n: dict[int, list[int]]
    ensembles: dict[str, ShadowEnsemble]
    state_types: list[str]
    num_observables: int = 15
    num_shots: int = 1500
    epsilon: float = 0.1
    seed: int = 0


def theoretical_variance_bound(ensemble_name: str, k: int, n: int) -> float:
    """Known closed-form ||P||_shadow^2 bounds; NaN where none is derived."""
    if ensemble_name == "pauli":
        return 3.0**k
    if ensemble_name == "clifford":
        return 3.0 * (2.0**n)
    return float("nan")


def run_benchmark_chunk(
    config: BenchmarkConfig,
    state_type: str,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run every k in config.k_values_by_n[n] and every ensemble, for a
    single (state_type, n) pair. Factored out of ``run_benchmark`` so a full
    grid can be computed incrementally / resumed in separate processes if
    needed (e.g. under a wall-clock-limited sandbox); the CLI entry point
    normally just calls ``run_benchmark`` directly for the whole grid."""
    rows: list[dict] = []
    state = states.STATE_BUILDERS[state_type](n, rng)
    for k in config.k_values_by_n.get(n, []):
        specs = [pauli_utils.random_pauli_spec(n, k, rng) for _ in range(config.num_observables)]
        true_vals = [
            state.expectation_value(Pauli(pauli_utils.spec_to_label(s, n))).real for s in specs
        ]
        for ens_name, ensemble in config.ensembles.items():
            ests = ensemble.estimate_many_shots(
                state, specs, n, config.num_shots, rng
            )  # (num_shots, num_observables)
            emp_mean = ests.mean(axis=0)
            emp_var = ests.var(axis=0, ddof=1)
            theory_bound = theoretical_variance_bound(ens_name, k, n)
            for i, spec in enumerate(specs):
                rows.append(
                    {
                        "state_type": state_type,
                        "n": n,
                        "k": k,
                        "ensemble": ens_name,
                        "pauli_index": i,
                        "pauli_label": pauli_utils.spec_to_label(spec, n),
                        "true_value": true_vals[i],
                        "empirical_mean": emp_mean[i],
                        "abs_bias": abs(emp_mean[i] - true_vals[i]),
                        "empirical_var": emp_var[i],
                        "theory_var_bound": theory_bound,
                        "sample_complexity_eps": emp_var[i] / config.epsilon**2,
                        "num_shots": config.num_shots,
                    }
                )
    return pd.DataFrame(rows)


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    chunks = [
        run_benchmark_chunk(config, state_type, n, rng)
        for state_type in config.state_types
        for n in config.n_values
    ]
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over the M random observables (mean + max empirical
    variance / sample complexity per (state_type, n, k, ensemble))."""
    agg = (
        df.groupby(["state_type", "n", "k", "ensemble"])
        .agg(
            mean_empirical_var=("empirical_var", "mean"),
            max_empirical_var=("empirical_var", "max"),
            theory_var_bound=("theory_var_bound", "first"),
            mean_sample_complexity=("sample_complexity_eps", "mean"),
            max_sample_complexity=("sample_complexity_eps", "max"),
            mean_abs_bias=("abs_bias", "mean"),
            num_observables=("pauli_index", "count"),
        )
        .reset_index()
    )
    return agg
