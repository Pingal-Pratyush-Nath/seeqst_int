"""Core logic for exp_1 (see ``exp_1.py`` at the repo root for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp1Config:
    n: int
    num_observables: int
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    state_types: list[str] = field(
        default_factory=lambda: ["haar_random", "ghz", "random_stabilizer"]
    )
    k: int | None = None  # None = fully random (any) Pauli string; else fixed weight k
    seed: int = 0


def _draw_observables(config: Exp1Config, n: int, rng: np.random.Generator) -> list[dict]:
    if config.k is None:
        return [pauli_utils.random_full_pauli_spec(n, rng) for _ in range(config.num_observables)]
    return [pauli_utils.random_pauli_spec(n, config.k, rng) for _ in range(config.num_observables)]


def run_exp1(config: Exp1Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see exp_1.py's module docstring for the two-layer
    randomization design (outer: random state, inner: random observable set
    for that same state).

    IMPORTANT: for a given (state, ensemble), the underlying stream of
    max(n_samples) classical-shadow snapshots is drawn ONCE (right after the
    state itself, before the inner observable-set loop) and REUSED across
    every inner observable-set draw for that state. This matches the actual
    point of classical shadows -- a single batch of measurements on a state
    can be reused to predict any number of different observables -- and is
    NOT the same as re-measuring the state from scratch for every candidate
    observable set (which would defeat that purpose and was a bug in an
    earlier version of this script).

    If ``verbose``, prints one line to the terminal every time the OUTER
    (state) randomization layer draws a new state, so you can watch that
    layer progress as the run happens.
    """
    rng = np.random.default_rng(config.seed)
    n_samples_sorted = sorted(set(int(x) for x in config.n_samples))
    max_n_sample = n_samples_sorted[-1]

    total_state_draws = config.num_state_repeats * len(config.state_types)
    state_draw_idx = 0

    rows: list[dict] = []
    for state_repeat in range(config.num_state_repeats):
        for state_type in config.state_types:
            state = states.STATE_BUILDERS[state_type](config.n, rng)  # outer layer: fresh state
            state_draw_idx += 1
            if verbose:
                print(
                    f"  [outer layer | state draw {state_draw_idx}/{total_state_draws}] "
                    f"state_repeat={state_repeat} state_type={state_type!r} n={config.n} "
                    f"-> drawing {max_n_sample} shadow snapshots per ensemble, then "
                    f"{config.num_observable_repeats} random observable set(s) to evaluate against them"
                )

            # Draw the snapshot stream ONCE per (state, ensemble) -- observable-
            # independent -- and reuse it for every inner observable-set draw below.
            snapshots_by_ensemble = {
                ens_name: ensemble.sample_snapshots(state, config.n, max_n_sample, rng)
                for ens_name, ensemble in config.ensembles.items()
            }

            for obs_repeat in range(config.num_observable_repeats):
                specs = _draw_observables(config, config.n, rng)  # inner layer: fresh observable set
                true_vals = np.array(
                    [
                        state.expectation_value(Pauli(pauli_utils.spec_to_label(s, config.n))).real
                        for s in specs
                    ]
                )
                for ens_name, ensemble in config.ensembles.items():
                    shots = ensemble.evaluate_snapshots(
                        snapshots_by_ensemble[ens_name], specs, config.n
                    )  # shape (max_n_sample, M) -- reuses the SAME snapshots across obs_repeat
                    cumulative_mean = np.cumsum(shots, axis=0) / np.arange(
                        1, max_n_sample + 1
                    ).reshape(-1, 1)  # running mean at every shot count, shape (max_n_sample, M)

                    for n_sample in n_samples_sorted:
                        est = cumulative_mean[n_sample - 1]  # (M,) mean over first n_sample shots
                        err = est - true_vals

                        # Empirical variance of the SINGLE-SHOT estimator o_hat_i^(1),
                        # estimated from the first n_sample raw (pre-averaging) shots
                        # for each observable, then averaged over the M observables.
                        # See the "mean variance / Lemma S1" note in exp_1.py's
                        # module docstring for exactly what this is.
                        per_observable_var = np.var(shots[:n_sample, :], axis=0, ddof=1)  # (M,)

                        rows.append(
                            {
                                "state_repeat": state_repeat,
                                "obs_repeat": obs_repeat,
                                "state_type": state_type,
                                "ensemble": ens_name,
                                "n": config.n,
                                "num_observables": config.num_observables,
                                "n_sample": n_sample,
                                "rmse": float(np.sqrt(np.mean(err**2))),
                                "mean_variance": float(np.mean(per_observable_var)),
                                "max_error": float(np.max(np.abs(err))),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average (and report std-dev of) each metric across BOTH randomization
    layers (all state_repeat x obs_repeat trials pooled together), for every
    (state_type, ensemble, n_sample)."""
    agg = (
        df.groupby(["state_type", "ensemble", "n_sample"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mean_variance_mean=("mean_variance", "mean"),
            mean_variance_std=("mean_variance", "std"),
            max_error_mean=("max_error", "mean"),
            max_error_std=("max_error", "std"),
            num_state_repeats=("state_repeat", "nunique"),
            num_observable_repeats=("obs_repeat", "nunique"),
        )
        .reset_index()
    )
    return agg
