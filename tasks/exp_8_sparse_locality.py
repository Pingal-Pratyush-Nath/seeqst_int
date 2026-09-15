"""Core logic for exp_8 (see ``experiments/exp_8/exp_8.py`` for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting.

exp_8 mirrors exp_6/exp_7's two-layer randomization design and running-mean
estimator exactly (see ``tasks/exp_6_bounded_xy_weight.py`` / exp_1's module
docstring for the full rationale), but differs in what the swept
hyperparameter ``m`` means and how observables are drawn:

  exp_6 swept ``l``, a cap on X/Y-weight, and used ONE fixed SEEQST-binomial
  q=1/(n+1) throughout (no tuning at all).

  exp_7 swept ``m``, the EXACT (or, under "at_least", the FLOOR of the)
  X/Y-locality that ``notes/main_theorem/main_theorem.tex``'s Theorem 1 is
  stated for, and tuned q=m/n to it.

  exp_8 sweeps ``m`` as a CEILING: X/Y-weight is drawn uniformly from
  {1, ..., m} (never above m), via
  ``pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)`` -- the same
  generator exp_6 already uses for its own (unconditionally-capped, always
  q=1/(n+1)) observable family, just reused here with the SEEQST family
  TUNED to the cap. This is exp_7's "at_least" family turned upside down:
  there the floor was known and q was tuned to it; here the ceiling is
  known and q is tuned to it. It is also the operational, list-of-separate-
  observables analogue of the m-sparse operators
  ``notes/working/seeqst_sparse_tuning.tex`` studies (S_m = {s : |s| <= m}
  the set of allowed X/Y-support patterns) -- see exp_8.md for the precise
  (and, importantly, DIFFERENT-from-that-note's) argument for why q=m/n is
  still the right tuning here.

Only ``rmse`` and ``max_error`` are computed per row, same as exp_6/exp_7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp8Config:
    n: int
    m: int  # ceiling on X/Y-weight: weight ~ Uniform{1, ..., m}
    num_observables: int
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    state_types: list[str] = field(
        default_factory=lambda: ["haar_random", "ghz", "random_stabilizer"]
    )
    seed: int = 0


def _draw_observables(config: Exp8Config, n: int, rng: np.random.Generator) -> list[dict]:
    # weight ~ Uniform{1, ..., m}, Z-padded to full weight n -- the exact
    # same generator exp_6 uses for its own (untuned) family, and the same
    # one exp_7's "at_least" mode uses with a floor instead of a ceiling
    # (min_xy=config.m there vs. min_xy=1, l=config.m here).
    return [
        pauli_utils.random_bounded_xy_spec(n, config.m, rng, min_xy=1)
        for _ in range(config.num_observables)
    ]


def run_exp8(config: Exp8Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see ``tasks/exp_6_bounded_xy_weight.py::run_exp6`` /
    ``tasks/exp_7_tuned_locality.py::run_exp7`` for the two-layer
    randomization design (outer: random state, inner: random observable set
    for that same state) and the running-mean checkpoint construction, both
    reused here unchanged.

    IMPORTANT: for a given (state, ensemble), the underlying stream of
    max(n_samples) classical-shadow snapshots is drawn ONCE (right after the
    state itself, before the inner observable-set loop) and REUSED across
    every inner observable-set draw for that state -- exactly as in exp_1,
    exp_6, exp_7.

    If ``verbose``, prints one line to the terminal every time the OUTER
    (state) randomization layer draws a new state.
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
                    f"m={config.m} -> drawing {max_n_sample} shadow snapshots per ensemble, "
                    f"then {config.num_observable_repeats} random observable set(s) to "
                    f"evaluate against them"
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

                        rows.append(
                            {
                                "state_repeat": state_repeat,
                                "obs_repeat": obs_repeat,
                                "state_type": state_type,
                                "ensemble": ens_name,
                                "n": config.n,
                                "m": config.m,
                                "num_observables": config.num_observables,
                                "n_sample": n_sample,
                                "rmse": float(np.sqrt(np.mean(err**2))),
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
            max_error_mean=("max_error", "mean"),
            max_error_std=("max_error", "std"),
            num_state_repeats=("state_repeat", "nunique"),
            num_observable_repeats=("obs_repeat", "nunique"),
        )
        .reset_index()
    )
    return agg
