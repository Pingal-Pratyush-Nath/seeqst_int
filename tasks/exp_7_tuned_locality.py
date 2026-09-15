"""Core logic for exp_7 (see ``experiments/exp_7/exp_7.py`` for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting.

exp_7 mirrors exp_6's two-layer randomization design and running-mean
estimator exactly (see ``tasks/exp_6_bounded_xy_weight.py`` / exp_1's module
docstring for the full rationale), but differs in what "the swept
hyperparameter" is and how observables are drawn:

  exp_6 swept ``l``, a CAP on X/Y-weight (observable weight drawn uniformly
  from {1, ..., l}, always Z-padded to full weight n), and used ONE fixed
  SEEQST-binomial q=1/(n+1) throughout.

  exp_7 instead sweeps ``m``, the locality that
  ``notes/main_theorem/main_theorem.tex``'s Theorem 1 / Corollary 5 are
  stated for, and TUNES the SEEQST family to it (q = m/n for the binomial
  ensemble, l = m for the uniform-subset-size ensemble -- see
  ``experiments/exp_7/exp_7.py::build_ensembles``). Two observable families
  are available, chosen via ``Exp7Config.observable_type``:

    "exact"     -- X/Y-weight is EXACTLY m always, via
                   ``pauli_utils.random_exact_xy_spec(n, m, rng, rest_mode)``.
                   This is the theorem's own regime.
    "at_least"  -- X/Y-weight is AT LEAST m (drawn uniformly from
                   {m, ..., l}, l an upper cap, Z-padded), via
                   ``pauli_utils.random_bounded_xy_spec(n, l, rng,
                   min_xy=m)`` -- no new generator needed, that function
                   already supports a floor. The SEEQST family stays tuned
                   to the FLOOR m (not the true random weight), by design:
                   this is a controlled "we only know a lower bound"
                   robustness study, one notch more forgiving than exp_6's
                   fully-uninformative-range design.

Only ``rmse`` and ``max_error`` are computed per row, same as exp_6 and for
the same reason (exp_1's ``mean_variance`` is dropped, and the underlying
per-shot variance computation is skipped entirely rather than just omitted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp7Config:
    n: int
    m: int
    observable_type: str  # "exact" or "at_least"
    rest_mode: str  # "random" | "z" | "identity" -- only meaningful when observable_type == "exact"
    l: int  # upper cap on xy-weight -- only meaningful when observable_type == "at_least"
    num_observables: int
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    state_types: list[str] = field(
        default_factory=lambda: ["haar_random", "ghz", "random_stabilizer"]
    )
    seed: int = 0


def _draw_observables(config: Exp7Config, n: int, rng: np.random.Generator) -> list[dict]:
    if config.observable_type == "exact":
        return [
            pauli_utils.random_exact_xy_spec(n, config.m, rng, rest_mode=config.rest_mode)
            for _ in range(config.num_observables)
        ]
    elif config.observable_type == "at_least":
        return [
            pauli_utils.random_bounded_xy_spec(n, config.l, rng, min_xy=config.m)
            for _ in range(config.num_observables)
        ]
    else:
        raise ValueError(
            f"observable_type={config.observable_type!r} must be 'exact' or 'at_least'"
        )


def run_exp7(config: Exp7Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see ``tasks/exp_6_bounded_xy_weight.py::run_exp6``
    for the two-layer randomization design (outer: random state, inner:
    random observable set for that same state) and the running-mean
    checkpoint construction, both reused here unchanged.

    IMPORTANT: for a given (state, ensemble), the underlying stream of
    max(n_samples) classical-shadow snapshots is drawn ONCE (right after the
    state itself, before the inner observable-set loop) and REUSED across
    every inner observable-set draw for that state -- exactly as in exp_1
    and exp_6.

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
                    f"m={config.m} observable_type={config.observable_type!r} -> drawing "
                    f"{max_n_sample} shadow snapshots per ensemble, then "
                    f"{config.num_observable_repeats} random observable set(s) to evaluate "
                    f"against them"
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
                                "observable_type": config.observable_type,
                                "rest_mode": config.rest_mode,
                                "l": config.l,
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
