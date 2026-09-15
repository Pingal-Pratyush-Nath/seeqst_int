"""Core logic for exp_9 (see ``experiments/exp_9/exp_9.py`` for the full
explanation, CLI, and plotting).

exp_9 is exp_7 trimmed down to exactly the comparison the project currently
cares about, plus one new capability:

  * Observable family: ONLY exp_7's "exact" family (X/Y-weight exactly
    ``m``, via ``pauli_utils.random_exact_xy_spec``), with ONLY the two
    ``rest_mode`` values that matter for showing SEEQST's b-independence
    against Pauli's b-dependence -- "random" (each non-XY qubit iid Z or I)
    and "z" (all-Z tail, i.e. full weight n). exp_7's "at_least" family and
    its "identity" rest_mode are dropped entirely. ``seeqst_unifsize_tuned``
    and ``seeqst_binomial_untuned`` (both were exp_7 apparatus specific to
    the dropped "at_least"/floor-robustness study) are dropped along with
    it.

  * Ensembles: only the four the project is actively comparing --
    ``pauli``, ``clifford``, ``seeqst_uniform``, and
    ``seeqst_binomial_tuned`` (q=m/n). See
    ``experiments/exp_9/exp_9.py::build_ensembles``.

  * NEW: a selectable point estimator, ``Exp9Config.estimator`` --
    "mean" (exp_7's running empirical mean, unchanged) or
    "median_of_means" (HKP's own estimator -- see
    ``notes/working/seeqst_sample_complexity.tex``, "sample complexity via
    classical shadows": split the first N_sample single-shot estimates into
    ``K`` equal-size groups, average within each group, then take the
    per-observable MEDIAN across the ``K`` group-means). ``K`` is fixed for
    the whole run (``Exp9Config.mom_num_groups``) -- exactly how HKP's own
    theorem uses it (K depends only on M and delta, not on N_sample or
    epsilon) -- and is clamped down to ``n_sample`` itself (with the
    remainder past the last full group of size ``n_sample // K`` dropped)
    for any checkpoint smaller than the requested K, so a run never crashes
    on its own smallest ``--n-samples`` checkpoint; see
    ``experiments/exp_9/exp_9.py`` for the default
    ``K = 2*ceil(ln(2M/delta))``.

Both estimators are computed from the SAME underlying stream of
``max(n_samples)`` classical-shadow snapshots per (state, ensemble) -- the
two-layer randomization design, snapshot-stream reuse across the inner
observable-set draws, and the RMSE/MaxError metrics are otherwise IDENTICAL
to exp_7 (see ``tasks/exp_7_tuned_locality.py`` for the full rationale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp9Config:
    n: int
    m: int
    rest_mode: str  # "random" | "z"
    num_observables: int
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    estimator: str = "mean"  # "mean" | "median_of_means"
    mom_num_groups: int = 1  # K; only meaningful when estimator == "median_of_means"
    state_types: list[str] = field(
        default_factory=lambda: ["haar_random", "ghz", "random_stabilizer"]
    )
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rest_mode not in ("random", "z"):
            raise ValueError(f"rest_mode={self.rest_mode!r} must be 'random' or 'z'")
        if self.estimator not in ("mean", "median_of_means"):
            raise ValueError(
                f"estimator={self.estimator!r} must be 'mean' or 'median_of_means'"
            )
        if self.mom_num_groups < 1:
            raise ValueError(f"mom_num_groups={self.mom_num_groups} must be >= 1")


def _draw_observables(config: Exp9Config, n: int, rng: np.random.Generator) -> list[dict]:
    return [
        pauli_utils.random_exact_xy_spec(n, config.m, rng, rest_mode=config.rest_mode)
        for _ in range(config.num_observables)
    ]


def _checkpoint_estimates(
    shots: np.ndarray, n_samples_sorted: list[int], estimator: str, mom_num_groups: int
) -> dict[int, np.ndarray]:
    """``shots`` has shape (max_n_sample, M) -- one row per drawn snapshot,
    one column per observable in the current inner draw. Returns
    ``{n_sample: (M,) point estimate using only the first n_sample rows}``
    for every requested checkpoint.
    """
    out: dict[int, np.ndarray] = {}
    if estimator == "mean":
        # Single cumsum covers every checkpoint at once (exp_7's trick,
        # unchanged) -- O(max_n_sample) instead of redoing the mean per
        # checkpoint.
        max_n_sample = shots.shape[0]
        cumulative_mean = np.cumsum(shots, axis=0) / np.arange(
            1, max_n_sample + 1
        ).reshape(-1, 1)
        for n_sample in n_samples_sorted:
            out[n_sample] = cumulative_mean[n_sample - 1]
    else:  # "median_of_means"
        # K is fixed for the whole run, but each checkpoint reshapes its OWN
        # first n_sample shots into K groups (bigger groups as n_sample
        # grows), so this can't reuse a single running structure the way
        # the mean can -- cheap regardless, this reshape+mean+median is
        # negligible next to the snapshot sampling itself.
        for n_sample in n_samples_sorted:
            k_eff = max(1, min(mom_num_groups, n_sample))
            batch_size = n_sample // k_eff
            usable = batch_size * k_eff  # drop any remainder past the last full group
            batches = shots[:usable].reshape(k_eff, batch_size, -1)
            batch_means = batches.mean(axis=1)  # (k_eff, M)
            out[n_sample] = np.median(batch_means, axis=0)  # (M,)
    return out


def run_exp9(config: Exp9Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see ``tasks/exp_7_tuned_locality.py::run_exp7`` for
    the two-layer randomization design (outer: random state, inner: random
    observable set for that same state) and the per-(state, ensemble)
    snapshot-stream reuse, both carried over unchanged. The only change from
    exp_7 is HOW a checkpoint's point estimate is built from that shared
    stream of shots -- see ``_checkpoint_estimates`` above.

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
                    f"m={config.m} rest_mode={config.rest_mode!r} "
                    f"estimator={config.estimator!r} -> drawing {max_n_sample} shadow "
                    f"snapshots per ensemble, then {config.num_observable_repeats} random "
                    f"observable set(s) to evaluate against them"
                )

            # Draw the snapshot stream ONCE per (state, ensemble) --
            # observable-independent -- and reuse it for every inner
            # observable-set draw below.
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
                    estimates = _checkpoint_estimates(
                        shots, n_samples_sorted, config.estimator, config.mom_num_groups
                    )

                    for n_sample in n_samples_sorted:
                        err = estimates[n_sample] - true_vals

                        rows.append(
                            {
                                "state_repeat": state_repeat,
                                "obs_repeat": obs_repeat,
                                "state_type": state_type,
                                "ensemble": ens_name,
                                "n": config.n,
                                "m": config.m,
                                "rest_mode": config.rest_mode,
                                "num_observables": config.num_observables,
                                "estimator": config.estimator,
                                "mom_num_groups": config.mom_num_groups,
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
