"""Core logic for exp_10 (see ``experiments/exp_10/exp_10.py`` for the full
explanation, CLI, and plotting).

exp_10 is exp_9 with exactly one thing swapped: the observable family.
Everything else -- the four-ensemble roster, the selectable point
estimator, the two-layer randomization design, snapshot-stream reuse, and
the RMSE/MaxError metrics -- is unchanged from exp_9 (see
``tasks/exp_9_tuned_locality_mom.py`` for the full rationale on all of
that, carried over verbatim here).

  * Observable family: exp_9 draws exp_7's "exact-m" family (X/Y-weight
    EXACTLY ``m``, via ``pauli_utils.random_exact_xy_spec``). exp_10 draws
    exp_8's "sparse-m" (ceiling) family instead: X/Y-weight drawn UNIFORMLY
    from ``{1, ..., m}`` -- never above ``m`` -- with every other qubit set
    to Z, via ``pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)``.
    This is the operational, list-of-separately-tracked-observables
    counterpart of the **m-sparse operators**
    ``notes/working/seeqst_sparse_tuning.tex`` studies (``S_m = {s subseteq
    [n] : |s| <= m}``, the set of X/Y-support patterns allowed to appear) --
    see ``experiments/exp_8/exp_8.md`` for the precise (and important)
    argument for why ``q=m/n`` tuning is minimax-optimal for this family
    only when ``m <= n/2``, which carries over unchanged to exp_10 since
    the observable generator and SEEQST tuning are identical to exp_8's.
    Unlike exp_9's "exact" family, this generator has no ``rest_mode``
    knob: the non-XY qubits are unconditionally Z (``random_bounded_xy_spec``
    always Z-pads), so ``Exp10Config`` has no ``rest_mode`` field.

  * Ensembles: the SAME four exp_9 compares -- ``pauli``, ``clifford``,
    ``seeqst_uniform``, and ``seeqst_binomial_tuned`` (``q=m/n``, tuned to
    the ceiling ``m`` exactly as exp_9 tunes it to the exact locality
    ``m`` -- see ``experiments/exp_10/exp_10.py::build_ensembles``, which is
    byte-for-byte exp_9's). exp_8's extra ``seeqst_binomial_untuned`` /
    ``seeqst_unifsize_tuned`` ensembles and its ``m>n/2`` runtime warning
    are deliberately NOT carried over -- exp_10 stays a minimal, direct
    "exp_9 with one swapped observable family" diff.

  * Point estimator: exp_9's selectable ``Exp10Config.estimator`` --
    "mean" (running empirical mean) or "median_of_means" (HKP's own
    estimator) -- carried over unchanged, including the ``K``-clamping
    behavior for small ``n_sample`` checkpoints. See
    ``tasks/exp_9_tuned_locality_mom.py`` for the full estimator rationale.

Both estimators are computed from the SAME underlying stream of
``max(n_samples)`` classical-shadow snapshots per (state, ensemble) -- the
two-layer randomization design, snapshot-stream reuse across the inner
observable-set draws, and the RMSE/MaxError metrics are IDENTICAL to exp_9
(and, through it, exp_7/exp_8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp10Config:
    n: int
    m: int  # ceiling on X/Y-weight: weight ~ Uniform{1, ..., m} (exp_8's "sparse" family)
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
        if self.estimator not in ("mean", "median_of_means"):
            raise ValueError(
                f"estimator={self.estimator!r} must be 'mean' or 'median_of_means'"
            )
        if self.mom_num_groups < 1:
            raise ValueError(f"mom_num_groups={self.mom_num_groups} must be >= 1")


def _draw_observables(config: Exp10Config, n: int, rng: np.random.Generator) -> list[dict]:
    # weight ~ Uniform{1, ..., m}, Z-padded to full weight n -- exp_8's
    # ceiling/"sparse" generator (the SAME one exp_6 uses for its own
    # untuned family), swapped in for exp_9's exact-m generator. No
    # rest_mode knob: random_bounded_xy_spec always Z-pads.
    return [
        pauli_utils.random_bounded_xy_spec(n, config.m, rng, min_xy=1)
        for _ in range(config.num_observables)
    ]


def _checkpoint_estimates(
    shots: np.ndarray, n_samples_sorted: list[int], estimator: str, mom_num_groups: int
) -> dict[int, np.ndarray]:
    """``shots`` has shape (max_n_sample, M) -- one row per drawn snapshot,
    one column per observable in the current inner draw. Returns
    ``{n_sample: (M,) point estimate using only the first n_sample rows}``
    for every requested checkpoint. Identical to exp_9's version -- see
    ``tasks/exp_9_tuned_locality_mom.py`` for the full rationale.
    """
    out: dict[int, np.ndarray] = {}
    if estimator == "mean":
        max_n_sample = shots.shape[0]
        cumulative_mean = np.cumsum(shots, axis=0) / np.arange(
            1, max_n_sample + 1
        ).reshape(-1, 1)
        for n_sample in n_samples_sorted:
            out[n_sample] = cumulative_mean[n_sample - 1]
    else:  # "median_of_means"
        for n_sample in n_samples_sorted:
            k_eff = max(1, min(mom_num_groups, n_sample))
            batch_size = n_sample // k_eff
            usable = batch_size * k_eff  # drop any remainder past the last full group
            batches = shots[:usable].reshape(k_eff, batch_size, -1)
            batch_means = batches.mean(axis=1)  # (k_eff, M)
            out[n_sample] = np.median(batch_means, axis=0)  # (M,)
    return out


def run_exp10(config: Exp10Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see ``tasks/exp_9_tuned_locality_mom.py::run_exp9``
    for the two-layer randomization design (outer: random state, inner:
    random observable set for that same state) and the per-(state,
    ensemble) snapshot-stream reuse, both carried over unchanged. The only
    change from exp_9 is WHICH observable family ``_draw_observables`` above
    draws from.

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
                    f"m={config.m} (sparse ceiling) "
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
