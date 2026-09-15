"""Core logic for exp_4 (see ``experiments/exp_4/exp_4.py`` for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting.

Compares two DIFFERENT derandomization strategies, head to head, on
identical drawn states and observable sets:

  - ``derandomized_seeqst``: our own SEEQST-specific derandomization
    (``experiments/Archived/exp_3/exp_3.py``'s exact-match circuit selection, plus
    exp_2's demand-proportional shot allocation) -- entangling GHZ-block
    circuits, observables grouped onto a small number of distinct circuits,
    shots drawn i.i.d. from a categorical distribution over those circuits.
  - ``derandomized_pauli``: Huang, Kueng & Preskill's derandomized classical
    shadows (arXiv:2103.07510), via
    ``codes/predicting-quantum-properties/data_acquisition_shadow.derandomized_classical_shadow``
    -- product (non-entangling) single-qubit measurements, with the ENTIRE
    measurement schedule computed once, deterministically, by their greedy
    algorithm (no per-shot randomness in the schedule itself; the state and
    the drawn observables are the only randomness upstream of it).

Neither arm has a "measure once, reuse for many observable sets" step (the
way exp_1 does): both circuit-selection strategies are observable-set
dependent by construction, so a fresh batch of measurements is drawn from
scratch for every inner obs_repeat draw, for both methods -- same reasoning
as exp_2's derandomized_seeqst arm, just now applying to both arms since
exp_4 has no non-derandomized baseline at all.

The SEEQST-derandomized-protocol orchestration logic below (grouping +
allocation + hit-conditional checkpointing) is intentionally DUPLICATED from
``tasks/exp_2_derandomized_scaling.py``'s ``run_exp2``, not imported/shared
-- same convention exp_2 already uses for ``_draw_observables`` (duplicated
from exp_1's, "kept identical on purpose" so each task module has no
dependency on another experiment's task module). This keeps exp_2
completely untouched and every exp_N task file independently readable end
to end. What IS genuinely shared (not duplicated) is
``experiments.Archived.exp_3.exp_3``'s ``select_subset_and_branch`` /
``circuit_for_subset_branch`` -- that module is built as a reusable utility,
unlike the task-specific orchestration functions (it lives under
``Archived/`` for folder organization only -- it is NOT deprecated, this is
a live dependency of both exp_2 and exp_4).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli, Statevector

from common import pauli_utils, states
from common.external_paths import import_huang_data_acquisition_shadow
from ensembles.pauli_ensemble import _basis_circuit
from experiments.Archived.exp_3.exp_3 import circuit_for_subset_branch, select_subset_and_branch

_data_acquisition_shadow = import_huang_data_acquisition_shadow()


@dataclass
class Exp4Config:
    n: int
    num_observables: int
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    state_types: list[str] = field(
        default_factory=lambda: ["haar_random", "ghz", "random_stabilizer"]
    )
    k: int | None = None  # None = fully random (any) Pauli string; else fixed weight k
    seed: int = 0
    include_seeqst: bool = True              # run the derandomized_seeqst arm
    include_derandomized_pauli: bool = True  # run the derandomized_pauli (Huang) arm


def _draw_observables(config: Exp4Config, n: int, rng: np.random.Generator) -> list[dict]:
    """Same observable-generating distribution as exp_1/exp_2's
    ``_draw_observables`` (duplicated, not imported -- see module
    docstring)."""
    if config.k is None:
        return [pauli_utils.random_full_pauli_spec(n, rng) for _ in range(config.num_observables)]
    return [pauli_utils.random_pauli_spec(n, config.k, rng) for _ in range(config.num_observables)]


def _checkpoint_rows(
    method: str,
    M: int,
    n_samples_sorted: list[int],
    true_vals: np.ndarray,
    raw_values_by_obs: list[list[float]],
    shot_index_by_obs: list[list[int]],
    extra: dict,
) -> list[dict]:
    """Shared checkpoint-evaluation step for both arms: given each
    observable's matching-shot history (values + 1-indexed shot positions),
    compute RMSE/MeanVariance/MaxError/Coverage/MeanMatchesPerObservable at
    every n_sample checkpoint. Zero-match observables are EXCLUDED from
    RMSE/MeanVariance/MaxError at that checkpoint (nothing to estimate with
    yet) -- ``coverage`` tracks how many that excludes. Same convention as
    exp_2 (see exp_2.md's "Metrics" section for the rationale)."""
    rows = []
    for n_sample in n_samples_sorted:
        errs = []
        variances = []
        match_counts = np.empty(M, dtype=int)
        for obs_i in range(M):
            n_i = bisect_right(shot_index_by_obs[obs_i], n_sample)
            match_counts[obs_i] = n_i
            if n_i == 0:
                continue
            vals_so_far = raw_values_by_obs[obs_i][:n_i]
            est = float(np.mean(vals_so_far))
            errs.append(est - true_vals[obs_i])
            if n_i >= 2:
                variances.append(float(np.var(vals_so_far, ddof=1)))

        coverage = float(np.mean(match_counts >= 1))
        errs_arr = np.array(errs) if errs else np.array([np.nan])
        rows.append(
            {
                "method": method,
                "n_sample": n_sample,
                "rmse": float(np.sqrt(np.mean(errs_arr**2))) if errs else float("nan"),
                "mean_variance": float(np.mean(variances)) if variances else float("nan"),
                "max_error": float(np.max(np.abs(errs_arr))) if errs else float("nan"),
                "coverage": coverage,
                "mean_matches_per_observable": float(np.mean(match_counts)),
                **extra,
            }
        )
    return rows


def _run_seeqst_arm(
    specs: list[dict],
    pauli_objs: list[Pauli],
    true_vals: np.ndarray,
    state: Statevector,
    n: int,
    n_samples_sorted: list[int],
    max_n_sample: int,
    rng: np.random.Generator,
) -> list[dict]:
    """The derandomized-SEEQST protocol (exp_2/exp_3), for a single (state,
    observable-set) trial -- see ``exp_2_derandomized_scaling.run_exp2`` and
    ``experiments/exp_2/exp_2.md`` for the full derivation (duplicated here,
    not imported -- see this module's docstring)."""
    M = len(specs)

    keys = [select_subset_and_branch(s, n) for s in specs]
    distinct_keys = list(dict.fromkeys(keys))
    key_to_group_idx = {k: i for i, k in enumerate(distinct_keys)}
    obs_group_idx = [key_to_group_idx[k] for k in keys]

    group_members: dict[int, list[int]] = defaultdict(list)
    for obs_i, gi in enumerate(obs_group_idx):
        group_members[gi].append(obs_i)

    counts = Counter(obs_group_idx)
    probs = np.array([counts[gi] / M for gi in range(len(distinct_keys))])
    group_circuits = [
        circuit_for_subset_branch(subset, branch, n) for subset, branch in distinct_keys
    ]

    shot_group_idx = rng.choice(len(distinct_keys), size=max_n_sample, p=probs)

    raw_values_by_obs: list[list[float]] = [[] for _ in range(M)]
    shot_index_by_obs: list[list[int]] = [[] for _ in range(M)]
    for t in range(max_n_sample):
        gi = int(shot_group_idx[t])
        qc = group_circuits[gi]
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        for obs_i in group_members[gi]:
            val = psi_pre.expectation_value(pauli_objs[obs_i]).real
            raw_values_by_obs[obs_i].append(val)
            shot_index_by_obs[obs_i].append(t + 1)

    return _checkpoint_rows(
        "derandomized_seeqst", M, n_samples_sorted, true_vals,
        raw_values_by_obs, shot_index_by_obs,
        extra={"num_distinct_circuits": len(distinct_keys), "num_schedule_rounds": max_n_sample},
    )


def _run_derandomized_pauli_arm(
    specs: list[dict],
    true_vals: np.ndarray,
    state: Statevector,
    n: int,
    n_samples_sorted: list[int],
    max_n_sample: int,
) -> list[dict]:
    """Huang-Kueng-Preskill's derandomized classical shadows, for a single
    (state, observable-set) trial. Calls ``derandomized_classical_shadow``
    exactly ONCE with ``num_of_measurements_per_observable = max_n_sample``:
    since an observable's match count can advance by at most 1 per round,
    and the algorithm doesn't stop until EVERY observable has reached
    max_n_sample matches, the returned schedule is guaranteed to have at
    least max_n_sample rounds -- enough to evaluate every requested
    checkpoint from a single call, the same "one long stream, many
    checkpoints" trick exp_1/exp_2 use for their own methods. See
    ``experiments/exp_4/exp_4.md`` for the caveat this implies (a schedule
    calibrated for a large target isn't bit-for-bit identical to one
    calibrated exactly for a small target -- unavoidable without re-running
    the (expensive) derandomization call once per checkpoint)."""
    M = len(specs)
    all_observables = [[(letter, pos) for pos, letter in spec.items()] for spec in specs]

    schedule = _data_acquisition_shadow.derandomized_classical_shadow(
        all_observables, max_n_sample, n
    )
    num_schedule_rounds = len(schedule)
    assert num_schedule_rounds >= max_n_sample, (
        f"derandomized_classical_shadow returned only {num_schedule_rounds} rounds for a "
        f"target of {max_n_sample} -- expected >= target (every observable needs >= target "
        "matching rounds, and match count advances by at most 1 per round). This would "
        "indicate a change in the upstream algorithm's termination behavior."
    )
    schedule = schedule[:max_n_sample]  # only ever need the first max_n_sample rounds

    raw_values_by_obs: list[list[float]] = [[] for _ in range(M)]
    shot_index_by_obs: list[list[int]] = [[] for _ in range(M)]
    for t, bases in enumerate(schedule):
        qc = _basis_circuit(bases, n)
        outcome, _ = state.evolve(qc).measure()
        outcome = str(outcome)
        signs = [1 if outcome[n - 1 - q] == "0" else -1 for q in range(n)]

        for obs_i, spec in enumerate(specs):
            matched = all(bases[pos] == letter for pos, letter in spec.items())
            if not matched:
                continue
            product = 1
            for pos in spec:
                product *= signs[pos]
            raw_values_by_obs[obs_i].append(float(product))
            shot_index_by_obs[obs_i].append(t + 1)

    return _checkpoint_rows(
        "derandomized_pauli", M, n_samples_sorted, true_vals,
        raw_values_by_obs, shot_index_by_obs,
        extra={"num_distinct_circuits": float("nan"), "num_schedule_rounds": num_schedule_rounds},
    )


def run_exp4(config: Exp4Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x n_sample) grid for
    BOTH derandomization protocols on identical states and observable sets
    -- see ``experiments/exp_4/exp_4.md`` for the full explanation.
    """
    rng = np.random.default_rng(config.seed)
    n_samples_sorted = sorted(set(int(x) for x in config.n_samples))
    max_n_sample = n_samples_sorted[-1]
    n = config.n

    total_state_draws = config.num_state_repeats * len(config.state_types)
    state_draw_idx = 0

    rows: list[dict] = []
    for state_repeat in range(config.num_state_repeats):
        for state_type in config.state_types:
            state = states.STATE_BUILDERS[state_type](n, rng)
            state_draw_idx += 1
            if verbose:
                print(
                    f"  [outer layer | state draw {state_draw_idx}/{total_state_draws}] "
                    f"state_repeat={state_repeat} state_type={state_type!r} n={n} "
                    f"-> {config.num_observable_repeats} random observable set(s), each "
                    f"re-measured from scratch for both methods"
                )

            for obs_repeat in range(config.num_observable_repeats):
                specs = _draw_observables(config, n, rng)
                M = len(specs)
                labels = [pauli_utils.spec_to_label(s, n) for s in specs]
                pauli_objs = [Pauli(label) for label in labels]
                true_vals = np.array([state.expectation_value(p).real for p in pauli_objs])

                base_row = {
                    "state_repeat": state_repeat,
                    "obs_repeat": obs_repeat,
                    "state_type": state_type,
                    "n": n,
                    "num_observables": M,
                }

                if config.include_seeqst:
                    seeqst_rows = _run_seeqst_arm(
                        specs, pauli_objs, true_vals, state, n, n_samples_sorted, max_n_sample, rng
                    )
                    rows.extend({**base_row, **r} for r in seeqst_rows)
                    if verbose:
                        print(
                            f"    [inner layer | obs draw {obs_repeat + 1}/{config.num_observable_repeats}] "
                            f"derandomized_seeqst: M={M} -> {seeqst_rows[0]['num_distinct_circuits']} distinct circuit(s)"
                        )

                if config.include_derandomized_pauli:
                    pauli_rows = _run_derandomized_pauli_arm(
                        specs, true_vals, state, n, n_samples_sorted, max_n_sample
                    )
                    rows.extend({**base_row, **r} for r in pauli_rows)
                    if verbose:
                        print(
                            f"    [inner layer | obs draw {obs_repeat + 1}/{config.num_observable_repeats}] "
                            f"derandomized_pauli: M={M}, target={max_n_sample} -> schedule of "
                            f"{pauli_rows[0]['num_schedule_rounds']} round(s)"
                        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average (and report std-dev of) each metric across BOTH
    randomization layers, for every (state_type, method, n_sample) --
    ``method`` is 'derandomized_seeqst' and/or 'derandomized_pauli'
    depending on which arms were enabled."""
    agg = (
        df.groupby(["state_type", "method", "n_sample"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mean_variance_mean=("mean_variance", "mean"),
            mean_variance_std=("mean_variance", "std"),
            max_error_mean=("max_error", "mean"),
            max_error_std=("max_error", "std"),
            coverage_mean=("coverage", "mean"),
            mean_matches_per_observable_mean=("mean_matches_per_observable", "mean"),
            num_state_repeats=("state_repeat", "nunique"),
            num_observable_repeats=("obs_repeat", "nunique"),
        )
        .reset_index()
    )
    return agg
