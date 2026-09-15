"""Core logic for exp_2 (see ``exp_2.py`` at the repo root for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli, Statevector

from common import pauli_utils, states
from experiments.Archived.exp_3.exp_3 import circuit_for_subset_branch, select_subset_and_branch
from ensembles.seeqst_ensemble import SEEQSTEnsemble


@dataclass
class Exp2Config:
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
    include_random_seeqst: bool = True  # also compute exp_1-style random-SEEQST rows, same states/observables


def _draw_observables(config: Exp2Config, n: int, rng: np.random.Generator) -> list[dict]:
    """Same observable-generating distribution as exp_1's ``_draw_observables``
    (duplicated, not imported, since it's typed against ``Exp1Config`` there)
    -- kept identical on purpose so exp_1 and exp_2 draw M random observables
    from the SAME distribution, for a fair, apples-to-apples comparison."""
    if config.k is None:
        return [pauli_utils.random_full_pauli_spec(n, rng) for _ in range(config.num_observables)]
    return [pauli_utils.random_pauli_spec(n, config.k, rng) for _ in range(config.num_observables)]


def run_exp2(config: Exp2Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x n_sample) grid for
    the derandomized-SEEQST protocol -- see exp_2.py's module docstring for
    the full explanation of the protocol and the metrics below.

    Per (state, observable-set) pair:
      1. Derandomize every one of the M observables into its exact-match
         SEEQST (subset, branch) circuit (``experiments.Archived.exp_3.exp_3``,
         the derandomization module -- see that file's docstring for the
         rule; it lives under ``Archived/`` for folder organization only --
         it is NOT deprecated, this is a live dependency).
      2. Group the M observables by their distinct circuit, and draw
         max(n_samples) shots total, each shot's circuit sampled from the
         categorical distribution {circuit -> (# observables using it) / M}.
      3. Every shot updates the running raw-value history of EVERY
         observable that shares its circuit (a single shot can inform
         several observables at once -- this "measurement grouping" is the
         entire efficiency gain over random SEEQST/exp_1).
      4. At each n_sample checkpoint, each observable's estimate is the mean
         of its OWN matching-shot history *up to that point in the shared
         shot stream* (a variable, possibly zero, count per observable --
         see "coverage" below).

    IMPORTANT -- fresh snapshots per observable set: unlike exp_1 (where one
    snapshot stream was drawn per state and reused across every inner
    observable-set draw), here the sampling distribution over circuits is
    itself derived from the observable set, so a new state has to be
    re-measured from scratch for every inner obs_repeat draw. There is no
    "measure once, reuse for many observable sets" step here, because the
    circuit-sampling policy is observable-set-dependent by construction.

    IMPORTANT -- no beta rescaling: unlike exp_1's estimator
    (``beta(P) * <psi_pre|P|psi_pre>``, unbiased only because P's
    measurement circuit is drawn from the SAME distribution beta(P) was
    computed under), here every shot used for observable P_i comes from
    P_i's own exactly-diagonalizing circuit, so ``<psi_pre|P_i|psi_pre>`` is
    ALREADY an exactly unbiased single-shot estimate of Tr(P_i rho) on its
    own (a deterministic +-1 value whose Born-rule average is exactly the
    true expectation value -- ordinary basis-rotated measurement, no
    "hit or miss" and no shadow inverse-map needed). Multiplying by beta(P)
    here would be a bug: it would inflate every estimate by a factor of
    beta(P) (2 or 2^(n+1)), since there is no "miss" probability left to
    correct for.

    If ``verbose``, prints one line per outer-layer state draw, as in
    exp_1.

    RANDOM-SEEQST COMPARISON (``config.include_random_seeqst``, default True):
    for a true apples-to-apples baseline, we also run exp_1's *exact*
    protocol for the SEEQST ensemble -- uniformly-random (subset, branch)
    per shot, one shared snapshot stream per state reused across every
    obs_repeat, beta(P)-rescaled hit-or-miss estimator -- on the SAME states
    (and, per state, the same observable sets) drawn for the derandomized
    protocol above, rather than reading a separately-run exp_1's output
    file. This guarantees the comparison is on identical states/observables
    (not just the same n and seed), and needs no prior exp_1 run to exist.
    Rows from this baseline are tagged ``method='random_seeqst'`` (the
    derandomized protocol's rows are tagged ``method='derandomized_seeqst'``)
    with ``coverage`` and ``mean_matches_per_observable`` fixed at 1.0 and
    n_sample respectively, since every shot there contributes (via beta
    rescaling, not exclusion) to every observable's running mean -- see
    ``ensembles/seeqst_ensemble.py`` and exp_1's ``run_exp1`` for that
    estimator's own derivation.
    """
    rng = np.random.default_rng(config.seed)
    n_samples_sorted = sorted(set(int(x) for x in config.n_samples))
    max_n_sample = n_samples_sorted[-1]
    n = config.n

    seeqst_ensemble = SEEQSTEnsemble() if config.include_random_seeqst else None

    total_state_draws = config.num_state_repeats * len(config.state_types)
    state_draw_idx = 0

    rows: list[dict] = []
    for state_repeat in range(config.num_state_repeats):
        for state_type in config.state_types:
            state = states.STATE_BUILDERS[state_type](n, rng)  # outer layer: fresh state
            state_draw_idx += 1

            # Random-SEEQST baseline: one shared snapshot stream for this
            # state, drawn once and reused across every obs_repeat below --
            # exactly exp_1's "measure once, mine many observable sets"
            # design (unlike the derandomized protocol, this ensemble's
            # sampling distribution doesn't depend on the observables).
            random_snapshots = (
                seeqst_ensemble.sample_snapshots(state, n, max_n_sample, rng)
                if seeqst_ensemble is not None
                else None
            )

            if verbose:
                print(
                    f"  [outer layer | state draw {state_draw_idx}/{total_state_draws}] "
                    f"state_repeat={state_repeat} state_type={state_type!r} n={n} "
                    f"-> {config.num_observable_repeats} random observable set(s), each "
                    f"re-measured from scratch for the derandomized protocol"
                    + (" (+ 1 shared random-SEEQST snapshot stream for this state)" if random_snapshots is not None else "")
                )

            for obs_repeat in range(config.num_observable_repeats):
                specs = _draw_observables(config, n, rng)  # inner layer: fresh observable set
                M = len(specs)
                labels = [pauli_utils.spec_to_label(s, n) for s in specs]
                pauli_objs = [Pauli(label) for label in labels]
                true_vals = np.array(
                    [state.expectation_value(p).real for p in pauli_objs]
                )

                # --- derandomize + group -------------------------------------------------
                keys = [select_subset_and_branch(s, n) for s in specs]  # (subset, branch) per obs
                distinct_keys = list(dict.fromkeys(keys))  # unique, first-seen order
                key_to_group_idx = {k: i for i, k in enumerate(distinct_keys)}
                obs_group_idx = [key_to_group_idx[k] for k in keys]  # length M

                group_members: dict[int, list[int]] = defaultdict(list)
                for obs_i, gi in enumerate(obs_group_idx):
                    group_members[gi].append(obs_i)

                counts = Counter(obs_group_idx)
                probs = np.array([counts[gi] / M for gi in range(len(distinct_keys))])
                group_circuits = [
                    circuit_for_subset_branch(subset, branch, n) for subset, branch in distinct_keys
                ]

                if verbose:
                    print(
                        f"    [inner layer | obs draw {obs_repeat + 1}/{config.num_observable_repeats}] "
                        f"M={M} observables -> {len(distinct_keys)} distinct SEEQST circuit(s) "
                        f"(grouping factor {M / len(distinct_keys):.2f}x)"
                    )

                # --- draw max_n_sample shots, frequency-weighted over distinct circuits ---
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
                        shot_index_by_obs[obs_i].append(t + 1)  # 1-indexed total-shot count

                # --- evaluate at every n_sample checkpoint --------------------------------
                for n_sample in n_samples_sorted:
                    errs = []
                    variances = []
                    match_counts = np.empty(M, dtype=int)
                    for obs_i in range(M):
                        n_i = bisect_right(shot_index_by_obs[obs_i], n_sample)
                        match_counts[obs_i] = n_i
                        if n_i == 0:
                            # No matching shot yet for this observable -- excluded from
                            # RMSE/MeanVariance/MaxError below (there's no data to form an
                            # estimate from). ``coverage`` tracks how many observables that
                            # excludes at this checkpoint.
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
                            "method": "derandomized_seeqst",
                            "state_repeat": state_repeat,
                            "obs_repeat": obs_repeat,
                            "state_type": state_type,
                            "n": n,
                            "num_observables": M,
                            "num_distinct_circuits": len(distinct_keys),
                            "n_sample": n_sample,
                            "rmse": float(np.sqrt(np.mean(errs_arr**2))) if errs else float("nan"),
                            "mean_variance": float(np.mean(variances)) if variances else float("nan"),
                            "max_error": float(np.max(np.abs(errs_arr))) if errs else float("nan"),
                            "coverage": coverage,
                            "mean_matches_per_observable": float(np.mean(match_counts)),
                        }
                    )

                # --- random-SEEQST baseline (exp_1's protocol), same state + same specs ---
                if random_snapshots is not None:
                    shots = seeqst_ensemble.evaluate_snapshots(random_snapshots, specs, n)  # (max_n_sample, M)
                    cumulative_mean = np.cumsum(shots, axis=0) / np.arange(
                        1, max_n_sample + 1
                    ).reshape(-1, 1)

                    for n_sample in n_samples_sorted:
                        est = cumulative_mean[n_sample - 1]
                        err = est - true_vals
                        per_observable_var = np.var(shots[:n_sample, :], axis=0, ddof=1)
                        rows.append(
                            {
                                "method": "random_seeqst",
                                "state_repeat": state_repeat,
                                "obs_repeat": obs_repeat,
                                "state_type": state_type,
                                "n": n,
                                "num_observables": M,
                                "n_sample": n_sample,
                                "rmse": float(np.sqrt(np.mean(err**2))),
                                "mean_variance": float(np.mean(per_observable_var)),
                                "max_error": float(np.max(np.abs(err))),
                                "coverage": 1.0,  # every shot contributes (via beta) to every observable
                                "mean_matches_per_observable": float(n_sample),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average (and report std-dev of) each metric across BOTH randomization
    layers (all state_repeat x obs_repeat trials pooled together), for every
    (state_type, method, n_sample) -- ``method`` is 'derandomized_seeqst' or,
    if ``config.include_random_seeqst`` was set, also 'random_seeqst'.
    RMSE/MaxError/MeanVariance rows are averaged with NaNs (from n_sample
    checkpoints with zero covered observables in that trial) excluded via
    pandas' default skipna behaviour."""
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
