"""Core logic for exp_5 (see ``exp_5.py`` at the repo root for the full
explanation, CLI, and plotting). Kept separate so the experiment logic can
be unit-tested / reused independently of argument parsing and plotting,
exactly as ``tasks/exp_1_error_scaling.py`` does for exp_1.

exp_5 asks the same "error vs. number of shadow snapshots" question as
exp_1, but for the lattice Schwinger model Hamiltonian (Kokail et al.
arXiv:1810.03421 Eq. 1-2 / HKP SI arXiv:2002.08953 Eq. S32, main-text
Fig. 5 -- see ``common/hamiltonians.py``) instead of randomly drawn Pauli
observables. That swap changes the design in one structural way: exp_1 has
TWO randomization layers (a random state, and independently for that state
a random observable set) because both the state and the observables it
studies are, by design, arbitrary. Here neither is arbitrary -- the
Hamiltonian's terms are fixed by (n, w, m, g), and the states studied
(``ground_state``, ``neel``) are specific physically-meaningful states, not
random draws. So there is only ONE randomization layer left: measurement
(shot) noise, repeated ``--num-repeats`` times per state purely to get error
bars on the metrics below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli, Statevector

from common import states
from common.hamiltonians import TermGroup, build_schwinger_system, classify_term, neel_state
from common.pauli_utils import spec_to_label
from ensembles.base import ShadowEnsemble

TERM_GROUPS: list[TermGroup] = ["z_single", "zz_long_range", "hopping"]


@dataclass
class Exp5Config:
    n: int
    w: float
    m: float
    g: float
    n_samples: list[int]
    num_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    state_types: list[str] = field(default_factory=lambda: ["ground_state", "neel"])
    seed: int = 0


def _build_state(state_type: str, n: int, system, rng: np.random.Generator) -> Statevector:
    if state_type == "ground_state":
        return system.ground_state
    if state_type == "neel":
        return neel_state(n)
    if state_type == "haar_random":
        return states.STATE_BUILDERS["haar_random"](n, rng)
    raise ValueError(f"unknown state_type {state_type!r} (expected ground_state, neel, or haar_random)")


def run_exp5(config: Exp5Config, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full (repeat x state_type x ensemble x n_sample) grid against
    the Schwinger Hamiltonian's own fixed terms. Returns ``(raw_terms,
    raw_energy)``:

      raw_terms  -- one row per (repeat, state_type, ensemble, n_sample,
                    term_group), ``term_group`` in ``TERM_GROUPS`` --
                    ``classify_term``/Check 5 of
                    ``tests/verify_schwinger_hamiltonian.py`` confirm these
                    three are the only term types this Hamiltonian produces.
                    Same RMSE / MeanVariance / MaxError metrics as exp_1
                    (see that module's docstring for the exact definitions),
                    computed within each physical term group separately
                    rather than lumped together, since the whole point of
                    this experiment is that the ensembles perform very
                    differently on Z-type vs. hopping terms.
      raw_energy -- one row per (repeat, state_type, ensemble, n_sample) --
                    the total-energy estimate ``E_hat = identity_offset +
                    sum_i c_i * o_hat_i``, formed from the SAME shared
                    per-shot snapshots used for raw_terms. Because it's
                    built shot-by-shot from data that already mixes every
                    term together, its empirical variance
                    (``energy_variance``) automatically includes the
                    cross-term covariances ``Cov[o_hat_i, o_hat_j]`` induced
                    by sharing snapshots across terms -- no separate
                    derivation of those covariances is needed. For
                    comparison, ``energy_variance_naive`` is the variance
                    you'd get by (incorrectly) assuming the per-term
                    estimators are independent, ``sum_i c_i^2 Var[o_hat_i]``;
                    comparing the two columns shows directly whether shared-
                    snapshot correlations matter for this Hamiltonian.

    Just like exp_1: for each (state_type, ensemble, repeat), ONE stream of
    ``max(n_samples)`` snapshots is drawn (``ShadowEnsemble.sample_snapshots``)
    BEFORE looking at which Hamiltonian term is being estimated, and reused
    (``ShadowEnsemble.evaluate_snapshots``) across every term and every
    ``n_sample`` checkpoint -- "measure once, mine many times". Unlike
    exp_1, the observable set (the Hamiltonian's terms) is never redrawn --
    only the measurement noise differs between repeats.

    If ``verbose``, prints one line per (repeat, state_type) draw.
    """
    rng = np.random.default_rng(config.seed)
    n_samples_sorted = sorted(set(int(x) for x in config.n_samples))
    max_n_sample = n_samples_sorted[-1]

    system = build_schwinger_system(config.n, w=config.w, m=config.m, g=config.g)
    specs = [spec for _, spec in system.terms]
    coeffs = np.array([c for c, _ in system.terms], dtype=float)
    term_groups = [classify_term(spec, config.n) for spec in specs]
    group_indices = {grp: [i for i, g in enumerate(term_groups) if g == grp] for grp in TERM_GROUPS}

    total_draws = config.num_repeats * len(config.state_types)
    draw_idx = 0

    term_rows: list[dict] = []
    energy_rows: list[dict] = []
    for repeat in range(config.num_repeats):
        for state_type in config.state_types:
            state = _build_state(state_type, config.n, system, rng)
            draw_idx += 1
            if verbose:
                print(
                    f"  [repeat {repeat + 1}/{config.num_repeats} | draw {draw_idx}/{total_draws}] "
                    f"state_type={state_type!r} n={config.n} -> drawing {max_n_sample} shadow "
                    f"snapshots per ensemble against the Hamiltonian's {len(specs)} fixed terms "
                    f"({len(group_indices['z_single'])} z_single, {len(group_indices['zz_long_range'])} "
                    f"zz_long_range, {len(group_indices['hopping'])} hopping)"
                )

            true_vals = np.array(
                [state.expectation_value(Pauli(spec_to_label(s, config.n))).real for s in specs]
            )
            true_energy = system.identity_offset + float(np.dot(coeffs, true_vals))

            for ens_name, ensemble in config.ensembles.items():
                # Draw ONCE per (repeat, state_type, ensemble) -- observable-
                # independent -- and reuse across every term / checkpoint below.
                snapshots = ensemble.sample_snapshots(state, config.n, max_n_sample, rng)
                shots = ensemble.evaluate_snapshots(snapshots, specs, config.n)  # (max_n_sample, num_terms)
                cumulative_mean = np.cumsum(shots, axis=0) / np.arange(1, max_n_sample + 1).reshape(-1, 1)

                # Single-shot estimator of sum_i c_i O_i, from the SAME shots
                # array -- this is what makes energy_variance below capture
                # cross-term correlations "for free".
                energy_shots = shots @ coeffs  # (max_n_sample,)
                energy_cumulative_mean = np.cumsum(energy_shots) / np.arange(1, max_n_sample + 1)

                for n_sample in n_samples_sorted:
                    est = cumulative_mean[n_sample - 1]
                    err = est - true_vals
                    per_term_var = np.var(shots[:n_sample, :], axis=0, ddof=1)

                    for group in TERM_GROUPS:
                        idx = group_indices[group]
                        if not idx:  # e.g. n=2 has no zz_long_range term at all
                            continue
                        term_rows.append(
                            {
                                "repeat": repeat,
                                "state_type": state_type,
                                "ensemble": ens_name,
                                "n": config.n,
                                "n_sample": n_sample,
                                "term_group": group,
                                "num_terms": len(idx),
                                "rmse": float(np.sqrt(np.mean(err[idx] ** 2))),
                                "mean_variance": float(np.mean(per_term_var[idx])),
                                "max_error": float(np.max(np.abs(err[idx]))),
                            }
                        )

                    energy_est = system.identity_offset + float(energy_cumulative_mean[n_sample - 1])
                    energy_error = energy_est - true_energy
                    energy_variance = float(np.var(energy_shots[:n_sample], ddof=1))
                    energy_variance_naive = float(np.sum((coeffs**2) * per_term_var))

                    energy_rows.append(
                        {
                            "repeat": repeat,
                            "state_type": state_type,
                            "ensemble": ens_name,
                            "n": config.n,
                            "n_sample": n_sample,
                            "true_energy": true_energy,
                            "energy_estimate": energy_est,
                            "energy_error": energy_error,
                            "energy_abs_error": abs(energy_error),
                            "energy_variance": energy_variance,
                            "energy_variance_naive": energy_variance_naive,
                        }
                    )
    return pd.DataFrame(term_rows), pd.DataFrame(energy_rows)


def summarize_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Average each per-term-group metric across repeats, for every
    (state_type, ensemble, term_group, n_sample)."""
    if df.empty:
        return df
    agg = (
        df.groupby(["state_type", "ensemble", "term_group", "n_sample"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mean_variance_mean=("mean_variance", "mean"),
            mean_variance_std=("mean_variance", "std"),
            max_error_mean=("max_error", "mean"),
            max_error_std=("max_error", "std"),
            num_terms=("num_terms", "first"),
            num_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )
    return agg


def summarize_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Average each energy-level metric across repeats, for every
    (state_type, ensemble, n_sample). ``energy_rmse`` is the RMS of the
    signed ``energy_error`` across repeats (distinct from
    ``energy_abs_error_mean``, the mean of the per-repeat absolute errors --
    the same RMS-vs-mean-absolute distinction exp_1 draws between RMSE and
    MaxError, just both computed here across repeats instead of across
    observables)."""
    if df.empty:
        return df
    agg = (
        df.groupby(["state_type", "ensemble", "n_sample"])
        .agg(
            energy_error_mean=("energy_error", "mean"),
            energy_abs_error_mean=("energy_abs_error", "mean"),
            energy_abs_error_std=("energy_abs_error", "std"),
            energy_rmse=("energy_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            energy_variance_mean=("energy_variance", "mean"),
            energy_variance_naive_mean=("energy_variance_naive", "mean"),
            true_energy=("true_energy", "first"),
            num_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )
    return agg
