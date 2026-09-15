"""Core logic for exp_11 (see ``experiments/exp_11/exp_11.py`` for the full
explanation, CLI, and plotting).

exp_11 is exp_9 with the SAME hyperparameters and the SAME four-ensemble
roster / selectable point estimator, but a fundamentally different
observable: instead of ``num_observables`` separate single-Pauli-string
observables (exp_9's "exact-m" family) or separate weight<=m single strings
(exp_10's "sparse-m" family), exp_11 tracks ``num_observables`` COMBINED
``S_m``-sparse OPERATORS -- ``A = sum_i c_i P_i``, each a random real
superposition of ``num_terms`` Pauli strings with X/Y-weight in
``{min_xy, ..., m}``, via ``pauli_utils.random_sparse_operator`` (see that
function's docstring for why any such superposition is guaranteed
``S_m``-sparse, and why sampling ``num_terms`` terms rather than
enumerating the full ``S_m`` basis is the practical choice). This is the
operational counterpart of the actual object
``notes/working/seeqst_sparse_tuning.tex`` studies -- unlike exp_8/exp_10,
which track a LIST of separately-scored single-term observables (the
``max_i beta(P_i)`` sample-complexity criterion), exp_11 estimates genuine
linear combinations.

HOW A COMBINED OPERATOR IS ESTIMATED FROM SHADOWS -- THE KEY DESIGN CHOICE
--------------------------------------------------------------------------
Every ``ShadowEnsemble.evaluate_snapshots`` call already returns, for each
snapshot ``s`` and each Pauli term ``P_i``, a single-shot estimate
``o_hat_i^(s)`` with ``E[o_hat_i^(s)] = tr(P_i rho)`` exactly (see
``ensembles/base.py``'s module docstring). Since expectation is linear,

    E[ sum_i c_i * o_hat_i^(s) ]  =  sum_i c_i * tr(P_i rho)  =  tr(A rho)

so ``sum_i c_i * o_hat_i^(s)`` -- the SAME per-shot combination the true
operator uses -- is ITSELF an unbiased single-shot estimator of ``A``'s
expectation value, using only ONE physical measurement round per shot (the
whole point of classical shadows: one batch of measurements predicts many
different observables, here combined into one). exp_11 therefore:

  1. Evaluates every INDIVIDUAL term of every combined operator against the
     ensemble's shared snapshot stream (one ``evaluate_snapshots`` call per
     ensemble, over the flattened list of all ``num_observables *
     num_terms`` terms -- exactly like exp_9/exp_10's per-Pauli columns).
  2. For each combined operator, forms ONE per-shot column as the weighted
     sum (its own ``coeffs``) of its own term columns -- shape
     ``(max_n_sample, num_observables)``, the SAME shape exp_9/exp_10 feed
     into the point estimator.
  3. Passes that combined-per-shot array into ``_checkpoint_estimates``
     UNCHANGED from exp_9/exp_10 -- "mean" or "median_of_means" is applied
     directly to the combined stream, not separately per term and then
     recombined.

Point (3) is deliberate, not an approximation: for "mean", summing first
then averaging is EXACTLY equal to averaging each term first and then
summing (linearity of the mean), so it makes no difference there. For
"median_of_means" it genuinely matters -- median is NOT linear, so
"median-of-means of the combined stream" and "linear combination of each
term's own median-of-means" are different estimators. Applying
median-of-means to the COMBINED stream (as done here) is the standard
classical-shadows treatment of a linear combination of observables (HKP's
own median-of-means estimator is stated for a single scalar random
variable per checkpoint -- here, that scalar is already the per-shot
estimator of the FULL combined operator ``A``, not of any individual term)
and is what actually lets median-of-means exploit cancellation between
terms of opposite sign, rather than pessimistically robustifying each term
in isolation.

TUNING CAVEAT -- READ BEFORE TREATING seeqst_binomial_tuned's q=m/n AS
"PROVEN OPTIMAL" HERE
--------------------------------------------------------------------------
``notes/working/seeqst_sparse_tuning.tex``'s theorem (q*=m/n, to leading
order, for m<=n/2 -- see ``experiments/exp_8/exp_8.md``'s full two-regime
derivation) is about the shadow-norm bound of the operator formed by
summing EVERY term in the full S_m basis (``sum_{s in S_m} c_s P_s``,
every eligible ``s``). exp_11's combined operators instead sum only
``num_terms`` RANDOMLY SAMPLED eligible terms -- a genuine ``S_m``-sparse
operator by the same linearity argument (see
``pauli_utils.random_sparse_operator``), but not the specific object that
theorem's constant was derived for. Using ``q=m/n`` (the SAME rule
exp_9/exp_10 already use, tuned to the ceiling ``m``) here is a reasonable,
theory-motivated default -- every term that COULD appear still has
X/Y-weight <= m, so the worst-case single-term shadow-norm contribution is
governed by the same ``beta_q(k)`` endpoint analysis exp_8.md works out --
but it is not a claim that ``q=m/n`` is proven minimax-optimal for THIS
specific random-subset-of-terms construction. Treat any
``seeqst_binomial_tuned`` advantage/disadvantage seen here as evidence
about the practical (ceiling-tuned, list-style) rule carried over to a
summed-operator setting, not as a direct empirical test of the sparse-
tuning note's own theorem (exp_8 already documents where that combined-
operator claim would need to be tested for the FULL basis, not a sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli

from common import pauli_utils, states
from ensembles.base import ShadowEnsemble


@dataclass
class Exp11Config:
    n: int
    m: int  # ceiling on EACH term's X/Y-weight: term weight ~ Uniform{min_xy, ..., m}
    num_terms: int  # number of Pauli terms summed into EACH combined sparse operator
    num_observables: int  # M, number of independent random COMBINED operators per draw
    n_samples: list[int]
    num_state_repeats: int
    num_observable_repeats: int
    ensembles: dict[str, ShadowEnsemble]
    estimator: str = "mean"  # "mean" | "median_of_means"
    mom_num_groups: int = 1  # K; only meaningful when estimator == "median_of_means"
    min_xy: int = 1  # floor on each term's X/Y-weight (>=1 keeps every term genuinely X/Y-bearing)
    rest_mode: str = "random"  # "random" | "z" | "identity" -- each term's non-XY qubits
    coeff_dist: str = "normal"  # "normal" | "uniform_pm1" -- each operator's term coefficients
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
        if self.num_terms < 1:
            raise ValueError(f"num_terms={self.num_terms} must be >= 1")
        if not (0 <= self.min_xy <= self.m <= self.n):
            raise ValueError(
                f"require 0 <= min_xy={self.min_xy} <= m={self.m} <= n={self.n}"
            )
        if self.rest_mode not in ("random", "z", "identity"):
            raise ValueError(
                f"rest_mode={self.rest_mode!r} must be one of 'random', 'z', 'identity'"
            )
        if self.coeff_dist not in ("normal", "uniform_pm1"):
            raise ValueError(
                f"coeff_dist={self.coeff_dist!r} must be 'normal' or 'uniform_pm1'"
            )


def _draw_sparse_operators(
    config: Exp11Config, n: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, list[dict]]]:
    """``num_observables`` independent random S_m-sparse combined operators,
    each ``(coeffs, specs)`` with ``len(specs) == num_terms`` -- see
    ``pauli_utils.random_sparse_operator``.
    """
    return [
        pauli_utils.random_sparse_operator(
            n,
            config.m,
            config.num_terms,
            rng,
            min_xy=config.min_xy,
            rest_mode=config.rest_mode,
            coeff_dist=config.coeff_dist,
        )
        for _ in range(config.num_observables)
    ]


def _checkpoint_estimates(
    shots: np.ndarray, n_samples_sorted: list[int], estimator: str, mom_num_groups: int
) -> dict[int, np.ndarray]:
    """Identical to exp_9/exp_10's version -- see
    ``tasks/exp_9_tuned_locality_mom.py`` for the full rationale. ``shots``
    here has shape (max_n_sample, num_observables), one column per COMBINED
    operator's own per-shot estimator (see module docstring point 2-3 above)
    rather than one column per individual Pauli term.
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
            usable = batch_size * k_eff
            batches = shots[:usable].reshape(k_eff, batch_size, -1)
            batch_means = batches.mean(axis=1)
            out[n_sample] = np.median(batch_means, axis=0)
    return out


def run_exp11(config: Exp11Config, verbose: bool = True) -> pd.DataFrame:
    """Run the full (state_repeat x observable_repeat x state_type x ensemble
    x n_sample) grid -- see ``tasks/exp_9_tuned_locality_mom.py::run_exp9``
    for the two-layer randomization design (outer: random state, inner:
    random observable set for that same state) and the per-(state, ensemble)
    snapshot-stream reuse, both carried over unchanged. The change from
    exp_9/exp_10 is entirely in the inner loop body -- see the module
    docstring's "HOW A COMBINED OPERATOR IS ESTIMATED FROM SHADOWS" section.

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
                    f"m={config.m} num_terms={config.num_terms} "
                    f"estimator={config.estimator!r} -> drawing {max_n_sample} shadow "
                    f"snapshots per ensemble, then {config.num_observable_repeats} random "
                    f"combined-operator set(s) to evaluate against them"
                )

            # Draw the snapshot stream ONCE per (state, ensemble) --
            # observable-independent -- and reuse it for every inner
            # observable-set draw below.
            snapshots_by_ensemble = {
                ens_name: ensemble.sample_snapshots(state, config.n, max_n_sample, rng)
                for ens_name, ensemble in config.ensembles.items()
            }

            for obs_repeat in range(config.num_observable_repeats):
                operators = _draw_sparse_operators(config, config.n, rng)  # inner layer

                # Flatten every term of every combined operator into ONE spec
                # list, so each ensemble's snapshot stream is queried once.
                flat_specs: list[dict] = []
                term_slices: list[tuple[int, int]] = []
                start = 0
                for coeffs, specs in operators:
                    flat_specs.extend(specs)
                    term_slices.append((start, start + len(specs)))
                    start += len(specs)

                true_vals = np.array(
                    [
                        sum(
                            c * state.expectation_value(
                                Pauli(pauli_utils.spec_to_label(s, config.n))
                            ).real
                            for c, s in zip(coeffs, specs)
                        )
                        for coeffs, specs in operators
                    ]
                )

                for ens_name, ensemble in config.ensembles.items():
                    term_shots = ensemble.evaluate_snapshots(
                        snapshots_by_ensemble[ens_name], flat_specs, config.n
                    )  # shape (max_n_sample, num_observables * num_terms)

                    # Combine each operator's own term columns with its own
                    # coefficients into ONE per-shot column per operator --
                    # see module docstring for why this happens BEFORE the
                    # point estimator, not after.
                    combined_shots = np.empty((term_shots.shape[0], len(operators)))
                    for i, (coeffs, _specs) in enumerate(operators):
                        s0, s1 = term_slices[i]
                        combined_shots[:, i] = term_shots[:, s0:s1] @ np.asarray(coeffs)

                    estimates = _checkpoint_estimates(
                        combined_shots, n_samples_sorted, config.estimator, config.mom_num_groups
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
                                "num_terms": config.num_terms,
                                "num_observables": config.num_observables,
                                "min_xy": config.min_xy,
                                "rest_mode": config.rest_mode,
                                "coeff_dist": config.coeff_dist,
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
