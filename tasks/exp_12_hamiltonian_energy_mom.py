"""Core logic for exp_12 (see ``experiments/exp_12/exp_12.py`` for the full
explanation, CLI, and plotting).

WHAT exp_12 DOES, AND HOW IT DIFFERS FROM exp_9/exp_10/exp_11
--------------------------------------------------------------------------
Every earlier experiment (exp_9-exp_11) estimates MANY random synthetic
observables against MANY random states, to build up a statistical picture
of an ensemble's typical behavior. exp_12 instead targets ONE fixed,
physically meaningful pair: a concrete numeric Jordan-Wigner electronic
Hamiltonian (``common.jw_hamiltonian``, verified against
``tests/JW_transformation.ipynb`` -- see that module's docstring) and its
EXACT ground state (found by direct diagonalization, not shadows), and
asks: how well does each ensemble's classical-shadow estimator recover the
Hamiltonian's own ground-state ENERGY, ``E_true = <ground|H|ground>``,
which we already know exactly?

Because both the state and the observable (H itself) are fixed rather than
freshly drawn every repeat, there is no "outer state layer" / "inner
observable layer" the way exp_9-exp_11 have -- there is nothing left to
randomize except which measurement outcomes happen to be sampled. So
exp_12 has a single repeat loop (``num_repeats``): each repeat draws a
fresh batch of ``max_n_sample`` shadow snapshots per ensemble and computes
its own realization of the checkpointed energy estimates; the reported
error bars are the spread ACROSS these repeats.

COMBINING H'S TERMS INTO A SINGLE ENERGY ESTIMATE -- SAME LOGIC AS exp_11
--------------------------------------------------------------------------
H = sum_i coeffs[i] * P_i (from ``jw_hamiltonian.jw_electronic_hamiltonian``,
typically ~100-600 terms for n=6). Exactly as in exp_11 (see
``tasks/exp_11_sparse_operator_mom.py``'s module docstring for the full
linearity argument): every ensemble's ``evaluate_snapshots`` gives an
unbiased single-shot estimate of EVERY term ``tr(P_i rho)`` from the SAME
snapshot stream, so ``sum_i coeffs[i] * o_hat_i^(s)`` is itself an unbiased
single-shot estimate of ``tr(H rho) = E_true``. exp_12 forms that combined
per-shot column ONCE per (repeat, ensemble) and only then applies the
point estimator (mean or median-of-means) -- identical reasoning and
identical ``_checkpoint_estimates`` implementation as exp_9/exp_10/exp_11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import jw_hamiltonian
from ensembles.base import ShadowEnsemble


class Exp12Config:
    def __init__(
        self,
        n: int,
        h: np.ndarray,
        v: np.ndarray,
        n_samples: list[int],
        num_repeats: int,
        ensembles: dict[str, ShadowEnsemble],
        estimator: str = "mean",
        mom_num_groups: int = 1,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.h = h
        self.v = v
        self.n_samples = n_samples
        self.num_repeats = num_repeats
        self.ensembles = ensembles
        self.estimator = estimator
        self.mom_num_groups = mom_num_groups
        self.seed = seed

        if estimator not in ("mean", "median_of_means"):
            raise ValueError(f"estimator={estimator!r} must be 'mean' or 'median_of_means'")
        if mom_num_groups < 1:
            raise ValueError(f"mom_num_groups={mom_num_groups} must be >= 1")
        if num_repeats < 1:
            raise ValueError(f"num_repeats={num_repeats} must be >= 1")


def _checkpoint_estimates(
    shots: np.ndarray, n_samples_sorted: list[int], estimator: str, mom_num_groups: int
) -> dict[int, float]:
    """Identical logic to exp_9/exp_10/exp_11's version, specialized to a
    single combined observable column (``shots`` has shape
    ``(max_n_sample,)`` here, not ``(max_n_sample, num_observables)`` --
    everything else, including the median-of-means group-clamping
    behavior, is unchanged; see
    ``tasks/exp_9_tuned_locality_mom.py`` for the full rationale).
    """
    out: dict[int, float] = {}
    if estimator == "mean":
        max_n_sample = shots.shape[0]
        cumulative_mean = np.cumsum(shots) / np.arange(1, max_n_sample + 1)
        for n_sample in n_samples_sorted:
            out[n_sample] = float(cumulative_mean[n_sample - 1])
    else:  # "median_of_means"
        for n_sample in n_samples_sorted:
            k_eff = max(1, min(mom_num_groups, n_sample))
            batch_size = n_sample // k_eff
            usable = batch_size * k_eff
            batches = shots[:usable].reshape(k_eff, batch_size)
            batch_means = batches.mean(axis=1)
            out[n_sample] = float(np.median(batch_means))
    return out


def run_exp12(config: Exp12Config, verbose: bool = True) -> tuple[pd.DataFrame, float, int]:
    """Returns ``(raw_df, E_true, num_terms)``.

    ``raw_df`` has one row per (repeat, ensemble, n_sample), with the
    signed error ``estimate - E_true`` and its absolute value -- there is
    only ONE observable (H itself) here, so "rmse"/"max_error" over
    multiple observables (exp_9-exp_11's metrics) collapse to a single
    ``abs_error`` per repeat; ``summarize`` below then averages that across
    repeats exactly like exp_9-exp_11 average across their two
    randomization layers.
    """
    coeffs, specs = jw_hamiltonian.jw_electronic_hamiltonian(config.n, config.h, config.v)
    state, e_true = jw_hamiltonian.ground_state(coeffs, specs, config.n)

    if verbose:
        print(
            f"  Hamiltonian: n={config.n}, {len(coeffs)} nonzero Pauli terms, "
            f"E_true (exact ground energy) = {e_true:.10g}"
        )

    rng = np.random.default_rng(config.seed)
    n_samples_sorted = sorted(set(int(x) for x in config.n_samples))
    max_n_sample = n_samples_sorted[-1]

    rows: list[dict] = []
    for repeat in range(config.num_repeats):
        if verbose:
            print(f"  [repeat {repeat + 1}/{config.num_repeats}]")
        for ens_name, ensemble in config.ensembles.items():
            snapshots = ensemble.sample_snapshots(state, config.n, max_n_sample, rng)
            term_shots = ensemble.evaluate_snapshots(snapshots, specs, config.n)
            # (max_n_sample, num_terms) -> ONE combined per-shot column,
            # using H's own coefficients -- see module docstring.
            combined_shots = term_shots @ coeffs

            estimates = _checkpoint_estimates(
                combined_shots, n_samples_sorted, config.estimator, config.mom_num_groups
            )
            for n_sample in n_samples_sorted:
                err = estimates[n_sample] - e_true
                rows.append(
                    {
                        "repeat": repeat,
                        "ensemble": ens_name,
                        "n": config.n,
                        "estimator": config.estimator,
                        "mom_num_groups": config.mom_num_groups,
                        "n_sample": n_sample,
                        "estimate": estimates[n_sample],
                        "error": err,
                        "abs_error": abs(err),
                    }
                )
    return pd.DataFrame(rows), e_true, len(coeffs)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average (and report std-dev of) the absolute error across repeats,
    for every (ensemble, n_sample)."""
    agg = (
        df.groupby(["ensemble", "n_sample"])
        .agg(
            abs_error_mean=("abs_error", "mean"),
            abs_error_std=("abs_error", "std"),
            num_repeats=("repeat", "nunique"),
        )
        .reset_index()
    )
    return agg
