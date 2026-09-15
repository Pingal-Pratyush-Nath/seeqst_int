"""Core logic for exp_13 (see ``experiments/exp_13/exp_13.py`` for the full
explanation, CLI, and plotting).

exp_13 is exp_12's exact pipeline (fixed state = exact ground state, fixed
observable = H itself, combine-then-estimate, repeat loop for error bars --
see ``tasks/exp_12_hamiltonian_energy_mom.py``'s module docstring for the
full rationale, unchanged here) pointed at a DIFFERENT Hamiltonian: the
long-range p-wave pairing Kitaev chain
(``common.jw_hamiltonian.jw_long_range_kitaev``) instead of exp_12's
nearest-neighbour-only ``demo_integrals``. The point of swapping in this
Hamiltonian is that its pairing term genuinely produces Pauli weights up to
n (via the Jordan-Wigner Z-string between paired sites), unlike
``demo_integrals`` (weight <= 2 always) -- see that function's docstring
for the parameter regime (weak/zero hopping and chemical potential, small
alpha, larger n) actually needed for SEEQST/Clifford to beat Pauli in
aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import jw_hamiltonian
from ensembles.base import ShadowEnsemble


class Exp13Config:
    def __init__(
        self,
        n: int,
        hopping: float,
        mu: float,
        delta: float,
        alpha: float,
        n_samples: list[int],
        num_repeats: int,
        ensembles: dict[str, ShadowEnsemble],
        estimator: str = "mean",
        mom_num_groups: int = 1,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.hopping = hopping
        self.mu = mu
        self.delta = delta
        self.alpha = alpha
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
    """Identical to ``tasks/exp_12_hamiltonian_energy_mom.py``'s version --
    see that module for the full rationale."""
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


def run_exp13(config: Exp13Config, verbose: bool = True) -> tuple[pd.DataFrame, float, int]:
    """Returns ``(raw_df, E_true, num_terms)`` -- see
    ``tasks/exp_12_hamiltonian_energy_mom.py::run_exp12`` for the full
    per-row schema and rationale (identical here, just a different
    Hamiltonian source)."""
    coeffs, specs = jw_hamiltonian.jw_long_range_kitaev(
        config.n, config.hopping, config.mu, config.delta, config.alpha
    )
    state, e_true = jw_hamiltonian.ground_state(coeffs, specs, config.n)

    if verbose:
        print(
            f"  Long-range Kitaev chain: n={config.n}, hopping={config.hopping}, "
            f"mu={config.mu}, delta={config.delta}, alpha={config.alpha}, "
            f"{len(coeffs)} nonzero Pauli terms, E_true (exact ground energy) = {e_true:.10g}"
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
    """Identical to ``tasks/exp_12_hamiltonian_energy_mom.py::summarize``."""
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
