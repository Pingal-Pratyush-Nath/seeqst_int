#!/usr/bin/env python3
"""
================================================================================
 exp_5 -- Estimating a physical Hamiltonian's energy from classical shadows
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_0 and exp_1 both ask "how well do Pauli / Clifford / SEEQST shadows
predict RANDOM Pauli observables" -- a clean question, but not one any real
experiment actually cares about on its own. The README's own "not covered
here" list names the obvious next step: something physical. This experiment
is that step, for the application classical-shadow papers reach for first:
estimating the energy of a real Hamiltonian (Huang, Kueng & Preskill's own
main-text Fig. 5, arXiv:2002.08953).

The Hamiltonian is the lattice Schwinger model of Kokail et al.,
"Self-verifying variational quantum simulation of lattice models," Nature
569, 355 (2019) (arXiv:1810.03421) -- the exact system HKP's Fig. 5 uses,
and the system Kokail et al. ran variational quantum simulation (VQE-style)
on with up to 20 trapped ions. See ``common/hamiltonians.py`` for the full
construction (built directly from Kokail's Eq. 1-2 via qiskit
``SparsePauliOp`` algebra, verified six different independent ways in
``tests/verify_schwinger_hamiltonian.py``) and Kokail's own default
couplings ``w=1, m=0.9, g=1`` (their Fig. 2a caption), used here too.

WHY THIS IS A STRUCTURALLY DIFFERENT EXPERIMENT FROM exp_1, NOT JUST exp_1
WITH A DIFFERENT OBSERVABLE LIST
--------------------------------------------------------------------------------
exp_1 studies random states and random observable sets, so it needs TWO
layers of randomization to average over: which state, and which observables.
Here, neither is arbitrary:

  - The OBSERVABLES are the Hamiltonian's own Pauli terms -- fixed exactly by
    (n, w, m, g), never redrawn. HKP SI Eq. S32 (and
    ``tests/verify_schwinger_hamiltonian.py`` Check 5) show this Hamiltonian
    only ever produces three structural term types: a single on-site Z (the
    staggered mass), a long-range ZZ pair (from the Gauss-law electric-field
    term L_j^2, which couples every pair of links up to j -- "long range"
    despite the model being nearest-neighbour in the hopping term), and a
    nearest-neighbour XX or YY pair (the hopping term). This benchmark
    reports its metrics broken down by these three ``term_group``s
    separately, rather than lumped together, because -- unlike a uniformly
    random Pauli set -- the ensembles' performance on this Hamiltonian is
    NOT uniform across term types (see RESULT HEADLINE below).
  - The STATE is one of a small number of specific, physically-meaningful
    reference states, not a random draw from a state family:
      * ``ground_state`` -- the model's exact ground state (dense
        diagonalization, ``common/hamiltonians.ground_state``). This is the
        state a real VQE/VQS run is trying to prepare -- exactly what
        Kokail et al.'s "self-verifying" protocol uses classical-shadow-like
        measurements to check convergence against, and HKP Fig. 5's own
        target state.
      * ``neel`` -- the staggered classical/product reference state (the
        "bare vacuum", see ``common/hamiltonians.neel_state``): zero
        entanglement, by construction EXACTLY zero expectation value on
        every hopping term. Included as the opposite extreme from
        ``ground_state`` on the entanglement axis, and because it's the
        standard starting point for a VQE ansatz on this model.
      * ``haar_random`` (available, not run by default) -- a generic random
        state, reusing ``common/states.py``, as a structure-free baseline.
    So the only thing left to randomize over, for a fixed (n, w, m, g,
    state_type), is measurement (shot) noise -- ``--num-repeats`` independent
    draws of a snapshot stream, purely to put error bars on the metrics.
    There is one randomization layer here, not two.

MEASURE ONCE, MINE MANY TIMES (same discipline as exp_1)
--------------------------------------------------------------------------------
For each (repeat, state_type, ensemble), ONE stream of max(n_samples)
classical-shadow snapshots is drawn (``ShadowEnsemble.sample_snapshots``)
BEFORE looking at which Hamiltonian term is being estimated, and reused
(``ShadowEnsemble.evaluate_snapshots``) across every term in the Hamiltonian
and every N_sample checkpoint. The state is never re-measured just because a
different term (or the total energy) is being read off -- exactly the
property that makes classical shadows useful for Hamiltonian estimation in
the first place (Kokail et al. and HKP both stress this: one measurement
record, reused to bound the energy and check VQE convergence).

METRICS
--------------------------------------------------------------------------------
Per-term (grouped by ``term_group`` -- see ``tasks/exp_5_energy_estimation.py``
for the exact code): the same three metrics as exp_1, computed against the
Hamiltonian's own terms instead of random Pauli strings, averaged within
each term group separately.

    RMSE(N_sample)          = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MeanVariance(N_sample)  = mean_i Var[o_hat_i^(1)]   (single-shot variance,
                              should flatten as N_sample grows -- see exp_1's
                              docstring for the full Lemma S1 connection)
    MaxError(N_sample)      = max_i |o_hat_i(N_sample) - o_i|

where the mean/max above run over the terms in that group, i = 1..|group|.

Energy-level (the actual physical quantity -- "estimating energies"): the
running-mean TOTAL energy estimate

    E_hat(N_sample) = identity_offset + sum_i c_i * o_hat_i(N_sample)

formed from the SAME shared per-shot data as the per-term metrics above (c_i
are the Hamiltonian's real coefficients, identity_offset is the constant
term dropped by ``hamiltonian_terms``). Reported, averaged over repeats:

    energy_abs_error(N_sample) = mean over repeats of |E_hat(N_sample) - E_exact|
    energy_variance(N_sample)  = Var[ sum_i c_i * o_hat_i^(1) ], the
                                  single-shot TOTAL-energy estimator's
                                  variance -- computed from the per-shot
                                  weighted sum directly, so it automatically
                                  includes every cross-term covariance
                                  Cov[o_hat_i, o_hat_j] induced by sharing one
                                  snapshot stream across all the terms. No
                                  covariance is separately derived or assumed
                                  away.
    energy_variance_naive(N_sample) = sum_i c_i^2 * Var[o_hat_i], the
                                  variance you'd get by (incorrectly)
                                  pretending the per-term estimators are
                                  independent. Plotted alongside
                                  energy_variance so the gap between the two
                                  -- i.e. whether shared-snapshot correlations
                                  actually matter for this Hamiltonian and
                                  ensemble -- is visible directly, not just
                                  asserted.

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                system size (number of qubits/sites; must be even --
                      the model's own vacuum-boundary-condition requirement,
                      see common/hamiltonians.py)
  --w, --m, --g       Hamiltonian couplings (default: Kokail et al.'s own
                      w=1, m=0.9, g=1)
  --n-samples         list of N_sample checkpoints to evaluate at
  --num-repeats       independent measurement-noise repeats (error bars)
  --state-types       subset of ground_state / neel / haar_random to run
  --seed              RNG seed
  --version           explicit output folder name (default: auto-incrementing
                      v0, v1, ... -- see OUTPUT below)
  --quick             small/fast config for smoke-testing
  --quiet             suppress the per-draw progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_5/v<N>/ (same
auto-incrementing scheme as exp_1):
  hyperparameters.json      every setting used for this run, plus the
                             Hamiltonian's term-group counts and exact ground
                             energy for provenance
  raw_terms.csv     one row per (repeat, state_type, ensemble, n_sample, term_group)
  raw_energy.csv    one row per (repeat, state_type, ensemble, n_sample)
  summary_terms.csv     raw_terms.csv averaged over repeats
  summary_energy.csv    raw_energy.csv averaged over repeats
  plots (per state_type):
    {rmse,mean_variance,max_error}_vs_nsample_{z_single,zz_long_range,hopping}_<state>.png
        -- 9 plots, log-log, one line per ensemble, same reference-line
           convention as exp_1 (1/sqrt(N_sample) for RMSE/MaxError, flat for
           MeanVariance).
    energy_error_vs_nsample_<state>.png
        -- mean |energy error| vs. N_sample, one line per ensemble,
           1/sqrt(N_sample) reference.
    energy_variance_vs_nsample_<state>.png
        -- actual (solid) vs. naive/uncorrelated (dotted) single-shot energy
           variance, one colour per ensemble, flat reference at the actual
           variance's most-converged value.
  i.e. 11 plots x len(state_types) (22 for the 2 default state types).

USAGE
--------------------------------------------------------------------------------
    python exp_5.py --quick
    python exp_5.py --n 6
    python exp_5.py --n 8 --num-repeats 8 --n-samples 10 30 100 300 1000 3000 10000
    python exp_5.py --n 6 --state-types ground_state neel haar_random
    python exp_5.py --n 6 --m 0.0     # switch off the staggered mass term
    python exp_5.py --n 6 --version my_run_name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# See exp_1.py's identical bootstrap comment: this script lives two levels
# below the shadow_benchmark root, so the root needs adding to sys.path by
# hand for "from common..." etc. to resolve when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.hamiltonians import build_schwinger_system, classify_term
from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from tasks.exp_5_energy_estimation import (
    TERM_GROUPS,
    Exp5Config,
    run_exp5,
    summarize_energy,
    summarize_terms,
)

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_5/results/), matching every other exp_N script.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_5"

TERM_METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("mean_variance_mean", "mean_variance_std", "Mean variance  Var[o_hat]", "mean_variance", "flat"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
]
TERM_GROUP_LABEL = {
    "z_single": "mass (single Z)",
    "zz_long_range": "Gauss law (ZZ)",
    "hopping": "hopping (XX/YY)",
}
ENSEMBLE_STYLE = {"pauli": ("o", "tab:blue"), "clifford": ("s", "tab:orange"), "seeqst": ("^", "tab:green")}


def build_ensembles() -> dict:
    return {"pauli": PauliEnsemble(), "clifford": CliffordEnsemble(), "seeqst": SEEQSTEnsemble()}


def _inv_sqrt_n_reference(ax, xs: np.ndarray, anchor: tuple[float, float]) -> None:
    x0, y0 = anchor
    ref = y0 * np.sqrt(x0 / xs)
    ax.plot(xs, ref, "k--", alpha=0.4, lw=1, label=r"$\propto 1/\sqrt{N_{sample}}$ (reference)")


def _flat_reference(ax, xs: np.ndarray, value: float) -> None:
    ax.plot(xs, np.full_like(xs, value), "k--", alpha=0.4, lw=1, label="converged value (reference)")


def plot_term_group(summary_terms, state_type: str, term_group: str, out_dir: Path) -> None:
    sub = summary_terms[(summary_terms.state_type == state_type) & (summary_terms.term_group == term_group)]
    if sub.empty:  # e.g. n=2 has no zz_long_range term at all
        return
    for mean_col, std_col, ylabel, tag, ref_style in TERM_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ref_anchor = None
        for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
            se = sub[sub.ensemble == ens_name].sort_values("n_sample")
            if se.empty:
                continue
            ax.errorbar(
                se.n_sample, se[mean_col], yerr=se[std_col],
                marker=marker, color=color, label=ens_name, capsize=2, lw=1.5,
            )
            if ref_style == "inv_sqrt_n" and ref_anchor is None:
                ref_anchor = (se.n_sample.iloc[0], se[mean_col].iloc[0])
            elif ref_style == "flat" and ref_anchor is None:
                ref_anchor = se[mean_col].iloc[-1]
        xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
        if ref_style == "inv_sqrt_n" and ref_anchor is not None:
            _inv_sqrt_n_reference(ax, xs, ref_anchor)
        elif ref_style == "flat" and ref_anchor is not None:
            _flat_reference(ax, xs, ref_anchor)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  --  {TERM_GROUP_LABEL[term_group]}, {state_type}", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_vs_nsample_{term_group}_{state_type}.png", dpi=150)
        plt.close(fig)


def plot_energy_error(summary_energy, state_type: str, out_dir: Path) -> None:
    sub = summary_energy[summary_energy.state_type == state_type]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ref_anchor = None
    for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
        se = sub[sub.ensemble == ens_name].sort_values("n_sample")
        if se.empty:
            continue
        ax.errorbar(
            se.n_sample, se.energy_abs_error_mean, yerr=se.energy_abs_error_std,
            marker=marker, color=color, label=ens_name, capsize=2, lw=1.5,
        )
        if ref_anchor is None:
            ref_anchor = (se.n_sample.iloc[0], se.energy_abs_error_mean.iloc[0])
    xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
    if ref_anchor is not None:
        _inv_sqrt_n_reference(ax, xs, ref_anchor)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
    ax.set_ylabel("Mean |total energy error|")
    ax.set_title(f"Total energy error  --  {state_type}", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"energy_error_vs_nsample_{state_type}.png", dpi=150)
    plt.close(fig)


def plot_energy_variance(summary_energy, state_type: str, out_dir: Path) -> None:
    sub = summary_energy[summary_energy.state_type == state_type]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ref_anchor = None
    for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
        se = sub[sub.ensemble == ens_name].sort_values("n_sample")
        if se.empty:
            continue
        ax.plot(se.n_sample, se.energy_variance_mean, marker=marker, color=color, lw=1.5, label=f"{ens_name} (actual)")
        ax.plot(
            se.n_sample, se.energy_variance_naive_mean, marker=marker, color=color, lw=1, ls=":", alpha=0.55,
            label=f"{ens_name} (naive, ignores correlations)",
        )
        if ref_anchor is None:
            ref_anchor = se.energy_variance_mean.iloc[-1]
    xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
    if ref_anchor is not None:
        _flat_reference(ax, xs, ref_anchor)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
    ax.set_ylabel(r"Var[single-shot total energy estimator]")
    ax.set_title(f"Total energy variance  --  {state_type}", fontsize=11)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"energy_variance_vs_nsample_{state_type}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=6, help="system size (number of qubits/sites, must be even)")
    parser.add_argument("--w", type=float, default=1.0, help="hopping coupling (Kokail et al. Eq. 1)")
    parser.add_argument("--m", type=float, default=0.9, help="staggered mass (Kokail et al. Eq. 1)")
    parser.add_argument("--g", type=float, default=1.0, help="gauge/electric-field coupling (Kokail et al. Eq. 1)")
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
        help="N_sample checkpoints to evaluate the running-mean estimator at",
    )
    parser.add_argument(
        "--num-repeats", type=int, default=5,
        help="independent measurement-noise repeats, for error bars (state and Hamiltonian terms are fixed, not redrawn)",
    )
    parser.add_argument(
        "--state-types", nargs="+", default=["ground_state", "neel"],
        choices=["ground_state", "neel", "haar_random"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_5/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-draw progress lines")
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.n_samples = [10, 50, 200, 800]
        args.num_repeats = 2

    if args.n % 2 != 0:
        parser.error(f"--n must be even (the lattice Schwinger model needs vacuum boundary conditions), got {args.n}")

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp5Config(
        n=args.n, w=args.w, m=args.m, g=args.g,
        n_samples=args.n_samples,
        num_repeats=args.num_repeats,
        ensembles=build_ensembles(),
        state_types=args.state_types,
        seed=args.seed,
    )

    # Build once up front purely to report term-group counts / ground energy
    # in hyperparameters.json for provenance -- run_exp5 builds its own
    # (identical) system internally.
    preview_system = build_schwinger_system(args.n, w=args.w, m=args.m, g=args.g)
    group_counts = {g: 0 for g in TERM_GROUPS}
    for _coeff, spec in preview_system.terms:
        group_counts[classify_term(spec, args.n)] += 1

    print(
        f"Running exp_5 -> {out_dir}\n"
        f"  n={args.n}, w={args.w}, m={args.m}, g={args.g}\n"
        f"  Hamiltonian: {len(preview_system.terms)} terms {group_counts}, "
        f"ground_energy={preview_system.ground_energy:.6f}\n"
        f"  n_samples={sorted(set(args.n_samples))}, num_repeats={args.num_repeats}\n"
        f"  state_types={args.state_types}, seed={args.seed}"
    )
    t0 = time.time()
    raw_terms, raw_energy = run_exp5(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw_terms)} term rows, {len(raw_energy)} energy rows")

    summary_terms = summarize_terms(raw_terms)
    summary_energy = summarize_energy(raw_energy)
    raw_terms.to_csv(out_dir / "raw_terms.csv", index=False)
    raw_energy.to_csv(out_dir / "raw_energy.csv", index=False)
    summary_terms.to_csv(out_dir / "summary_terms.csv", index=False)
    summary_energy.to_csv(out_dir / "summary_energy.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "w": args.w,
        "m": args.m,
        "g": args.g,
        "n_samples": sorted(set(args.n_samples)),
        "num_repeats": args.num_repeats,
        "state_types": args.state_types,
        "seed": args.seed,
        "version": out_dir.name,
        "elapsed_seconds": round(elapsed, 1),
        "num_raw_term_rows": len(raw_terms),
        "num_raw_energy_rows": len(raw_energy),
        "hamiltonian_num_terms": len(preview_system.terms),
        "hamiltonian_term_group_counts": group_counts,
        "hamiltonian_identity_offset": preview_system.identity_offset,
        "hamiltonian_ground_energy": preview_system.ground_energy,
    }
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print(f"Wrote raw_terms.csv, raw_energy.csv, summary_terms.csv, summary_energy.csv, hyperparameters.json to {out_dir}")

    num_plots = 0
    for state_type in args.state_types:
        for term_group in TERM_GROUPS:
            plot_term_group(summary_terms, state_type, term_group, out_dir)
            num_plots += 3
        plot_energy_error(summary_energy, state_type, out_dir)
        plot_energy_variance(summary_energy, state_type, out_dir)
        num_plots += 2
    print(f"Wrote {num_plots} plots to {out_dir}")


if __name__ == "__main__":
    main()
