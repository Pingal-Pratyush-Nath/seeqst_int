#!/usr/bin/env python3
"""
================================================================================
 exp_7 -- Locality-tuned SEEQST: exact-m vs. at-least-m X/Y observables
================================================================================

WHAT THIS EXPERIMENT DOES
--------------------------------------------------------------------------------
exp_6 benchmarked Pauli / Clifford / three SEEQST variants on observables
whose X/Y-weight was drawn from a wide, uninformative range {1,...,l}, all
measured with ONE fixed SEEQST-binomial q=1/(n+1) -- and found
seeqst_binomial can badly underperform there (see
results/exp_6/v0/why_binomial_underperforms.md). The theory note written
since (notes/main_theorem/main_theorem.tex, Theorem 1 / Corollary 5) explains
why: the Binomial family's shadow norm is only EXACTLY optimal, and only
provably beats Pauli and Clifford, when q is tuned to a SINGLE, FIXED, KNOWN
X/Y-locality m (q* = m/n).

exp_7 is the direct empirical test of that theorem. It compares SIX
ensembles --

    pauli                     PauliEnsemble()                       (beta = 3^weight(P))
    clifford                  CliffordEnsemble()                    (beta = 2^n+1, exact)
    seeqst_uniform             SEEQSTEnsemble()                       (flat/uniform SEEQST, q=1/2)
    seeqst_binomial_tuned      SEEQSTBinomialEnsemble(q=m/n)          (Theorem 1's optimal q)
    seeqst_binomial_untuned    SEEQSTBinomialEnsemble(q=1/(n+1))      (exp_6's baseline q, for contrast)
    seeqst_unifsize_tuned      SEEQSTUniformSizeEnsemble(l=...)       (capped-uniform-size -- l=m under
                                                                       "exact", l=the upper cap under
                                                                       "at_least"; see note below)

-- on TWO selectable observable families (``--observable-type``), both
built around a single locality parameter ``m`` (``--m``):

    "exact"     X/Y-weight is EXACTLY m always
                (pauli_utils.random_exact_xy_spec(n, m, rng, rest_mode)).
                This is the regime Theorem 1 / Corollary 5 are stated for.
                The non-XY qubits' Z-vs-I split is controlled by
                ``--rest-mode`` {random, z, identity} (default "random" --
                see pauli_utils.random_exact_xy_spec's docstring for why:
                every SEEQST ensemble's cost is PROVABLY independent of this
                split by the general eigenvalue theorem, while Pauli's own
                cost is NOT, since Z counts toward its weight and I doesn't
                -- "random" is the only mode that makes that contrast
                visible in the data, rather than only ever exercising one
                fixed b).

    "at_least"  X/Y-weight is drawn uniformly from {m, ..., l} (l an upper
                cap, ``--l``, default min(m+2, n)) and every other qubit is
                Z (pauli_utils.random_bounded_xy_spec(n, l, rng,
                min_xy=m) -- already supports this, no new generator
                needed). seeqst_binomial_tuned stays tuned to the FLOOR m,
                not the true random weight -- deliberately: this is a
                controlled "we only know a lower bound on the locality"
                robustness study, one notch more forgiving than exp_6's
                fully uninformative range, and it shows how quickly
                performance DEGRADES as the true weight exceeds what q was
                tuned for (the same beta(k) exponential-blowup mechanism
                documented in why_binomial_underperforms.md, now with a
                controllable floor instead of no floor at all).

                seeqst_unifsize_tuned CANNOT be tuned to the floor m the
                same way, and this is not a design choice -- its
                inverse_weight raises ValueError outright (p(a)=0, a
                singular channel, not merely high variance) for any spec
                whose xy_weight exceeds its own l. Under "at_least" it is
                therefore tuned to the CAP l instead, exactly mirroring how
                exp_6 tuned its own seeqst_unifsize_l to the generator's
                cap. This is itself a vivid, concrete illustration of the
                general eigenvalue theorem's "p(a)=0 => no unbiased
                estimator" clause: unlike the Binomial family, which always
                degrades gracefully, uniform-subset-size simply cannot
                serve an unknown-above-m weight at all unless its own cap
                is raised to cover it.

Every generated observable stays inside the support of every tested
ensemble (xy_weight >= 1 = m >= 1 always; seeqst_unifsize_tuned's own cap
is m under "exact" and the generator's own l under "at_least", so it can
never see a weight above what it was built to handle) -- same invariant
exp_6 maintained.

The two-layer randomization design, running-mean estimator, and per-ensemble
snapshot-stream reuse are IDENTICAL to exp_1/exp_6 -- see exp_1.py's module
docstring for the full rationale.

METRICS -- ONLY TWO (no mean_variance), same as exp_6
--------------------------------------------------------------------------------
    RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
    MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
averaged over BOTH randomization layers (every outer state x every inner
observable-set draw, pooled together).

WHAT YOU CAN CONFIGURE
--------------------------------------------------------------------------------
  --n                       system size (number of qubits)
  --m                       the swept hyperparameter: exact X/Y-locality
                            ("exact" mode) or floor on it ("at_least" mode).
                            Must satisfy 1<=m<=n-1 (needed for the tuned
                            q*=m/n to lie strictly in (0,1)).
  --observable-type         "exact" or "at_least" (default "exact")
  --rest-mode               "random" | "z" | "identity" (default "random");
                            only used when --observable-type=exact
  --l                       upper cap on X/Y-weight for "at_least" mode
                            (default: min(m+2, n)); ignored (with a printed
                            note) for "exact" mode
  --q-binomial-untuned      seeqst_binomial_untuned's q (default: 1/(n+1),
                            exp_6's own default -- the tuned ensemble's q is
                            always exactly m/n, not independently settable)
  --num-observables         M, number of random observables per draw
  --n-samples               list of N_sample checkpoints to evaluate at
  --num-state-repeats       outer layer: how many random states to draw
  --num-observable-repeats  inner layer: how many random observable sets to
                            draw per state
  --states                  which of haar_random / ghz / random_stabilizer to run
  --seed                    RNG seed, for reproducibility
  --version                 output subfolder name (default: auto-incrementing
                            v0, v1, v2, ...)
  --quick                   small/fast configuration for smoke-testing
  --quiet                   suppress the per-state-draw progress lines

OUTPUT
--------------------------------------------------------------------------------
Each run is saved to its own versioned subfolder results/exp_7/v<N>/:
  hyperparameters.json                  all settings used for this run
  raw.csv        one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample)
  summary.csv    the above averaged over both randomization layers
  rmse_vs_nsample_<state>.png       )  one figure per state type,
  max_error_vs_nsample_<state>.png  )  one metric per figure, 6 ensemble lines

To see how the metrics depend on m itself (holding n, observable_type, and
n_sample fixed), run this script several times at different --m and pass
the resulting results/exp_7/v<N>/ directories to
``experiments/exp_7/plot_m_dependence.py`` -- the empirical companion to
notes/main_theorem/main_theorem_figure.pdf's panel (b).

USAGE
--------------------------------------------------------------------------------
    python exp_7.py --quick
    python exp_7.py --n 6 --m 2 --observable-type exact --rest-mode random
    python exp_7.py --n 6 --m 2 --observable-type at_least --l 4
    python exp_7.py --n 7 --m 3 --version my_run_name   # explicit output folder name
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives two levels below the shadow_benchmark root (same as
# every other exp_*.py), so we add that root to sys.path ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from common.versioning import next_version_dir
from ensembles.clifford_ensemble import CliffordEnsemble
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from ensembles.seeqst_unifsize_ensemble import SEEQSTUniformSizeEnsemble
from tasks.exp_7_tuned_locality import Exp7Config, run_exp7, summarize

# Results always live under the shadow_benchmark root's results/ folder
# (NOT experiments/exp_7/results/), same convention as every other exp_*.py.
BASE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp_7"

METRICS = [
    ("rmse_mean", "rmse_std", "RMSE", "rmse", "inv_sqrt_n"),
    ("max_error_mean", "max_error_std", "Max error", "max_error", "inv_sqrt_n"),
]
ENSEMBLE_STYLE = {
    "pauli": ("o", "tab:blue"),
    "clifford": ("s", "tab:orange"),
    "seeqst_uniform": ("^", "tab:green"),
    "seeqst_binomial_tuned": ("D", "tab:red"),
    "seeqst_binomial_untuned": ("P", "tab:pink"),
    "seeqst_unifsize_tuned": ("v", "tab:purple"),
}


def build_ensembles(n: int, m: int, l_for_unifsize: int, q_untuned: float) -> dict:
    """``l_for_unifsize`` is m under observable_type="exact" (true weight is
    always exactly m) but must be the observable generator's own upper cap
    l under "at_least" (true weight can exceed m there, and
    SEEQSTUniformSizeEnsemble.inverse_weight raises -- not degrades -- once
    weight exceeds its l) -- see exp_7.py's module docstring, "at_least"
    paragraph, for the full explanation. seeqst_binomial_tuned's q=m/n, by
    contrast, is ALWAYS tuned to the floor m regardless of observable_type,
    since the Binomial family degrades gracefully rather than failing."""
    q_tuned = m / n
    return {
        "pauli": PauliEnsemble(),
        "clifford": CliffordEnsemble(),
        "seeqst_uniform": SEEQSTEnsemble(),
        "seeqst_binomial_tuned": SEEQSTBinomialEnsemble(q_tuned),
        "seeqst_binomial_untuned": SEEQSTBinomialEnsemble(q_untuned),
        "seeqst_unifsize_tuned": SEEQSTUniformSizeEnsemble(l_for_unifsize),
    }


def plot_state(summary, state_type: str, out_dir: Path) -> None:
    sub = summary[summary.state_type == state_type]
    if sub.empty:
        return
    for mean_col, std_col, ylabel, tag, ref_style in METRICS:
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
        if ref_style == "inv_sqrt_n" and ref_anchor is not None:
            x0, y0 = ref_anchor
            xs = np.array(sorted(sub.n_sample.unique()), dtype=float)
            ref = y0 * np.sqrt(x0 / xs)
            ax.plot(xs, ref, "k--", alpha=0.4, lw=1, label=r"$\propto 1/\sqrt{N_{sample}}$ (reference)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of shadow snapshots  $N_{sample}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. $N_{{sample}}$  --  state = {state_type}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_vs_nsample_{state_type}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=6, help="system size (number of qubits)")
    parser.add_argument(
        "--m", type=int, default=2,
        help="swept hyperparameter: exact X/Y-locality (exact mode) or floor on it "
        "(at_least mode); must satisfy 1<=m<=n-1",
    )
    parser.add_argument(
        "--observable-type", type=str, default="exact", choices=["exact", "at_least"],
        help="'exact': X/Y-weight is exactly m. 'at_least': X/Y-weight ~ Uniform{m,...,l}",
    )
    parser.add_argument(
        "--rest-mode", type=str, default="random", choices=["random", "z", "identity"],
        help="only used when --observable-type=exact: how the n-m non-XY qubits are "
        "set ('random': iid Z/I per qubit: default. 'z': all Z, exp_6's convention. "
        "'identity': all I, main_theorem.tex's convention)",
    )
    parser.add_argument(
        "--l", type=int, default=None,
        help="upper cap on X/Y-weight for --observable-type=at_least (default: "
        "min(m+2, n)); ignored for --observable-type=exact",
    )
    parser.add_argument(
        "--q-binomial-untuned", type=float, default=None,
        help="per-qubit Bernoulli inclusion probability for seeqst_binomial_untuned "
        "(default: 1/(n+1), exp_6's own default)",
    )
    parser.add_argument("--num-observables", type=int, default=10, help="M, number of random observables")
    parser.add_argument(
        "--n-samples", type=int, nargs="+",
        default=[10, 30, 100, 300, 1000, 3000, 10000],
        help="N_sample checkpoints to evaluate the running-mean estimator at",
    )
    parser.add_argument(
        "--num-state-repeats", type=int, default=3,
        help="outer layer: how many random states to draw",
    )
    parser.add_argument(
        "--num-observable-repeats", type=int, default=3,
        help="inner layer: how many random observable sets to draw per state",
    )
    parser.add_argument(
        "--states", nargs="+", default=["haar_random", "ghz", "random_stabilizer"],
        choices=["haar_random", "ghz", "random_stabilizer"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version", type=str, default=None,
        help="output subfolder name under results/exp_7/ (default: auto-incrementing v0, v1, ...)",
    )
    parser.add_argument("--quick", action="store_true", help="small/fast configuration for smoke-testing")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-state-draw progress lines from the outer randomization layer",
    )
    args = parser.parse_args()

    if args.quick:
        args.n = 4
        args.m = 2
        args.num_observables = 5
        args.n_samples = [10, 50, 200, 800]
        args.num_state_repeats = 2
        args.num_observable_repeats = 2

    if not (1 <= args.m <= args.n - 1):
        parser.error(f"--m must satisfy 1 <= m <= n-1 (got m={args.m}, n={args.n})")

    if args.observable_type == "exact":
        if args.l is not None:
            print(f"Note: --l={args.l} is ignored (--observable-type=exact uses weight exactly m)")
        l_effective = args.m
    else:
        l_effective = args.l if args.l is not None else min(args.m + 2, args.n)
        if not (args.m <= l_effective <= args.n):
            parser.error(
                f"--l must satisfy m <= l <= n for --observable-type=at_least "
                f"(got l={l_effective}, m={args.m}, n={args.n})"
            )

    q_untuned = args.q_binomial_untuned if args.q_binomial_untuned is not None else 1.0 / (args.n + 1)
    # seeqst_unifsize_tuned's cap: m itself under "exact" (weight is always
    # exactly m there); the generator's own upper cap under "at_least"
    # (weight can exceed m there, and unifsize raises rather than degrades
    # if its cap is set below the true weight -- see build_ensembles' docstring).
    l_for_unifsize = args.m if args.observable_type == "exact" else l_effective

    out_dir = (BASE_RESULTS_DIR / args.version) if args.version else next_version_dir(BASE_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Exp7Config(
        n=args.n,
        m=args.m,
        observable_type=args.observable_type,
        rest_mode=args.rest_mode,
        l=l_effective,
        num_observables=args.num_observables,
        n_samples=args.n_samples,
        num_state_repeats=args.num_state_repeats,
        num_observable_repeats=args.num_observable_repeats,
        ensembles=build_ensembles(args.n, args.m, l_for_unifsize, q_untuned),
        state_types=args.states,
        seed=args.seed,
    )

    print(
        f"Running exp_7 -> {out_dir}\n"
        f"  n={args.n}, m={args.m}, observable_type={args.observable_type!r}, "
        f"rest_mode={args.rest_mode!r}, l={l_effective}, l_for_unifsize={l_for_unifsize}\n"
        f"  q_tuned={args.m / args.n:.6g}, q_untuned={q_untuned:.6g}, "
        f"M={args.num_observables}, n_samples={sorted(set(args.n_samples))}\n"
        f"  num_state_repeats={args.num_state_repeats}, num_observable_repeats={args.num_observable_repeats}\n"
        f"  states={args.states}, seed={args.seed}"
    )
    t0 = time.time()
    raw = run_exp7(config, verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -- {len(raw)} rows")

    summary = summarize(raw)
    raw.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hyperparameters = {
        "n": args.n,
        "m": args.m,
        "observable_type": args.observable_type,
        "rest_mode": args.rest_mode,
        "l": l_effective,
        "l_for_unifsize": l_for_unifsize,
        "q_tuned": args.m / args.n,
        "q_untuned": q_untuned,
        "num_observables": args.num_observables,
        "n_samples": sorted(set(args.n_samples)),
        "num_state_repeats": args.num_state_repeats,
        "num_observable_repeats": args.num_observable_repeats,
        "states": args.states,
        "seed": args.seed,
        "version": out_dir.name,
        "elapsed_seconds": round(elapsed, 1),
        "num_raw_rows": len(raw),
    }
    with open(out_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    print(f"Wrote raw.csv, summary.csv, hyperparameters.json to {out_dir}")

    for state_type in args.states:
        plot_state(summary, state_type, out_dir)
    print(f"Wrote {len(METRICS) * len(args.states)} plots to {out_dir}")


if __name__ == "__main__":
    main()
