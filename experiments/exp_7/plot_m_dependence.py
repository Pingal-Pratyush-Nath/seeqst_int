#!/usr/bin/env python3
"""
================================================================================
 plot_m_dependence.py -- aggregate several exp_7 runs into metric-vs-m plots
================================================================================

exp_7.py runs at a SINGLE value of --m per invocation, for the same reason
exp_6.py's --l is single-valued per run (see exp_6.py's module docstring):
m changes the underlying random process itself (the observable generator
AND, via q=m/n / l=m, the tuned SEEQST ensembles' own sampling), so
sweeping it in-run would be a structurally different, higher-risk loop axis.

This script is the cheap, read-only way to get the across-m view anyway --
directly mirroring ``experiments/exp_6/plot_l_dependence.py``. It takes
several ALREADY-RUN results/exp_7/v<N>/ directories (same n, same
observable_type, and -- when observable_type=="exact" -- same rest_mode,
differing only in m), reads each one's hyperparameters.json + summary.csv,
and for every state type produces a metric-vs-m plot (one line per
ensemble, evaluated at the largest N_sample checkpoint shared by ALL the
given runs) -- the direct empirical counterpart to
notes/main_theorem/main_theorem_figure.pdf's panel (b), which plots the
EXACT (closed-form) log2(shadow norm) vs. m/n for Pauli, Clifford, and the
optimally-tuned Binomial family.

FLATNESS SANITY CHECK -- CONDITIONAL ON THE OBSERVABLE FAMILY
--------------------------------------------------------------------------------
Unlike exp_6 (where Pauli/Clifford/seeqst_uniform were ALL flat across l,
because every exp_6 observable had full weight n regardless of l), exp_7's
"should be flat across m" set depends on which observable family was used:

  * clifford and seeqst_uniform are ALWAYS flat: clifford's beta=2^n+1 and
    seeqst_uniform's beta=2^(n+1) for any non-identity spec, regardless of
    weight or m.

  * pauli is flat across m ONLY when total Pauli weight is pinned to n
    regardless of m -- true for observable_type="at_least" (always
    Z-padded to full weight n, exactly like exp_6) and for
    observable_type="exact" with rest_mode="z" (ditto). It is NOT flat
    under rest_mode="identity" (weight is exactly m, so pauli's cost
    3^m genuinely grows with m -- that IS Theorem 1 / Corollary 5's own
    comparison) or rest_mode="random" (expected weight is (n+m)/2, again
    trending with m). Flagging pauli's trend as a bug under those two
    modes would be wrong -- it is the point of the plot, not a bug signal.

This script computes the right SHOULD_BE_FLAT set automatically from the
given runs' own hyperparameters.json (observable_type / rest_mode), rather
than hard-coding one set the way plot_l_dependence.py could.

USAGE
--------------------------------------------------------------------------------
    python plot_m_dependence.py v1 v2 v3 v4
    python plot_m_dependence.py --out-dir results/exp_7/m_dependence v1 v2 v3

Each positional argument is either a directory name under results/exp_7/
(e.g. "v1") or a full/relative path to a results directory containing
hyperparameters.json + summary.csv.

OUTPUT
--------------------------------------------------------------------------------
Writes, to --out-dir (default results/exp_7/m_dependence/ -- NOT
version-numbered, since this is a read-only DERIVED view recomputed from
already-versioned raw data each time it is run, so overwriting it on every
invocation is safe and intended):
  rmse_vs_m_<state>.png
  max_error_vs_m_<state>.png
one pair per state type present in the given runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.exp_7.exp_7 import BASE_RESULTS_DIR, ENSEMBLE_STYLE

FLATNESS_WARN_THRESHOLD = 0.5  # relative spread (max-min)/mean above which we print a NOTE
ALWAYS_FLAT = ("clifford", "seeqst_uniform")


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    p2 = BASE_RESULTS_DIR / arg
    if p2.is_dir():
        return p2
    raise FileNotFoundError(
        f"could not find a results directory for {arg!r} "
        f"(tried {p} and {p2})"
    )


def load_runs(run_args: list[str]) -> list[dict]:
    """Load (m, n, observable_type, rest_mode, states, summary_df) for each
    given run directory, sorted by m."""
    runs = []
    for arg in run_args:
        run_dir = _resolve_run_dir(arg)
        hp_path = run_dir / "hyperparameters.json"
        summary_path = run_dir / "summary.csv"
        if not hp_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"{run_dir} is missing hyperparameters.json or summary.csv")
        hyperparameters = json.loads(hp_path.read_text())
        summary = pd.read_csv(summary_path)
        runs.append(
            {
                "dir": run_dir,
                "m": hyperparameters["m"],
                "n": hyperparameters["n"],
                "observable_type": hyperparameters["observable_type"],
                "rest_mode": hyperparameters.get("rest_mode"),
                "states": hyperparameters["states"],
                "summary": summary,
            }
        )

    ns = {r["n"] for r in runs}
    if len(ns) > 1:
        raise ValueError(f"runs have different n (must match to compare across m): {ns}")
    obs_types = {r["observable_type"] for r in runs}
    if len(obs_types) > 1:
        raise ValueError(
            f"runs have different observable_type (must match to compare across m): {obs_types}"
        )
    if obs_types == {"exact"}:
        rest_modes = {r["rest_mode"] for r in runs}
        if len(rest_modes) > 1:
            raise ValueError(
                f"runs have different rest_mode (must match to compare across m "
                f"under observable_type=exact): {rest_modes}"
            )
    ms = [r["m"] for r in runs]
    if len(set(ms)) != len(ms):
        raise ValueError(f"duplicate m value across the given runs: {ms}")

    runs.sort(key=lambda r: r["m"])
    return runs


def should_be_flat_set(runs: list[dict]) -> tuple[str, ...]:
    """The set of ensembles expected to be ~constant across m for these
    runs (see module docstring): clifford/seeqst_uniform always, plus pauli
    too iff total Pauli weight is pinned to n regardless of m."""
    observable_type = runs[0]["observable_type"]
    rest_mode = runs[0]["rest_mode"]
    pauli_is_flat = (observable_type == "at_least") or (rest_mode == "z")
    return ALWAYS_FLAT + (("pauli",) if pauli_is_flat else ())


def shared_state_types(runs: list[dict]) -> list[str]:
    common = set(runs[0]["states"])
    for r in runs[1:]:
        common &= set(r["states"])
    canonical = ["haar_random", "ghz", "random_stabilizer"]
    return [s for s in canonical if s in common]


def shared_max_n_sample(runs: list[dict]) -> int:
    """Largest n_sample checkpoint present in EVERY given run's summary.csv."""
    per_run_max = [int(r["summary"].n_sample.max()) for r in runs]
    shared = set(runs[0]["summary"].n_sample.unique())
    for r in runs[1:]:
        shared &= set(r["summary"].n_sample.unique())
    if not shared:
        raise ValueError(
            f"the given runs share no common n_sample checkpoint "
            f"(per-run max n_sample values: {per_run_max}) -- rerun exp_7.py "
            f"with matching --n-samples across the m's you want to compare"
        )
    return max(shared)


def build_m_curves(runs: list[dict], state_type: str, n_sample: int, metric_col: str) -> dict:
    """{ensemble_name: (list of m, list of metric value)}, m's in ascending order."""
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for r in runs:
        sub = r["summary"]
        row = sub[
            (sub.state_type == state_type) & (sub.n_sample == n_sample)
        ]
        for _, rec in row.iterrows():
            xs, ys = curves.setdefault(rec.ensemble, ([], []))
            xs.append(r["m"])
            ys.append(rec[metric_col])
    return curves


def print_flatness_check(runs: list[dict], state_types: list[str], n_sample: int) -> None:
    flat_set = should_be_flat_set(runs)
    print(
        f"\nFlatness sanity check ({', '.join(flat_set)} should be ~constant across m, "
        f"at n_sample={n_sample}, given observable_type={runs[0]['observable_type']!r} "
        f"rest_mode={runs[0]['rest_mode']!r}):"
    )
    any_flag = False
    for state_type in state_types:
        for metric_col, metric_name in (("rmse_mean", "rmse"), ("max_error_mean", "max_error")):
            curves = build_m_curves(runs, state_type, n_sample, metric_col)
            for ens_name in flat_set:
                if ens_name not in curves:
                    continue
                _, ys = curves[ens_name]
                if not ys:
                    continue
                spread = (max(ys) - min(ys)) / (sum(ys) / len(ys))
                flag = spread > FLATNESS_WARN_THRESHOLD
                any_flag = any_flag or flag
                tag = "NOTE (unexpectedly large spread -- check for a bug)" if flag else "ok"
                print(
                    f"  state={state_type:<18} metric={metric_name:<10} ensemble={ens_name:<24} "
                    f"relative spread across m = {spread:.3f}  [{tag}]"
                )
    if not any_flag:
        print("  all within threshold -- no bug signal.")


def plot_metric_vs_m(runs: list[dict], state_type: str, n_sample: int, metric_col: str,
                      ylabel: str, tag: str, out_dir: Path) -> None:
    curves = build_m_curves(runs, state_type, n_sample, metric_col)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ens_name, (marker, color) in ENSEMBLE_STYLE.items():
        if ens_name not in curves:
            continue
        xs, ys = curves[ens_name]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, marker=marker, color=color, label=ens_name, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$m$  (X/Y-locality: exact value, or floor under at\_least)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. $m$  --  state = {state_type}  (at $N_{{sample}}$={n_sample})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_vs_m_{state_type}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "runs", nargs="+",
        help="results/exp_7/v<N>/ directory names (e.g. v1 v2 v3) or paths, one per m value",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="where to write the plots (default: results/exp_7/m_dependence/)",
    )
    args = parser.parse_args()

    runs = load_runs(args.runs)
    state_types = shared_state_types(runs)
    n_sample = shared_max_n_sample(runs)
    n = runs[0]["n"]
    ms = [r["m"] for r in runs]

    print(
        f"Aggregating {len(runs)} exp_7 run(s) at n={n}, m={ms}, "
        f"observable_type={runs[0]['observable_type']!r}, rest_mode={runs[0]['rest_mode']!r}"
    )
    print(f"Shared state types: {state_types}")
    print(f"Using n_sample={n_sample} (largest checkpoint shared by all given runs)")

    out_dir = Path(args.out_dir) if args.out_dir else (BASE_RESULTS_DIR / "m_dependence")
    out_dir.mkdir(parents=True, exist_ok=True)

    for state_type in state_types:
        plot_metric_vs_m(runs, state_type, n_sample, "rmse_mean", "RMSE", "rmse", out_dir)
        plot_metric_vs_m(runs, state_type, n_sample, "max_error_mean", "Max error", "max_error", out_dir)
    print(f"Wrote {2 * len(state_types)} plots to {out_dir}")

    print_flatness_check(runs, state_types, n_sample)


if __name__ == "__main__":
    main()
