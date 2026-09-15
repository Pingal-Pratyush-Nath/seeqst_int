#!/usr/bin/env python3
"""
================================================================================
 plot_l_dependence.py -- aggregate several exp_6 runs into metric-vs-l plots
================================================================================

exp_6.py runs at a SINGLE value of --l per invocation (see exp_6.py's module
docstring for why: l changes the underlying random process itself -- both
the observable generator and the seeqst_unifsize_l ensemble's sampling
depend on it -- so sweeping it in-run would be a structurally different,
higher-risk loop axis, not worth stacking onto exp_6's core run loop).

This script is the cheap, read-only way to get the across-l view anyway: it
takes several ALREADY-RUN results/exp_6/v<N>/ directories (same n and
state_types, differing only in l), reads each one's hyperparameters.json +
summary.csv, and for every state type produces a metric-vs-l plot (one line
per ensemble, evaluated at the largest N_sample checkpoint shared by ALL the
given runs) -- the direct empirical counterpart to the theory note's
(``QIP_notes/SEEQST_shadows_threshold.tex``) phase-diagram figure, which
plots asymptotic shadow-norm rate vs. weight fraction f=k/n.

SANITY CHECK THIS SCRIPT PRINTS
--------------------------------------------------------------------------------
Under exp_6's observable family, three of the five ensembles' inverse_weight
does NOT depend on the X/Y-weight k at all (every observable here has FULL
weight n, so Pauli's beta=3^k=3^n and Clifford's beta=2^n+1 are both
constant regardless of l; SEEQST-uniform's beta=2^(n+1) for any
non-identity spec, also regardless of l) -- so pauli / clifford /
seeqst_uniform's curves should come out visually (and numerically) FLAT
across l. Only seeqst_binomial and seeqst_unifsize_l actually vary with l.
This script computes, for each of those three "should be flat" ensembles,
the relative spread (max-min)/mean of its metric value across the given l's,
and prints a NOTE if that spread exceeds a threshold -- a large systematic
trend there would signal a bug (e.g. a stale snapshot cache mixing runs, or
an inverse_weight regression), not real physics.

USAGE
--------------------------------------------------------------------------------
    python plot_l_dependence.py v1 v2 v3 v4
    python plot_l_dependence.py --out-dir results/exp_6/l_dependence v1 v2 v3

Each positional argument is either a directory name under results/exp_6/
(e.g. "v1") or a full/relative path to a results directory containing
hyperparameters.json + summary.csv.

OUTPUT
--------------------------------------------------------------------------------
Writes, to --out-dir (default results/exp_6/l_dependence/ -- NOT
version-numbered like exp_6.py's own output, since this is a read-only
DERIVED view recomputed from already-versioned raw data each time it is
run, so overwriting it on every invocation is safe and intended):
  rmse_vs_l_<state>.png
  max_error_vs_l_<state>.png
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

from experiments.exp_6.exp_6 import BASE_RESULTS_DIR, ENSEMBLE_STYLE

FLATNESS_WARN_THRESHOLD = 0.5  # relative spread (max-min)/mean above which we print a NOTE
SHOULD_BE_FLAT = ("pauli", "clifford", "seeqst_uniform")


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
    """Load (l, n, states, summary_df) for each given run directory, sorted by l."""
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
                "l": hyperparameters["l"],
                "n": hyperparameters["n"],
                "states": hyperparameters["states"],
                "summary": summary,
            }
        )

    ns = {r["n"] for r in runs}
    if len(ns) > 1:
        raise ValueError(f"runs have different n (must match to compare across l): {ns}")
    ls = [r["l"] for r in runs]
    if len(set(ls)) != len(ls):
        raise ValueError(f"duplicate l value across the given runs: {ls}")

    runs.sort(key=lambda r: r["l"])
    return runs


def shared_state_types(runs: list[dict]) -> list[str]:
    common = set(runs[0]["states"])
    for r in runs[1:]:
        common &= set(r["states"])
    # preserve a stable, canonical order rather than set() iteration order
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
            f"(per-run max n_sample values: {per_run_max}) -- rerun exp_6.py "
            f"with matching --n-samples across the l's you want to compare"
        )
    return max(shared)


def build_l_curves(runs: list[dict], state_type: str, n_sample: int, metric_col: str) -> dict:
    """{ensemble_name: (list of l, list of metric value)}, l's in ascending order."""
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for r in runs:
        sub = r["summary"]
        row = sub[
            (sub.state_type == state_type) & (sub.n_sample == n_sample)
        ]
        for _, rec in row.iterrows():
            xs, ys = curves.setdefault(rec.ensemble, ([], []))
            xs.append(r["l"])
            ys.append(rec[metric_col])
    return curves


def print_flatness_check(runs: list[dict], state_types: list[str], n_sample: int) -> None:
    print(
        f"\nFlatness sanity check (pauli / clifford / seeqst_uniform should be "
        f"~constant across l, at n_sample={n_sample}):"
    )
    any_flag = False
    for state_type in state_types:
        for metric_col, metric_name in (("rmse_mean", "rmse"), ("max_error_mean", "max_error")):
            curves = build_l_curves(runs, state_type, n_sample, metric_col)
            for ens_name in SHOULD_BE_FLAT:
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
                    f"  state={state_type:<18} metric={metric_name:<10} ensemble={ens_name:<15} "
                    f"relative spread across l = {spread:.3f}  [{tag}]"
                )
    if not any_flag:
        print("  all within threshold -- no bug signal.")


def plot_metric_vs_l(runs: list[dict], state_type: str, n_sample: int, metric_col: str,
                      ylabel: str, tag: str, out_dir: Path) -> None:
    curves = build_l_curves(runs, state_type, n_sample, metric_col)
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
    ax.set_xlabel(r"$l$  (max number of X/Y factors per observable)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. $l$  --  state = {state_type}  (at $N_{{sample}}$={n_sample})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_vs_l_{state_type}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "runs", nargs="+",
        help="results/exp_6/v<N>/ directory names (e.g. v1 v2 v3) or paths, one per l value",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="where to write the plots (default: results/exp_6/l_dependence/)",
    )
    args = parser.parse_args()

    runs = load_runs(args.runs)
    state_types = shared_state_types(runs)
    n_sample = shared_max_n_sample(runs)
    n = runs[0]["n"]
    ls = [r["l"] for r in runs]

    print(f"Aggregating {len(runs)} exp_6 run(s) at n={n}, l={ls}")
    print(f"Shared state types: {state_types}")
    print(f"Using n_sample={n_sample} (largest checkpoint shared by all given runs)")

    out_dir = Path(args.out_dir) if args.out_dir else (BASE_RESULTS_DIR / "l_dependence")
    out_dir.mkdir(parents=True, exist_ok=True)

    for state_type in state_types:
        plot_metric_vs_l(runs, state_type, n_sample, "rmse_mean", "RMSE", "rmse", out_dir)
        plot_metric_vs_l(runs, state_type, n_sample, "max_error_mean", "Max error", "max_error", out_dir)
    print(f"Wrote {2 * len(state_types)} plots to {out_dir}")

    print_flatness_check(runs, state_types, n_sample)


if __name__ == "__main__":
    main()
