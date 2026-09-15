#!/usr/bin/env python3
"""Correctness checks for exp_8 -- the ceiling-tuned sparse-m X/Y observable
family (see ``experiments/exp_8/exp_8.py``).

Run with: ``python tests/verify_exp8_observables.py`` from the
``shadow_benchmark`` root.

exp_8 reuses ``common.pauli_utils.random_bounded_xy_spec`` (already tested
generically) and the same two ensemble CLASSES exp_6/exp_7 already validate
(``SEEQSTBinomialEnsemble``, ``SEEQSTUniformSizeEnsemble``) with different
constructor arguments (q=m/n, l=m). So the checks here focus on:

  1. The observable family itself: weight ~ Uniform{1,...,m} (never above
     m), Z-padded, full weight n -- structurally, and via a chi-square-style
     uniformity check on the weight histogram.
  2. seeqst_unifsize_tuned(l=m) NEVER raises ValueError on this family --
     the concrete difference from exp_7's "at_least" mode flagged in
     exp_8.py's ``build_ensembles`` docstring (true weight can never exceed
     the tuned cap here, by construction of the generator).
  3. THE SCIENTIFIC POINT exp_8 is built to test, checked directly and
     numerically (not just asserted) over the FULL range 1<=m<=n-1 -- not
     just a few small-m examples, since an earlier draft of this argument
     was wrong for m>n/2 and only caught by sweeping the whole range: the
     minimax-optimal q (found by direct numerical minimization of
     max_{k=1}^m beta_q(k) over q, with NO reference to the closed-form
     formula) matches the TWO-REGIME piecewise prediction from exp_8.py's
     module docstring --
         q* = m/n,  value = 2^(1+n H2(m/n))   for m <= n/2
         q* = 1/2,  value = 2^(n+1)           for m >  n/2
     -- to high precision in both regimes, including the m=n/2 boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize_scalar

from common.pauli_utils import random_bounded_xy_spec, xy_weight, is_pure_z_type
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_unifsize_ensemble import SEEQSTUniformSizeEnsemble

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------------
# 1. weight in {1,...,m} always, full weight n, non-XY qubits all Z
# ---------------------------------------------------------------------
def run_weight_bounded_check() -> None:
    rng = np.random.default_rng(0)
    ok = True
    for n in (5, 8, 12):
        for m in range(1, n):
            for _ in range(30):
                spec = random_bounded_xy_spec(n, m, rng, min_xy=1)
                k = xy_weight(spec)
                if not (1 <= k <= m):
                    ok = False
                if len(spec) != n:  # Z-padded to full weight
                    ok = False
                non_xy = [letter for letter in spec.values() if letter not in ("X", "Y")]
                if not all(letter == "Z" for letter in non_xy):
                    ok = False
    check("1. xy_weight(spec) in {1,...,m} always, full weight n, rest all Z", ok)


# ---------------------------------------------------------------------
# 2. weight is (roughly) uniform over {1,...,m}, not concentrated at m
#    (distinguishes this family from exp_7's "exact" family, which always
#    has weight == m)
# ---------------------------------------------------------------------
def run_weight_uniformity_check() -> None:
    rng = np.random.default_rng(1)
    n, m = 10, 5
    num_draws = 20000
    counts = np.zeros(m + 1, dtype=int)  # counts[k] for k=1..m
    for _ in range(num_draws):
        spec = random_bounded_xy_spec(n, m, rng, min_xy=1)
        counts[xy_weight(spec)] += 1
    observed = counts[1:]
    expected = num_draws / m
    max_rel_dev = float(np.max(np.abs(observed - expected)) / expected)
    ok = max_rel_dev < 0.10  # generous tolerance, this is a sanity check not a rigorous test
    check(
        "2. weight ~ Uniform{1,...,m}, not concentrated at m",
        ok, f"counts={observed.tolist()}, expected~{expected:.0f} each, max rel dev={max_rel_dev:.3f}"
    )


# ---------------------------------------------------------------------
# 3. seeqst_unifsize_tuned(l=m) never raises ValueError on this family
#    (unlike exp_7's "at_least" mode, where true weight can exceed the
#    floor m and unifsize must be widened to the generator's own cap
#    instead -- see exp_8.py::build_ensembles docstring)
# ---------------------------------------------------------------------
def run_unifsize_never_raises_check() -> None:
    rng = np.random.default_rng(2)
    ok = True
    error_detail = ""
    for n in (5, 8, 12):
        for m in range(1, n):
            ensemble = SEEQSTUniformSizeEnsemble(m)
            for _ in range(30):
                spec = random_bounded_xy_spec(n, m, rng, min_xy=1)
                try:
                    ensemble.inverse_weight(spec, n)
                except (ValueError, NotImplementedError) as e:
                    ok = False
                    error_detail = f"n={n} m={m} spec={spec}: {e}"
                    break
            if not ok:
                break
        if not ok:
            break
    check(
        "3. SEEQSTUniformSizeEnsemble(l=m).inverse_weight never raises on exp_8's family",
        ok, error_detail
    )


# ---------------------------------------------------------------------
# 4. THE SCIENTIFIC POINT, full two-regime version: the minimax-optimal q
#    for max_{k=1}^m beta_q(k), found PURELY NUMERICALLY (no reference to
#    the closed-form argmin), matches
#        q* = m/n,  value = 2^(1+n H2(m/n))   for m <= n/2
#        q* = 1/2,  value = 2^(n+1)           for m >  n/2
#    (exp_8.py module docstring, "WHY q=m/n IS RIGHT HERE ONLY FOR m<=n/2").
#    Swept over the FULL range 1<=m<=n-1, not just a few small-m examples --
#    an earlier draft of this argument claimed q*=m/n unconditionally and
#    was only caught wrong by testing m>n/2 here.
# ---------------------------------------------------------------------
def beta(q: float, k: int, n: int) -> float:
    return 2.0 * q ** (-k) * (1.0 - q) ** (-(n - k))


def max_beta_over_k(q: float, m: int, n: int) -> float:
    return max(beta(q, k, n) for k in range(1, m + 1))


def h2(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def run_q_star_numerical_check() -> None:
    ok = True
    detail_lines = []
    for n in (6, 7, 10, 11, 20, 30):
        for m in range(1, n):
            res = minimize_scalar(
                lambda q: max_beta_over_k(q, m, n),
                bounds=(1e-9, 1 - 1e-9), method="bounded", options={"xatol": 1e-12},
            )
            q_num, val_num = res.x, res.fun
            if 2 * m <= n:
                q_theory, val_theory = m / n, 2.0 ** (1 + n * h2(m / n))
            else:
                q_theory, val_theory = 0.5, 2.0 ** (n + 1)
            rel_err_val = abs(val_num - val_theory) / val_theory
            # near m=n/2 the two branches meet and q is only weakly
            # identified (the objective is flat there), so gate on the
            # OPTIMAL VALUE matching, not q itself, near that boundary --
            # everywhere else require both to agree tightly.
            near_boundary = abs(2 * m - n) <= 1
            this_ok = rel_err_val < 1e-6 and (near_boundary or abs(q_num - q_theory) < 1e-4)
            ok = ok and this_ok
            detail_lines.append(
                f"n={n} m={m} (2m<=n:{2*m<=n}): q_num={q_num:.6f} q_theory={q_theory:.6f} "
                f"val_num={val_num:.6g} val_theory={val_theory:.6g} rel_err_val={rel_err_val:.2e}"
            )
    check(
        "4. numerically-minimized minimax q/value matches the two-regime "
        "piecewise formula (q*=m/n for m<=n/2, q*=1/2 for m>n/2) across "
        "the full range 1<=m<=n-1",
        ok, " | ".join(detail_lines)
    )
    if not ok:
        for line in detail_lines:
            print(f"       {line}")


# ---------------------------------------------------------------------
# 5. seeqst_binomial_tuned's beta at q=m/n, evaluated directly through
#    SEEQSTBinomialEnsemble (not the bare formula), agrees with the bare
#    formula and is exactly what's used by the max-over-list argument
# ---------------------------------------------------------------------
def run_ensemble_matches_formula_check() -> None:
    rng = np.random.default_rng(3)
    n, m = 9, 4
    q_tuned = m / n
    ensemble = SEEQSTBinomialEnsemble(q_tuned)
    ok = True
    for _ in range(50):
        spec = random_bounded_xy_spec(n, m, rng, min_xy=1)
        k = xy_weight(spec)
        got = ensemble.inverse_weight(spec, n)
        expected = beta(q_tuned, k, n)
        if not np.isclose(got, expected):
            ok = False
    check("5. SEEQSTBinomialEnsemble(q=m/n).inverse_weight matches the bare beta_q(k) formula", ok)


if __name__ == "__main__":
    run_weight_bounded_check()
    run_weight_uniformity_check()
    run_unifsize_never_raises_check()
    run_q_star_numerical_check()
    run_ensemble_matches_formula_check()
    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
