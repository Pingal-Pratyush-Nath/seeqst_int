#!/usr/bin/env python3
"""Correctness checks for ``common/pauli_utils.py::random_exact_xy_spec``,
the new generator added for exp_7's "exact-m" observable family (see
``experiments/exp_7/exp_7.py``).

Run with: ``python tests/verify_exp7_observables.py`` from the
``shadow_benchmark`` root.

No new ensemble CLASSES were added for exp_7 -- it reuses
``SEEQSTBinomialEnsemble`` and ``SEEQSTUniformSizeEnsemble`` with different
constructor arguments (q=m/n, l=m), and both were already validated by
``tests/verify_seeqst_binomial_and_unifsize.py`` for arbitrary q/l. So the
checks here focus on the one genuinely new piece of logic: the observable
generator itself, plus the specific scientific claim exp_7 is built to
show (SEEQST's cost is b-independent, Pauli's isn't).

Checks:
  1. For every rest_mode in {"random", "z", "identity"} and several (n, m):
     xy_weight(spec) == m always, and every one of those m positions has a
     letter in {X, Y}.
  2. rest_mode="z": total weight len(spec) == n always (every non-XY qubit
     present, as Z) -- i.e. this mode structurally matches exp_6's
     Z-padding convention restricted to weight exactly m.
  3. rest_mode="identity": total weight len(spec) == m always (no non-XY
     qubit appears in the spec at all) -- main_theorem.tex's convention.
  4. rest_mode="random": Monte-Carlo -- across many draws, the fraction of
     non-XY qubits that come out Z should be close to 1/2.
  5. Constructor/argument validation: m outside [0, n] and an unknown
     rest_mode both raise ValueError.
  6. Edge cases m=0 and m=n (the generator itself is more permissive than
     exp_7.py's CLI-level 1<=m<=n-1) still produce structurally valid specs.
  7. THE SCIENTIFIC POINT, checked directly: at fixed n, m (rest_mode=
     "random"), draw many specs and evaluate both PauliEnsemble's and
     SEEQSTBinomialEnsemble(q=m/n)'s inverse_weight on each. SEEQST's value
     must be IDENTICAL across every draw (b-independence, exact); Pauli's
     value must NOT be constant across draws (it depends on the random
     Z/I split too) -- this is the empirical contrast exp_7's "exact" +
     "random" configuration is designed to make visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.pauli_utils import random_exact_xy_spec, xy_weight
from ensembles.pauli_ensemble import PauliEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------------
# 1. xy_weight is exactly m, for every rest_mode
# ---------------------------------------------------------------------
def run_xy_weight_exact_check() -> None:
    rng = np.random.default_rng(0)
    ok = True
    for rest_mode in ("random", "z", "identity"):
        for n in (4, 6, 8):
            for m in range(1, n):
                for _ in range(20):
                    spec = random_exact_xy_spec(n, m, rng, rest_mode=rest_mode)
                    if xy_weight(spec) != m:
                        ok = False
                    xy_letters = [letter for letter in spec.values() if letter in ("X", "Y")]
                    if len(xy_letters) != m:
                        ok = False
    check("1. xy_weight(spec) == m for every draw, all rest_modes", ok)


# ---------------------------------------------------------------------
# 2. rest_mode="z" -> total weight == n
# ---------------------------------------------------------------------
def run_rest_z_full_weight_check() -> None:
    rng = np.random.default_rng(1)
    ok = True
    for n in (4, 6, 8):
        for m in range(1, n):
            for _ in range(20):
                spec = random_exact_xy_spec(n, m, rng, rest_mode="z")
                if len(spec) != n:
                    ok = False
                non_xy_letters = [letter for letter in spec.values() if letter not in ("X", "Y")]
                if not all(letter == "Z" for letter in non_xy_letters):
                    ok = False
                if len(non_xy_letters) != n - m:
                    ok = False
    check("2. rest_mode='z': total weight == n, every non-XY letter is Z", ok)


# ---------------------------------------------------------------------
# 3. rest_mode="identity" -> total weight == m
# ---------------------------------------------------------------------
def run_rest_identity_weight_check() -> None:
    rng = np.random.default_rng(2)
    ok = True
    for n in (4, 6, 8):
        for m in range(1, n):
            for _ in range(20):
                spec = random_exact_xy_spec(n, m, rng, rest_mode="identity")
                if len(spec) != m:
                    ok = False
                if any(letter not in ("X", "Y") for letter in spec.values()):
                    ok = False
    check("3. rest_mode='identity': total weight == m, no Z letters at all", ok)


# ---------------------------------------------------------------------
# 4. rest_mode="random" -> ~half of the non-XY qubits are Z (Monte Carlo)
# ---------------------------------------------------------------------
def run_rest_random_fraction_check() -> None:
    rng = np.random.default_rng(3)
    n, m = 8, 3
    num_draws = 4000
    z_count = 0
    total_rest = 0
    for _ in range(num_draws):
        spec = random_exact_xy_spec(n, m, rng, rest_mode="random")
        non_xy_letters = [letter for letter in spec.values() if letter not in ("X", "Y")]
        z_count += len(non_xy_letters)
        total_rest += n - m
    frac = z_count / total_rest
    ok = abs(frac - 0.5) < 0.03
    check(
        "4. rest_mode='random': empirical P(Z | non-XY qubit) ~ 1/2",
        ok, f"got {frac:.4f} over {total_rest} non-XY qubit draws (n={n}, m={m})"
    )


# ---------------------------------------------------------------------
# 5. Argument validation
# ---------------------------------------------------------------------
def run_validation_checks() -> None:
    rng = np.random.default_rng(4)
    raised_m_low = raised_m_high = raised_bad_mode = False
    try:
        random_exact_xy_spec(5, -1, rng)
    except ValueError:
        raised_m_low = True
    try:
        random_exact_xy_spec(5, 6, rng)
    except ValueError:
        raised_m_high = True
    try:
        random_exact_xy_spec(5, 2, rng, rest_mode="bogus")
    except ValueError:
        raised_bad_mode = True
    check("5a. m < 0 raises ValueError", raised_m_low)
    check("5b. m > n raises ValueError", raised_m_high)
    check("5c. unknown rest_mode raises ValueError", raised_bad_mode)


# ---------------------------------------------------------------------
# 6. Edge cases m=0 and m=n
# ---------------------------------------------------------------------
def run_edge_case_checks() -> None:
    rng = np.random.default_rng(5)
    n = 5
    ok = True
    for rest_mode in ("random", "z", "identity"):
        spec0 = random_exact_xy_spec(n, 0, rng, rest_mode=rest_mode)
        if xy_weight(spec0) != 0:
            ok = False
        specn = random_exact_xy_spec(n, n, rng, rest_mode=rest_mode)
        if xy_weight(specn) != n or len(specn) != n:
            ok = False
        if any(letter not in ("X", "Y") for letter in specn.values()):
            ok = False
    check("6. m=0 and m=n edge cases produce structurally valid specs", ok)


# ---------------------------------------------------------------------
# 7. THE SCIENTIFIC POINT: SEEQST cost is b-independent, Pauli's isn't
# ---------------------------------------------------------------------
def run_b_independence_contrast_check() -> None:
    rng = np.random.default_rng(6)
    n, m = 9, 4
    q_tuned = m / n
    seeqst = SEEQSTBinomialEnsemble(q_tuned)
    pauli = PauliEnsemble()

    seeqst_betas = []
    pauli_betas = []
    for _ in range(300):
        spec = random_exact_xy_spec(n, m, rng, rest_mode="random")
        seeqst_betas.append(seeqst.inverse_weight(spec, n))
        pauli_betas.append(pauli.inverse_weight(spec, n))

    seeqst_constant = np.allclose(seeqst_betas, seeqst_betas[0])
    expected_seeqst_beta = 2.0 * q_tuned ** (-m) * (1 - q_tuned) ** (-(n - m))
    seeqst_matches_formula = np.isclose(seeqst_betas[0], expected_seeqst_beta)
    pauli_varies = (max(pauli_betas) - min(pauli_betas)) > 0

    check(
        "7a. SEEQSTBinomialEnsemble(q=m/n).inverse_weight is IDENTICAL across "
        "300 random-rest draws at fixed m (b-independence)",
        seeqst_constant, f"unique values seen: {sorted(set(seeqst_betas))[:5]}"
    )
    check(
        "7b. that constant value matches the closed form 2 q^-m (1-q)^-(n-m)",
        seeqst_matches_formula, f"got {seeqst_betas[0]!r}, expected {expected_seeqst_beta!r}"
    )
    check(
        "7c. PauliEnsemble.inverse_weight VARIES across the same 300 draws "
        "(weight = m + random #Z among the rest, not b-independent)",
        pauli_varies, f"min={min(pauli_betas)}, max={max(pauli_betas)} (expected genuinely different)"
    )


if __name__ == "__main__":
    run_xy_weight_exact_check()
    run_rest_z_full_weight_check()
    run_rest_identity_weight_check()
    run_rest_random_fraction_check()
    run_validation_checks()
    run_edge_case_checks()
    run_b_independence_contrast_check()
    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
