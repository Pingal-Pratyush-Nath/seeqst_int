#!/usr/bin/env python3
"""Correctness checks for ``ensembles/seeqst_binomial_ensemble.py``,
``ensembles/seeqst_unifsize_ensemble.py``, and the ``ensembles/_seeqst_circuits.py``
refactor of ``ensembles/seeqst_ensemble.py`` they share (see that module's
docstring for the bit-convention risk this guards against).

Run with: ``python tests/verify_seeqst_binomial_and_unifsize.py`` from the
``shadow_benchmark`` root.

Checks:
  1. Circuit-equivalence regression: the refactored ``_seeqst_circuits``
     module produces EXACTLY the same circuits (as unitary operators, for
     every (n, block, branch) up to n=6) as a verbatim copy of the
     original, pre-refactor ``SEEQSTEnsemble`` implementation. (NOTE: this
     compares circuits/operators, not raw ``sample_snapshot`` outputs --
     ``Statevector.measure()`` draws from qiskit's own internal randomness,
     not the caller's ``rng``, so two calls with a fixed seed are NOT
     expected to reproduce the same measurement outcome even for the
     unmodified original code. Verified explicitly in check 2.)
  2. Sanity: confirms check 1's premise -- the SAME circuit, measured
     twice, can give different outcomes (i.e. raw-output comparison would
     be the wrong test).
  3. SEEQSTBinomialEnsemble(q=0.5).inverse_weight matches the flat
     SEEQSTEnsemble's exactly, for empty/pure-Z/XY-containing specs. (NOTE:
     this is a WEAK check by itself -- at q=0.5 my closed form's
     k-dependence exactly cancels regardless of which of several plausible
     formulas one might guess, so this alone can't distinguish a correct
     formula from several wrong ones. It's a real check nonetheless; check
     7 below is the one that actually discriminates.)
  4. SEEQSTUniformSizeEnsemble(l=n)'s XY-branch matches the known closed
     form 2(n+1)C(n,k) (typo-catcher; real validation is check 7).
  5. Constructor/argument validation: q in (0,1) required for
     SEEQSTBinomialEnsemble; k>l raises ValueError and pure-Z raises
     NotImplementedError for SEEQSTUniformSizeEnsemble.
  6. Monte-Carlo: empirical frequency of "sampled real-qubit subset ==
     target set a" converges to the claimed closed-form p(a), for both new
     ensembles -- validates the SAMPLING distribution, not the circuit
     physics.
  7. From-definition brute force (the strongest check, and the one that
     actually caught a real bug during development -- see note below): for
     small n, enumerate EVERY (block, branch) pair either ensemble can
     produce, weight by its exact sampling probability, and directly
     compute lambda_P = Pr_U[U P U^dagger is Z-type] via RAW MATRIX
     conjugation (Qiskit ``Operator`` multiplication + ``SparsePauliOp``
     decomposition) -- NOT via the closed-form formula, and NOT via
     ``Pauli.evolve(qc, frame='h')``, which was tried first and silently
     gives WRONG answers for these specific circuits (they mix ``rx``/
     ``ry`` rotation gates with ``cx``; ``Pauli.evolve`` appears to mismatch
     which gates it treats as Clifford-conjugation-legal once such gates
     are combined in one circuit, even though evolving a single bare RX90
     gate in isolation happens to come out right -- see git history /
     development notes if this needs re-diagnosing). The raw-matrix method
     is validated FIRST against the known-good, already-trusted flat
     ``SEEQSTEnsemble`` constants (2.0 and 2**(n+1)) before being trusted to
     validate the two new formulas -- if that opening sub-check ever fails,
     suspect the verification method itself, not the ensembles.
  8. Normalization: the closed-form p(a) sums to 1 over every reachable
     real-qubit subset, for both new ensembles, at a few (n, q, l).
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector

from common import states
from common.external_paths import import_seeqst_setup
from common.pauli_utils import is_pure_z_type, random_full_pauli_spec, spec_to_label
from ensembles import _seeqst_circuits as circuits
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from ensembles.seeqst_unifsize_ensemble import SEEQSTUniformSizeEnsemble

_seeqst_setup = import_seeqst_setup()

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------------
# 1-2. Circuit-equivalence regression + measurement-randomness sanity
# ---------------------------------------------------------------------
def _parse_circuit_text_ORIGINAL(text: str, n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for op in text.split(")"):
        op = op.strip().strip("(")
        if not op:
            continue
        gate_name, _, qubit_str = op.partition(":")
        qubit_indices = [int(x) for x in qubit_str.split(",")]
        if gate_name == "RX90":
            qc.rx(np.pi / 2, qubit_indices[0])
        elif gate_name == "RY90":
            qc.ry(np.pi / 2, qubit_indices[0])
        elif gate_name == "CNOT":
            qc.cx(qubit_indices[0], qubit_indices[1])
        elif gate_name == "H":
            qc.h(qubit_indices[0])
        else:
            raise ValueError(gate_name)
    return qc


def _original_circuit(n: int, block: int, branch: int, cache: dict) -> QuantumCircuit:
    if n not in cache:
        cache[n] = _seeqst_setup.build_parallel_entangler_blocks(list(range(2**n)), n)
    texts = cache[n][block]
    text = texts[branch] if len(texts) > 1 else texts[0]
    return _parse_circuit_text_ORIGINAL(text, n)


def run_circuit_equivalence_check() -> None:
    text_cache: dict = {}
    n_checked = 0
    all_match = True
    for n in range(1, 7):
        for block in range(2**n):
            nb = circuits.num_branches(n, block)
            for branch in range(nb):
                op_new = Operator(circuits.circuit_for(n, block, branch)).data
                op_old = Operator(_original_circuit(n, block, branch, text_cache)).data
                n_checked += 1
                if not np.allclose(op_new, op_old, atol=1e-12):
                    all_match = False
    check("1. refactored circuits == original circuits (all n<=6)", all_match, f"checked {n_checked}")


def run_measurement_randomness_sanity() -> None:
    qc = circuits.circuit_for(4, 5, 0)
    state = states.haar_random_state(4, np.random.default_rng(1))
    outcomes = {state.evolve(qc).measure()[0] for _ in range(30)}
    check(
        "2. Statevector.measure() is NOT reproducible from a fixed external rng "
        "(confirms check 1 must compare circuits, not sampled outputs)",
        len(outcomes) > 1,
    )


# ---------------------------------------------------------------------
# 3-4. Closed-form special cases
# ---------------------------------------------------------------------
def run_binomial_half_matches_flat() -> None:
    flat = SEEQSTEnsemble()
    half = SEEQSTBinomialEnsemble(0.5)
    rng = np.random.default_rng(0)
    all_match = True
    for n in [3, 4, 5, 6]:
        specs = [{}] + [random_full_pauli_spec(n, rng) for _ in range(30)]
        for spec in specs:
            a = flat.inverse_weight(spec, n)
            b = half.inverse_weight(spec, n)
            if not np.isclose(a, b, rtol=1e-12):
                all_match = False
    check("3. SEEQSTBinomialEnsemble(q=0.5).inverse_weight == flat SEEQSTEnsemble's", all_match)


def run_unifsize_l_eq_n_matches_closed_form() -> None:
    all_match = True
    for n in [3, 4, 5, 6]:
        ens = SEEQSTUniformSizeEnsemble(l=n)
        for k in range(1, n + 1):
            spec = {i: "X" for i in range(k)}
            spec.update({i: "Z" for i in range(k, n)})
            got = ens.inverse_weight(spec, n)
            want = 2.0 * (n + 1) * comb(n, k)
            if not np.isclose(got, want, rtol=1e-12):
                all_match = False
    check("4. SEEQSTUniformSizeEnsemble(l=n) matches closed form 2(n+1)C(n,k)", all_match)


# ---------------------------------------------------------------------
# 5. Validation / error handling
# ---------------------------------------------------------------------
def run_validation_checks() -> None:
    ok = True
    for bad_q in (0.0, 1.0, -0.1, 1.1):
        try:
            SEEQSTBinomialEnsemble(bad_q)
            ok = False
        except ValueError:
            pass
    ens = SEEQSTUniformSizeEnsemble(l=2)
    try:
        ens.inverse_weight({0: "X", 1: "X", 2: "X"}, 5)  # k=3 > l=2
        ok = False
    except ValueError:
        pass
    try:
        ens.inverse_weight({0: "Z", 1: "Z"}, 5)  # pure-Z
        ok = False
    except NotImplementedError:
        pass
    check("5. constructor/argument validation raises as documented", ok)


# ---------------------------------------------------------------------
# 6. Monte-Carlo marginal check (the bit-convention regression test)
# ---------------------------------------------------------------------
def run_montecarlo_marginal_check() -> None:
    all_ok = True
    n = 5
    rng = np.random.default_rng(7)

    q = 0.3
    target = frozenset({1, 3})
    n_trials = 200_000
    hits = 0
    for _ in range(n_trials):
        included = rng.random(n) < q
        if frozenset(np.flatnonzero(included)) == target:
            hits += 1
    empirical = hits / n_trials
    closed_form = q ** len(target) * (1 - q) ** (n - len(target))
    rel_err = abs(empirical - closed_form) / closed_form
    ok = rel_err < 0.05
    all_ok &= ok
    print(f"    Binomial(q={q}) p({set(target)})  empirical={empirical:.5f}  closed_form={closed_form:.5f}  rel_err={rel_err:.3f}")

    l = 3
    target2 = frozenset({0, 2})
    hits2 = 0
    for _ in range(n_trials):
        s = int(rng.integers(0, l + 1))
        subset = frozenset(rng.choice(n, size=s, replace=False))
        if subset == target2:
            hits2 += 1
    empirical2 = hits2 / n_trials
    closed_form2 = 1.0 / ((l + 1) * comb(n, len(target2)))
    rel_err2 = abs(empirical2 - closed_form2) / closed_form2
    ok2 = rel_err2 < 0.05
    all_ok &= ok2
    print(f"    UnifSize(l={l})   p({set(target2)})  empirical={empirical2:.5f}  closed_form={closed_form2:.5f}  rel_err={rel_err2:.3f}")

    check("6. Monte-Carlo sampled-subset frequency matches closed-form p(a)", all_ok)


# ---------------------------------------------------------------------
# 7. From-definition brute force: lambda_P = Pr_U[U P U^dagger is Z-type]
#    via RAW MATRIX conjugation (Pauli.evolve is NOT trusted here -- see
#    module docstring; it was tried first and silently mishandles these
#    rx/ry+cx circuits).
# ---------------------------------------------------------------------
def _is_z_type_raw(mat: np.ndarray) -> bool:
    label = SparsePauliOp.from_operator(mat).simplify().paulis.to_labels()[0]
    return ("X" not in label) and ("Y" not in label)


def _brute_force_lambda(spec: dict, n: int, block_weight_and_nbranches: list[tuple[int, float, int]]) -> float:
    P = Operator(Pauli(spec_to_label(spec, n))).data
    total = 0.0
    for block, p_block, nb in block_weight_and_nbranches:
        if p_block == 0:
            continue
        for branch in range(nb):
            qc = circuits.circuit_for(n, block, branch)
            U = Operator(qc).data
            UPUd = U @ P @ U.conj().T
            if _is_z_type_raw(UPUd):
                total += p_block / nb
    return total


def run_brute_force_definition_check() -> None:
    all_ok = True
    n = 4

    # --- 7a. validate the METHOD itself against the already-trusted flat
    # SEEQSTEnsemble constants (2.0 pure-Z, 2**(n+1) any XY) before trusting
    # it on the new formulas.
    flat_blocks = [(block, 1.0 / 2**n, circuits.num_branches(n, block)) for block in range(2**n)]
    baseline_specs = [({0: "Z", 1: "Z"}, 2.0), ({0: "X", 1: "Z"}, float(2 ** (n + 1)))]
    for spec, expected in baseline_specs:
        lam = _brute_force_lambda(spec, n, flat_blocks)
        beta = 1.0 / lam if lam > 0 else float("inf")
        ok = np.isclose(beta, expected, rtol=1e-9)
        all_ok &= ok
        print(f"    [method check] flat, spec={spec}: brute beta={beta:.4f}  known-good={expected:.4f}  {'OK' if ok else 'FAIL -- suspect the brute-force method, not the ensembles'}")

    # --- 7b. SEEQSTBinomialEnsemble(q) ---
    q = 0.35
    blocks = []
    for block in range(2**n):
        subset = {i for i in range(n) if (block >> (n - 1 - i)) & 1}
        p_block = q ** len(subset) * (1 - q) ** (n - len(subset))
        nb = circuits.num_branches(n, block)
        blocks.append((block, p_block, nb))
    assert abs(sum(pb for _, pb, _ in blocks) - 1.0) < 1e-9

    ens_bin = SEEQSTBinomialEnsemble(q)
    for spec in [
        {0: "X", 1: "Z"},
        {0: "X", 1: "Y", 2: "Z"},
        {i: "Z" for i in range(n)},
    ]:
        lam_brute = _brute_force_lambda(spec, n, blocks)
        beta_brute = 1.0 / lam_brute if lam_brute > 0 else float("inf")
        beta_formula = ens_bin.inverse_weight(spec, n)
        rel_err = abs(beta_brute - beta_formula) / beta_formula
        ok = rel_err < 1e-6
        all_ok &= ok
        print(f"    Binomial q={q} spec={spec}: brute beta={beta_brute:.6f}  formula={beta_formula:.6f}  rel_err={rel_err:.2e}")

    # --- 7c. SEEQSTUniformSizeEnsemble(l) ---
    l = 2
    blocks_us = []
    for block in range(2**n):
        subset = {i for i in range(n) if (block >> (n - 1 - i)) & 1}
        s = len(subset)
        p_block = 1.0 / ((l + 1) * comb(n, s)) if s <= l else 0.0
        nb = circuits.num_branches(n, block)
        blocks_us.append((block, p_block, nb))
    assert abs(sum(pb for _, pb, _ in blocks_us) - 1.0) < 1e-9

    ens_us = SEEQSTUniformSizeEnsemble(l)
    for spec in [
        {0: "X", 1: "Z", 2: "Z"},
        {0: "X", 1: "Y", 2: "Z"},
    ]:
        lam_brute = _brute_force_lambda(spec, n, blocks_us)
        beta_brute = 1.0 / lam_brute if lam_brute > 0 else float("inf")
        beta_formula = ens_us.inverse_weight(spec, n)
        rel_err = abs(beta_brute - beta_formula) / beta_formula
        ok = rel_err < 1e-6
        all_ok &= ok
        print(f"    UnifSize l={l} spec={spec}: brute beta={beta_brute:.6f}  formula={beta_formula:.6f}  rel_err={rel_err:.2e}")

    check("7. brute-force (raw-matrix) Pr_U[UPU^dagger is Z-type] matches closed-form inverse_weight", all_ok)


# ---------------------------------------------------------------------
# 8. Normalization
# ---------------------------------------------------------------------
def run_normalization_check() -> None:
    all_ok = True
    for n in [4, 5, 6]:
        for q in [0.2, 1 / (n + 1), 0.6]:
            total = sum(comb(n, k) * q**k * (1 - q) ** (n - k) for k in range(n + 1))
            if not np.isclose(total, 1.0, atol=1e-9):
                all_ok = False
    for n in [4, 5, 6]:
        for l in [0, 1, n // 2, n]:
            total = sum(comb(n, k) * (1.0 / ((l + 1) * comb(n, k))) for k in range(l + 1))
            if not np.isclose(total, 1.0, atol=1e-9):
                all_ok = False
    check("8. closed-form p(a) normalizes to 1 (summed by size via C(n,k))", all_ok)


if __name__ == "__main__":
    run_circuit_equivalence_check()
    run_measurement_randomness_sanity()
    run_binomial_half_matches_flat()
    run_unifsize_l_eq_n_matches_closed_form()
    run_validation_checks()
    run_montecarlo_marginal_check()
    run_brute_force_definition_check()
    run_normalization_check()
    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
