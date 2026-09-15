#!/usr/bin/env python3
"""Verify ``experiments.Archived.exp_3.exp_3.circuit_for_pauli`` (lives under
``Archived/`` for folder organization only -- it's a live dependency of
exp_2/exp_4, not deprecated): for every tested Pauli
spec, check that the returned circuit U satisfies U P U^dagger = a diagonal
+-1 pattern (i.e. U really does make P exactly measurable: every conjugated
Pauli is unitarily similar to P, so its eigenvalues are still +-1 -- the
"exactly measurable" property is about U P U^dagger being DIAGONAL at all,
not about its eigenvalue magnitude. The corresponding beta(P) factor --
``SEEQSTEnsemble.inverse_weight``, 2 for pure-Z-type / 2^(n+1) for anything
with an X/Y factor -- is a separate quantity: it rescales the raw +-1
measurement outcome into an unbiased single-shot estimate of Tr(P rho), it
is not the eigenvalue of U P U^dagger itself).

Sweeps random Pauli specs (all weights, both pure-Z-type and X/Y-containing)
across n = 2..6 and asserts, for every one, that U P U^dagger is exactly
diagonal (off-diagonal Frobenius norm ~ 0) with every diagonal entry exactly
+-1 (real, unit magnitude).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qiskit.quantum_info import Operator, Pauli

from common import pauli_utils
from experiments.Archived.exp_3.exp_3 import circuit_for_pauli
from ensembles.seeqst_ensemble import SEEQSTEnsemble


def check_one(spec: dict[int, str], n: int) -> tuple[bool, float]:
    label = pauli_utils.spec_to_label(spec, n)
    P = Pauli(label).to_matrix()
    qc = circuit_for_pauli(spec, n)
    U = Operator(qc).data
    M = U @ P @ U.conj().T
    diag = np.diag(M)
    offdiag_norm = float(np.linalg.norm(M - np.diag(diag)))

    diag_ok = np.allclose(np.abs(np.real(diag)), 1.0, atol=1e-8) and np.allclose(
        np.imag(diag), 0.0, atol=1e-8
    )
    return offdiag_norm < 1e-8 and diag_ok, offdiag_norm


def main() -> None:
    rng = np.random.default_rng(0)
    # Only used to sanity-print beta(P) alongside PASS/FAIL lines below --
    # not used anywhere in building or checking the circuit itself.
    seeqst = SEEQSTEnsemble()

    num_failures = 0
    num_checked = 0
    max_offdiag = 0.0

    for n in range(2, 7):
        specs = [{}]  # identity, trivial case
        for k in range(1, n + 1):
            specs.extend(pauli_utils.random_pauli_spec(n, k, rng) for _ in range(4))
        specs.extend(pauli_utils.random_full_pauli_spec(n, rng) for _ in range(6))

        for spec in specs:
            ok, offdiag_norm = check_one(spec, n)
            num_checked += 1
            max_offdiag = max(max_offdiag, offdiag_norm)
            if not ok:
                num_failures += 1
                label = pauli_utils.spec_to_label(spec, n)
                beta = seeqst.inverse_weight(spec, n)
                print(f"FAIL  n={n} P={label!r} beta={beta} offdiag_norm={offdiag_norm:.6f}")

        print(f"n={n}: checked {len(specs)} Pauli specs")

    print()
    print(f"Total checked: {num_checked}, failures: {num_failures}, max off-diagonal norm: {max_offdiag:.2e}")
    if num_failures == 0:
        print("PASS: every derandomized circuit measures its target Pauli exactly (U P U^dagger diagonal, +-1 entries).")
    else:
        raise SystemExit(f"{num_failures} derandomized circuit(s) failed to measure their target Pauli exactly.")


if __name__ == "__main__":
    main()
