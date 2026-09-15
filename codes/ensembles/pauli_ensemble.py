"""Local random-Pauli / single-qubit-Clifford classical shadows.

This is the main protocol of Huang, Kueng & Preskill, "Predicting Many
Properties of a Quantum System from Very Few Measurements"
(arXiv:2002.08953), and the one implemented operationally in
``codes/predicting-quantum-properties``. Each qubit independently gets a
uniformly random measurement basis from {X, Y, Z}. The averaged channel is
M = M_1^{otimes n} with M_1(rho) = (rho + I)/3 (single-qubit depolarizing
twirl), so M^{-1}(P) = 3^k P for any weight-k Pauli string P (SI Example 2 /
Eq. S17). Because the single-qubit rotation into the sampled basis exactly
diagonalizes matching Pauli factors and leaves mismatched ones off-diagonal,
the generic ``<psi_pre|P|psi_pre>`` machinery in ``ensembles.base`` already
reproduces the textbook "hit or miss" estimator without any extra logic:
tr(P rho_hat) = 3^k * (product of measured signs) if every qubit in
support(P) happened to be measured in the matching Pauli basis, and 0
otherwise.
"""

from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ensembles.base import ShadowEnsemble, Snapshot

_BASES = ("X", "Y", "Z")


def _basis_circuit(bases: list[str], n: int) -> QuantumCircuit:
    """Circuit that rotates each qubit's sampled Pauli basis into Z
    (qubit q's gate acts on qiskit qubit index q, consistent with the
    rightmost-char-is-qubit-0 convention used throughout)."""
    qc = QuantumCircuit(n)
    for q, b in enumerate(bases):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        # 'Z' -> identity, nothing to add
    return qc


class PauliEnsemble(ShadowEnsemble):
    name = "pauli"

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        bases = list(rng.choice(_BASES, size=n))
        qc = _basis_circuit(bases, n)
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        return Snapshot(psi_pre=psi_pre)

    def inverse_weight(self, spec: dict, n: int) -> float:
        return 3.0 ** len(spec)

    def enumerate_unitaries(self, n: int) -> list[tuple[float, QuantumCircuit]]:
        """ALL 3^n local-basis choices, each with weight 1/3^n. The ensemble
        is finite and small enough to enumerate exactly for modest n, which
        is what lets the shadow-norm notebook compute an EXACT (rather than
        Monte-Carlo-sampled) operator average for this ensemble."""
        weight = 1.0 / (3**n)
        return [(weight, _basis_circuit(list(bases), n)) for bases in itertools.product(_BASES, repeat=n)]
