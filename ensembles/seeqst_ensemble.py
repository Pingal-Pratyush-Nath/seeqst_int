"""SEEQST classical shadows (flat/uniform-over-all-subsets distribution).

The SEEQST unitary ensemble S = {U_{I,E}, U_{I,O}}_{I subseteq [n]}
(Definition 3, ``papers/SEEQST_shadows.pdf``) partitions the n qubits into a
subset I, which gets entangled into a GHZ-type state and measured in one of
two conjugate "GHZ-even" / "GHZ-odd" bases, while the remaining qubits are
measured directly in the computational (Z) basis. This is the FLAT special
case: I is drawn UNIFORMLY among all 2^n subsets (independent of the branch
bit), so every one of the 2^{n+1} unitaries (2 branches x 2^n subsets) is
chosen with equal probability 1/2^{n+1}. For the general (non-uniform,
tunable) sampling-distribution version, see ``seeqst_binomial_ensemble.py``
and ``seeqst_unifsize_ensemble.py`` -- both reuse the SAME underlying
circuit-construction machinery as this class, factored out into
``ensembles/_seeqst_circuits.py`` (gate parsing, per-``(n,block,branch)``
circuit caching, and the real-qubit-subset <-> integer ``block`` conversion
that ``build_parallel_entangler_blocks`` expects) so that all three
SEEQST-family ensembles are guaranteed to agree on which real qubits a given
circuit entangles. See that module's docstring for why this matters and
what it does NOT protect against.

Inverse map. Proposition 9 / 11 of the SEEQST draft shows the averaged
channel M is diagonal in the Pauli basis with

    alpha_P = 1                     if P = I^{otimes n}
    alpha_P = 1/2                   if P != I^{otimes n} and P in {I,Z}^n
    alpha_P = 1/2^{n+1}             otherwise (P has an X or Y factor)

so M^{-1}(P) = beta(P) P with beta(P) = 1/alpha_P, i.e. beta = 2 for
non-trivial pure-Z-type strings and beta = 2^{n+1} for anything containing
an X or Y. Note the explicit, and rather severe, dependence on the full
system size n for any observable that is not pure-Z-type -- unlike the local
Pauli ensemble's dimension-independent 3^k. This is exactly the q=1/2
special case of ``seeqst_binomial_ensemble.SEEQSTBinomialEnsemble``'s
inverse map (see that module for the general closed form) --
``tests/verify_seeqst_binomial_and_unifsize.py`` checks the two agree
exactly.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from common.pauli_utils import is_pure_z_type
from ensembles import _seeqst_circuits as _circuits
from ensembles.base import ShadowEnsemble, Snapshot


class SEEQSTEnsemble(ShadowEnsemble):
    name = "seeqst"

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        subset = int(rng.integers(0, 2**n))
        branch = int(rng.integers(0, _circuits.num_branches(n, subset)))
        qc = _circuits.circuit_for(n, subset, branch)
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        return Snapshot(psi_pre=psi_pre)

    def inverse_weight(self, spec: dict, n: int) -> float:
        if not spec:
            return 1.0
        if is_pure_z_type(spec):
            return 2.0
        return float(2 ** (n + 1))

    def enumerate_unitaries(self, n: int) -> list[tuple[float, QuantumCircuit]]:
        """ALL 2^{n+1} SEEQST ensemble members (2 branches x 2^n subsets),
        each with weight 1/2^{n+1}. Finite and small enough to enumerate
        exactly for modest n. For subset=I=empty, the E and O branches
        coincide (both are the identity circuit -- see the SEEQST draft's
        Definition 3), so that single circuit is returned once with double
        weight (2/2^{n+1}) rather than being built twice."""
        weight = 1.0 / (2 ** (n + 1))
        out: list[tuple[float, QuantumCircuit]] = []
        for subset in range(2**n):
            nb = _circuits.num_branches(n, subset)
            if nb == 2:
                out.append((weight, _circuits.circuit_for(n, subset, 0)))
                out.append((weight, _circuits.circuit_for(n, subset, 1)))
            else:
                out.append((2 * weight, _circuits.circuit_for(n, subset, 0)))
        return out
