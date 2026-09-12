"""SEEQST classical shadows under the product ("Binomial") sampling
distribution: each qubit is independently included in the disentangled
subset I with probability q, i.e. p(I) = q^|I| (1-q)^(n-|I|).

This is the general-p SEEQST construction (Definition 2.3 of
``SEEQST_shadows_main_results.pdf``) specialised to a product measure --
see ``QIP_notes/SEEQST_shadows_threshold.tex`` Definition 3 for the same
family, where q=1/2 is called the "base" distribution (recovering
``seeqst_ensemble.SEEQSTEnsemble`` exactly) and q=1/(n+1) is the source
material's own low-weight-tuned construction.

Inverse map, from the general eigenvalue theorem (Theorem 3.3 /
Prop. 4.2 of the same source; lambda_{a,b} = (1/2) p(a) for a Pauli string
with X/Y-support a != 0, and lambda_{0,b} = 1/2 + (1/2) p_hat(b) for a
pure-Z-type string of support b, where p_hat is the Walsh-Hadamard
transform of p): because p here is a PRODUCT measure, both quantities have
closed forms --

    p(a)     = q^|a| (1-q)^(n-|a|)     (probability of the SPECIFIC set a)
    p_hat(b) = (1-2q)^|b|              (standard product-measure WHT)

so

    beta(P) = 1                                     P = identity
    beta(P) = 1 / (1/2 + 1/2 (1-2q)^j)               P pure-Z-type, weight j
    beta(P) = 2 q^{-k} (1-q)^{-(n-k)}                P has X/Y-support size k > 0

At q=1/2, (1-2q)=0 and this reduces EXACTLY to ``SEEQSTEnsemble``'s
(1.0, 2.0, 2**(n+1)) on all three branches --
``tests/verify_seeqst_binomial_and_unifsize.py`` checks this directly.
Sampling reuses ``ensembles._seeqst_circuits`` (see that module's docstring
for the real-qubit-subset <-> circuit-index convention).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from common.pauli_utils import is_pure_z_type, xy_weight
from ensembles import _seeqst_circuits as _circuits
from ensembles.base import ShadowEnsemble, Snapshot


class SEEQSTBinomialEnsemble(ShadowEnsemble):
    name = "seeqst_binomial"

    def __init__(self, q: float) -> None:
        if not (0.0 < q < 1.0):
            raise ValueError(
                f"q={q} must be strictly between 0 and 1 (q=0 or 1 makes any "
                "X/Y-containing observable's inverse weight blow up: q**-k "
                "divides by zero for k>0)."
            )
        self.q = float(q)

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        included = rng.random(n) < self.q  # independent Bernoulli(q) per qubit
        subset_qubits = np.flatnonzero(included)
        block = _circuits.qubit_subset_to_block(subset_qubits, n)
        branch = int(rng.integers(0, _circuits.num_branches(n, block)))
        qc = _circuits.circuit_for(n, block, branch)
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        return Snapshot(psi_pre=psi_pre)

    def inverse_weight(self, spec: dict, n: int) -> float:
        if not spec:
            return 1.0
        q = self.q
        if is_pure_z_type(spec):
            j = len(spec)
            return 1.0 / (0.5 + 0.5 * (1 - 2 * q) ** j)
        k = xy_weight(spec)
        return 2.0 * q ** (-k) * (1 - q) ** (-(n - k))
