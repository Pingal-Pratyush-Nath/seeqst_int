"""Global random-Clifford classical shadows.

The "random global Clifford basis" measurement primitive from Huang, Kueng &
Preskill (main text "Example 1" / SI Sec. 5B, Eq. S16): U is drawn uniformly
from the full n-qubit Clifford group Cl(2^n). The averaged channel is the
global depolarizing twirl M(rho) = (rho + I)/(2^n+1), so
M^{-1}(P) = (2^n+1) * P for any non-identity Pauli string P (tr(P)=0 drops
the constant term). We draw genuine Clifford unitaries via
``qiskit.quantum_info.random_clifford`` (uniform Haar measure over the
Clifford group), not a generic Haar-random unitary, to stay faithful to the
ensemble analyzed in the paper.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector, random_clifford

from ensembles.base import ShadowEnsemble, Snapshot


class CliffordEnsemble(ShadowEnsemble):
    name = "clifford"

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        seed = int(rng.integers(0, 2**31 - 1))
        cliff = random_clifford(n, seed=seed)
        qc = cliff.to_circuit()
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        return Snapshot(psi_pre=psi_pre)

    def inverse_weight(self, spec: dict, n: int) -> float:
        if not spec:
            # M^{-1}(I) = I exactly (beta(I) = 1), matching
            # PauliEnsemble (3**0 == 1, implicit), SEEQSTEnsemble, and
            # SEEQSTBinomialEnsemble's explicit identity handling. Every
            # other ensemble special-cases this; this one didn't, which
            # was never triggered before exp_12 because exp_9/exp_10/
            # exp_11's synthetic observables always used min_xy>=1 (never
            # drew the identity operator). A real Hamiltonian's diagonal
            # (one-body + two-body number-operator-like) part generally
            # DOES have a nonzero identity coefficient, so without this
            # check every ensemble that includes H's identity term would
            # get it amplified by (2**n+1) instead of left alone -- a
            # constant, non-shrinking bias in any combined-operator
            # estimate that includes H's trace part.
            return 1.0
        return float(2**n + 1)

    def sample_unitary_circuit(self, n: int, rng: np.random.Generator):
        """A single random Clifford circuit. Unlike PauliEnsemble/SEEQSTEnsemble,
        the Clifford group is far too large to enumerate exactly even for
        modest n, so anything that needs "the ensemble average over U" for
        this ensemble (e.g. a Monte-Carlo shadow-norm estimate) has to sample
        many draws from this method instead of enumerating all of them."""
        seed = int(rng.integers(0, 2**31 - 1))
        return random_clifford(n, seed=seed).to_circuit()
