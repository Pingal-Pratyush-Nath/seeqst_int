"""Common interface for classical-shadow measurement ensembles.

All three ensembles studied here (local random Pauli / single-qubit
Clifford, global random Clifford, and SEEQST) share one crucial structural
feature: their averaged measurement channel M is diagonal in the n-qubit
Pauli basis, so M^{-1}(P) = beta(P) * P for a scalar beta(P) that depends
only on the ensemble and (for SEEQST) on which Pauli P is. Combined with the
self-adjointness of M^{-1} (tr(X M^{-1}(Y)) = tr(M^{-1}(X) Y)), this means a
single-shot shadow estimate of tr(P * rho) can always be written as

    o_hat = beta(P) * <psi_pre | P | psi_pre>,      psi_pre = U^dagger |b>

where U is the sampled unitary and b the measured bitstring. This lets us
estimate expectation values for many different Pauli observables from a
*single* sampled snapshot (one U, one measurement) without ever forming the
explicit (exponentially large) shadow matrix rho_hat.

Subclasses implement ``sample_snapshot`` (draw U, measure, return enough
information to build psi_pre) and ``inverse_weight`` (beta(P)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit.quantum_info import Pauli, Statevector

from common.pauli_utils import spec_to_label


@dataclass
class Snapshot:
    """One measurement round: the post-measurement pre-image state
    psi_pre = U^dagger |b>, i.e. the state whose Pauli expectation values
    give tr(P * U^dagger |b><b| U) directly via Statevector.expectation_value.
    """

    psi_pre: Statevector


class ShadowEnsemble:
    name: str = "base"

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        raise NotImplementedError

    def inverse_weight(self, spec: dict, n: int) -> float:
        """beta(P) such that M^{-1}(P) = beta(P) * P."""
        raise NotImplementedError

    def estimate_one_shot(
        self,
        state: Statevector,
        pauli_specs: Iterable[dict],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw one shadow snapshot and return o_hat_i = tr(P_i rho_hat) for
        every Pauli spec in pauli_specs (all from the SAME snapshot, as is
        standard practice to amortize the cost of state preparation)."""
        snap = self.sample_snapshot(state, n, rng)
        out = np.empty(len(pauli_specs))
        for i, spec in enumerate(pauli_specs):
            label = spec_to_label(spec, n)
            expval = snap.psi_pre.expectation_value(Pauli(label)).real
            out[i] = self.inverse_weight(spec, n) * expval
        return out

    def estimate_many_shots(
        self,
        state: Statevector,
        pauli_specs: Iterable[dict],
        n: int,
        num_shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Shape (num_shots, num_paulis) array of single-shot estimates.

        NOTE: this draws a FRESH batch of ``num_shots`` snapshots every time
        it's called. If you need to evaluate several DIFFERENT sets of Pauli
        observables against the SAME underlying measurement data for a given
        state (the whole point of classical shadows -- one batch of
        measurements can be reused to predict any number of different
        observables), draw the snapshots once with ``sample_snapshots`` and
        reuse them across observable sets via ``evaluate_snapshots`` instead
        of calling this method repeatedly.
        """
        pauli_specs = list(pauli_specs)
        out = np.empty((num_shots, len(pauli_specs)))
        for s in range(num_shots):
            out[s] = self.estimate_one_shot(state, pauli_specs, n, rng)
        return out

    def sample_snapshots(
        self, state: Statevector, n: int, num_shots: int, rng: np.random.Generator
    ) -> list[Snapshot]:
        """Draw ``num_shots`` independent classical-shadow snapshots for
        ``state``, with NO reference to any particular observable. This is
        the "physical measurement" step; the resulting list can (and should)
        be reused with ``evaluate_snapshots`` to estimate as many different
        Pauli observables / observable sets as you like without re-measuring.
        """
        return [self.sample_snapshot(state, n, rng) for _ in range(num_shots)]

    def evaluate_snapshots(
        self, snapshots: list[Snapshot], pauli_specs: Iterable[dict], n: int
    ) -> np.ndarray:
        """Shape (len(snapshots), num_paulis) array of single-shot estimates
        o_hat_i^(s) = beta(P_i) * <psi_pre^(s) | P_i | psi_pre^(s)>, computed
        by reusing an ALREADY-DRAWN list of snapshots (from
        ``sample_snapshots``) -- no new measurements are drawn here.
        """
        pauli_specs = list(pauli_specs)
        out = np.empty((len(snapshots), len(pauli_specs)))
        for s, snap in enumerate(snapshots):
            for i, spec in enumerate(pauli_specs):
                label = spec_to_label(spec, n)
                expval = snap.psi_pre.expectation_value(Pauli(label)).real
                out[s, i] = self.inverse_weight(spec, n) * expval
        return out
