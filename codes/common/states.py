"""Test states for the shadow benchmarks: Haar-random pure states, GHZ
states, and random stabilizer states."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford, random_statevector


def haar_random_state(n: int, rng: np.random.Generator) -> Statevector:
    seed = int(rng.integers(0, 2**31 - 1))
    return random_statevector(2**n, seed=seed)


def ghz_state(n: int) -> Statevector:
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(1, n):
        qc.cx(0, q)
    return Statevector.from_label("0" * n).evolve(qc)


def random_stabilizer_state(n: int, rng: np.random.Generator) -> Statevector:
    seed = int(rng.integers(0, 2**31 - 1))
    cliff = random_clifford(n, seed=seed)
    return Statevector.from_label("0" * n).evolve(cliff)


STATE_BUILDERS = {
    "haar_random": haar_random_state,
    "ghz": lambda n, rng: ghz_state(n),
    "random_stabilizer": random_stabilizer_state,
}
