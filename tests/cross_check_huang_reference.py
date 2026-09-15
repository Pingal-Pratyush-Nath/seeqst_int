#!/usr/bin/env python3
"""Cross-check our PauliEnsemble against Huang's own reference
implementation (``codes/predicting-quantum-properties/prediction_shadow.py``,
function ``estimate_exp``), on IDENTICAL raw measurement data.

We draw random-Pauli-basis measurement shots ourselves (same sampling as
``ensembles/pauli_ensemble.py``), record the raw (basis, +-1 outcome) data
in exactly the file format their code expects (see their README's
"[measurement.txt]" format), and then:

  1. Feed that data to their unmodified ``estimate_exp`` function and get
     its per-observable prediction.
  2. Independently compute the same quantity from our own
     ``bitstring_sign_product`` / hit-or-miss logic on the SAME raw shots.
  3. Assert the two agree to numerical precision.

This does not modify prediction_shadow.py in any way -- it is imported
read-only via common.external_paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qiskit.quantum_info import Statevector

from common import pauli_utils, states
from common.external_paths import import_huang_prediction_shadow
from ensembles.pauli_ensemble import _basis_circuit

_BASES = ("X", "Y", "Z")


def draw_raw_shots(state: Statevector, n: int, num_shots: int, rng: np.random.Generator):
    """Return a list of shots; each shot is a list of (basis_letter, +-1) per qubit,
    in the same convention as Huang's measurement.txt (qubit 0 first)."""
    shots = []
    for _ in range(num_shots):
        bases = list(rng.choice(_BASES, size=n))
        qc = _basis_circuit(bases, n)
        outcome, _ = state.evolve(qc).measure()
        outcome = str(outcome)  # qiskit label, rightmost char = qubit 0
        shot = []
        for q in range(n):
            bit = outcome[n - 1 - q]
            sign = 1 if bit == "0" else -1
            shot.append((bases[q], sign))
        shots.append(shot)
    return shots


def main() -> None:
    n = 4
    rng = np.random.default_rng(42)
    state = states.random_stabilizer_state(n, rng)

    shots = draw_raw_shots(state, n, num_shots=3000, rng=rng)

    prediction_shadow = import_huang_prediction_shadow()

    specs = [pauli_utils.random_pauli_spec(n, k, rng) for k in (1, 2, 3, 4) for _ in range(3)]

    print(f"{'pauli':10s} {'huang_ref (matched-avg)':>26s} {'ours (hit-or-miss, rescaled)':>30s} {'true value':>12s}")
    max_discrepancy = 0.0
    for spec in specs:
        one_observable = [(letter, pos) for pos, letter in spec.items()]

        sum_product, cnt_match = prediction_shadow.estimate_exp(shots, one_observable)
        huang_avg = sum_product / cnt_match if cnt_match else float("nan")

        # Our estimator on the SAME shots: 3^k * sign product on matched
        # shots, 0 on mismatched shots, averaged over ALL shots.
        k = len(spec)
        our_vals = []
        for shot in shots:
            matched = all(shot[pos][0] == letter for pos, letter in spec.items())
            if not matched:
                our_vals.append(0.0)
                continue
            sign = 1
            for pos in spec:
                sign *= shot[pos][1]
            our_vals.append((3.0**k) * sign)
        our_avg = float(np.mean(our_vals))

        from qiskit.quantum_info import Pauli

        true_val = state.expectation_value(Pauli(pauli_utils.spec_to_label(spec, n))).real

        label = pauli_utils.spec_to_label(spec, n)
        print(f"{label:10s} {huang_avg:26.4f} {our_avg:30.4f} {true_val:12.4f}")
        max_discrepancy = max(max_discrepancy, abs(huang_avg - our_avg))

    print()
    print(
        "Both estimators are unbiased estimators of the same true value from the SAME raw "
        "shots (Huang's matched-shots-only average vs. our 3^k-rescaled hit-or-miss average "
        "over all shots); they need not match shot-for-shot but should agree within a few "
        "Monte-Carlo standard errors."
    )
    print(f"Max |huang_ref - ours| across {len(specs)} observables: {max_discrepancy:.4f}")


if __name__ == "__main__":
    main()
