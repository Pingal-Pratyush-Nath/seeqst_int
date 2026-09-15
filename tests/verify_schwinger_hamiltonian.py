#!/usr/bin/env python3
"""Correctness checks for ``common/hamiltonians.py``'s lattice Schwinger
model construction, independent of the ``exp_5`` benchmark itself.

Unlike ``tests/cross_check_huang_reference.py`` and
``tests/verify_derandomized_seeqst.py`` (which cross-check against another
implementation's *behaviour*), most of what matters here is that the
Hamiltonian is the one Kokail et al. / HKP actually define. We check:

  1. An independent, from-scratch construction (plain numpy Kronecker
     products, no qiskit, no SparsePauliOp -- a genuinely different code
     path) reproduces ``schwinger_hamiltonian``'s matrix exactly, for
     several n. This is the strongest check: two implementations that share
     no code, built from the same equations, agreeing to machine precision.
  2. A hand-solved n=2 special case matches the constructed Hamiltonian's
     Pauli decomposition term-for-term (see module docstring below for the
     by-hand derivation).
  3. Hermiticity (trivial from construction, but cheap to confirm).
  4. Charge conservation, [H, sum_j Z_j] = 0 -- Kokail et al.'s own stated
     symmetry of this Hamiltonian (also used by the CP-symmetry reduction
     discussed in notes/applications/lattice_schwinger_vqs).
  5. Structural check: only the Pauli-string types HKP's SI Eq. S32 allows
     ever appear with a nonzero coefficient -- nearest-neighbour XX/YY
     bonds, single Z, and ZZ pairs with both indices <= n-1 (0-indexed:
     <= n-2). In particular, no XZ/YZ/XY-mixed term, no weight >= 3 term,
     and no Z_j Z_{n-1} term (the last site never appears in a ZZ pair),
     should appear.
  6. The exact ground energy from ``ground_state`` matches
     ``np.linalg.eigh`` applied directly to the independently-built numpy
     matrix from check 1.
  7. ``neel_state``'s definition holds directly: ``<Z_j> = (-1)**(j+1)`` at
     every site, and every hopping term has exactly zero expectation value
     (an unentangled computational basis state can't have off-diagonal
     X/Y support). This validates the second reference state ``exp_5`` uses
     alongside the true ground state.

HAND DERIVATION FOR n=2 (used in check 2)
--------------------------------------------------------------------------
One bond (sites 1,2). Hopping: (w/2)(X0 X1 + Y0 Y1). Mass:
(m/2)[(-1)^1 Z0 + (-1)^2 Z1] = -(m/2) Z0 + (m/2) Z1. Electric field:
L_1 = L_0 + (1/2)(Z0 + (-1)^1 I) = (1/2)(Z0 - I) (only one link for n=2, so
this is the only L_j, matching Eq. S32's j,j' <= n-1 = 1 range -- i.e. no
ZZ term at all for n=2). g*L_1^2 = g * (1/4)(Z0-I)^2 = g*(1/4)(2I - 2 Z0)
[using Z0^2 = I] = (g/2) I - (g/2) Z0. Total:
    H = (w/2)(X0X1 + Y0Y1) - (m/2 + g/2) Z0 + (m/2) Z1 + (g/2) I.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

from common.hamiltonians import (
    classify_term,
    electric_field_ops,
    ground_state,
    hamiltonian_terms,
    neel_state,
    schwinger_hamiltonian,
)
from common.pauli_utils import spec_to_label

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_at(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """op on qubit `site` (0-indexed, qubit 0 = rightmost factor, matching
    this codebase's convention throughout), identity elsewhere."""
    mats = [I2] * n
    mats[site] = op
    out = np.array([[1.0]], dtype=complex)
    for k in range(n - 1, -1, -1):  # build with qubit (n-1) leftmost, qubit 0 rightmost
        out = np.kron(out, mats[k])
    return out


def brute_force_schwinger(n: int, w: float, m: float, g: float) -> np.ndarray:
    """Fully independent construction: no qiskit, no SparsePauliOp, no
    shared code with common/hamiltonians.py beyond the defining equations
    themselves. Builds L_j as an explicit 2^n x 2^n diagonal-ish matrix via
    the same recursion, in plain numpy."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    for j in range(1, n):  # hopping bonds (site j, j+1), 1-indexed
        q1, q2 = j - 1, j
        XX = _kron_at(X, q1, n) @ _kron_at(X, q2, n)
        YY = _kron_at(Y, q1, n) @ _kron_at(Y, q2, n)
        H += (w / 2.0) * (XX + YY)

    for j in range(1, n + 1):  # mass term, all n sites
        H += (m / 2.0) * ((-1) ** j) * _kron_at(Z, j - 1, n)

    L = np.zeros((dim, dim), dtype=complex)  # L_0 = 0
    for j in range(1, n):  # links j = 1..n-1
        L = L + 0.5 * (_kron_at(Z, j - 1, n) + ((-1) ** j) * np.eye(dim, dtype=complex))
        H += g * (L @ L)

    return H


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


def main() -> None:
    w, m, g = 1.0, 0.9, 1.0
    all_ok = True

    print("=== Check 1: independent (plain-numpy) construction matches SparsePauliOp construction ===")
    for n in (2, 4, 6):
        H_ours = schwinger_hamiltonian(n, w=w, m=m, g=g).to_matrix()
        H_brute = brute_force_schwinger(n, w=w, m=m, g=g)
        max_diff = float(np.max(np.abs(H_ours - H_brute)))
        all_ok &= check(f"n={n}: max|H_sparsepauliop - H_bruteforce| = {max_diff:.2e}", max_diff < 1e-9)

    print("\n=== Check 2: hand-derived n=2 special case ===")
    n = 2
    h2 = schwinger_hamiltonian(n, w=w, m=m, g=g)
    terms = dict()
    for label, coeff in zip(h2.paulis.to_labels(), h2.coeffs):
        terms[label] = complex(coeff)
    expected = {
        "XX": w / 2.0,
        "YY": w / 2.0,
        "ZI": -(m / 2.0 + g / 2.0),  # qubit 0 = Z0, rightmost char -> label "ZI" means Z on qubit1(leftmost)? check below
        "IZ": m / 2.0,
        "II": g / 2.0,
    }
    # NOTE: qiskit label convention is leftmost=highest qubit index, so for n=2,
    # "ZI" = Z on qubit 1 (site 2), "IZ" = Z on qubit 0 (site 1). Fix the expected
    # dict to that convention explicitly (built via spec_to_label to avoid hand-
    # reversing it ourselves):
    expected = {
        "XX": w / 2.0,
        "YY": w / 2.0,
        spec_to_label({0: "Z"}, 2): -(m / 2.0 + g / 2.0),  # site 1 = qubit 0
        spec_to_label({1: "Z"}, 2): m / 2.0,               # site 2 = qubit 1
        "II": g / 2.0,
    }
    same_keys = set(terms) == set(expected)
    all_ok &= check("n=2: same set of nonzero Pauli labels", same_keys, f"got {sorted(terms)}, expected {sorted(expected)}")
    if same_keys:
        max_diff = max(abs(terms[k] - expected[k]) for k in expected)
        all_ok &= check(f"n=2: coefficients match hand derivation (max diff {max_diff:.2e})", max_diff < 1e-9)

    print("\n=== Check 3: Hermiticity ===")
    for n in (2, 4, 6):
        H = schwinger_hamiltonian(n, w=w, m=m, g=g).to_matrix()
        max_asym = float(np.max(np.abs(H - H.conj().T)))
        all_ok &= check(f"n={n}: max|H - H^dagger| = {max_asym:.2e}", max_asym < 1e-9)

    print("\n=== Check 4: charge conservation [H, sum_j Z_j] = 0 ===")
    for n in (4, 6):
        H = schwinger_hamiltonian(n, w=w, m=m, g=g).to_matrix()
        Sz = sum(_kron_at(Z, q, n) for q in range(n))
        comm = H @ Sz - Sz @ H
        max_comm = float(np.max(np.abs(comm)))
        all_ok &= check(f"n={n}: max|[H, sum Z_j]| = {max_comm:.2e}", max_comm < 1e-8)

    print("\n=== Check 5: only HKP Eq. S32's allowed Pauli-string types appear (via classify_term) ===")
    for n in (6, 8):
        h = schwinger_hamiltonian(n, w=w, m=m, g=g)
        bad = []
        terms_here = hamiltonian_terms(h, n, drop_identity=True)
        for coeff, spec in terms_here:
            sites = sorted(spec.keys())
            try:
                group = classify_term(spec, n)
            except ValueError as exc:
                bad.append((spec, str(exc)))
                continue
            # classify_term only checks the TYPE; separately confirm the site
            # indices are valid for this n (Eq. S32's j, j' <= n-1 range,
            # i.e. 0-indexed <= n-2 -- see the module docstring):
            if group == "zz_long_range" and sites[-1] > n - 2:
                bad.append((spec, f"ZZ pair touches last site (0-indexed max allowed {n - 2})"))
            elif group == "hopping" and sites[-1] > n - 1:
                bad.append((spec, "hopping pair out of range"))
        all_ok &= check(
            f"n={n}: all {len(terms_here)} terms classify as a valid Schwinger-model term type",
            not bad, str(bad[:5]),
        )

    print("\n=== Check 6: ground_state() energy matches independent numpy eigh ===")
    for n in (4, 6):
        h = schwinger_hamiltonian(n, w=w, m=m, g=g)
        e_ours, psi = ground_state(h, n)
        H_brute = brute_force_schwinger(n, w=w, m=m, g=g)
        e_brute = float(np.min(np.linalg.eigvalsh(H_brute)))
        all_ok &= check(f"n={n}: E0_ours={e_ours:.8f} vs E0_bruteforce={e_brute:.8f}", abs(e_ours - e_brute) < 1e-8)
        # Also actually exercise the returned Statevector's expectation_value
        # machinery (not just the eigenvalue) -- this is the exact code path
        # exp_5 depends on to get true (exact) Hamiltonian-term values on the
        # ground state, and caught a real bug (a non-contiguous eigenvector
        # slice from np.linalg.eigh that qiskit's Pauli expectation-value
        # code rejects) that checking the eigenvalue alone did not.
        e_via_expval = psi.expectation_value(h).real
        all_ok &= check(
            f"n={n}: <psi|H|psi> via Statevector.expectation_value = {e_via_expval:.8f} matches E0_ours",
            abs(e_via_expval - e_ours) < 1e-8,
        )

    print("\n=== Check 7: neel_state() matches its own definition ===")
    for n in (4, 6):
        state = neel_state(n)
        max_z_diff = 0.0
        for j in range(1, n + 1):  # Kokail 1-indexed site label
            site_idx = j - 1
            z_val = state.expectation_value(Pauli(spec_to_label({site_idx: "Z"}, n))).real
            expected_z = float((-1) ** (j + 1))
            max_z_diff = max(max_z_diff, abs(z_val - expected_z))
        all_ok &= check(f"n={n}: max|<Z_j> - (-1)^(j+1)| = {max_z_diff:.2e}", max_z_diff < 1e-9)

        max_hop = 0.0
        h = schwinger_hamiltonian(n, w=w, m=m, g=g)
        for coeff, spec in hamiltonian_terms(h, n, drop_identity=True):
            if classify_term(spec, n) != "hopping":
                continue
            val = state.expectation_value(Pauli(spec_to_label(spec, n))).real
            max_hop = max(max_hop, abs(val))
        all_ok &= check(f"n={n}: max|<hopping term>| on neel_state = {max_hop:.2e}", max_hop < 1e-9)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
