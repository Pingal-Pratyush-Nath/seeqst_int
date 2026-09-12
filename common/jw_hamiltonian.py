"""Numeric Jordan-Wigner electronic Hamiltonians, in this benchmark's
portable ``(coeffs, specs)`` representation (see
``common/pauli_utils.py::random_sparse_operator``'s docstring for why that
representation -- not a built qiskit operator -- is used throughout this
codebase).

CONVENTION AND CORRECTNESS
--------------------------------------------------------------------------
This reimplements ``tests/JW_transformation.ipynb``'s Jordan-Wigner algebra
numerically (concrete numbers from the start, instead of building a sympy
expression and substituting afterward -- a performance/simplicity choice,
not a different algorithm): Huggins convention n_p = (I+Z_p)/2,
a_p^dagger = Z_0...Z_{p-1} (X_p+iY_p)/2, a_p = Z_0...Z_{p-1} (X_p-iY_p)/2.

That notebook's own Jordan-Wigner construction was independently verified
(same conversation, cross-checked with plain numpy, no sympy/qiskit
needed) against three convention-agnostic tests: (1) the canonical
anticommutation relations {a_p, a_q^dagger} = delta_pq I etc. hold exactly
for the resulting operators, (2) the qubit Hamiltonian is exactly Hermitian
(and has purely real eigenvalues) whenever h, v satisfy the standard
Hermitian/chemist-symmetric integral conditions, and (3) for a
non-interacting (v=0) instance, the full many-body spectrum exactly
matches the analytically-known "subset-sum of single-particle eigenvalues"
result. This module's algebra is identical to what was checked there;
``jw_electronic_hamiltonian`` additionally re-indexes the internal
left-to-right JW string into THIS codebase's own spec convention (see
``pauli_utils.spec_to_label``: spec key ``i`` <-> qiskit label position
``n-1-i``), which was itself checked bit-for-bit against
``spec_to_label``'s actual output (same conversation) before being used
here.

``ground_state`` below builds the dense ``2**n x 2**n`` matrix straight
from the SAME ``(coeffs, specs)`` this module hands to
``ShadowEnsemble.evaluate_snapshots`` (via ``pauli_utils.spec_to_label``
and a plain kron-product construction that matches qiskit's own
``Pauli(label).to_matrix()`` convention exactly), so the returned
``Statevector`` and the specs used for shadow estimation are guaranteed to
refer to the exact same operator -- no separate/independent bit-ordering
assumption to get wrong.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from qiskit.quantum_info import Statevector

from common.pauli_utils import is_pure_z_type, spec_to_label, xy_weight

_MULT_TABLE = {
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}

_PAULI_2X2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _mul1(a: str, b: str) -> tuple[complex, str]:
    if a == "I":
        return 1, b
    if b == "I":
        return 1, a
    if a == b:
        return 1, "I"
    return _MULT_TABLE[(a, b)]


def _mul_strings(P: str, Q: str) -> tuple[complex, str]:
    phase = 1
    out = []
    for p, q in zip(P, Q):
        ph, r = _mul1(p, q)
        phase *= ph
        out.append(r)
    return phase, "".join(out)


def _mul_ops(A: dict, B: dict) -> dict:
    result: dict = defaultdict(complex)
    for P, cP in A.items():
        for Q, cQ in B.items():
            phase, R = _mul_strings(P, Q)
            result[R] += cP * cQ * phase
    return dict(result)


def _creation(p: int, n: int) -> dict:
    X, Y = ["I"] * n, ["I"] * n
    for j in range(p):
        X[j] = Y[j] = "Z"
    X[p], Y[p] = "X", "Y"
    return {"".join(X): 0.5, "".join(Y): 0.5j}


def _annihilation(p: int, n: int) -> dict:
    X, Y = ["I"] * n, ["I"] * n
    for j in range(p):
        X[j] = Y[j] = "Z"
    X[p], Y[p] = "X", "Y"
    return {"".join(X): 0.5, "".join(Y): -0.5j}


def jw_electronic_hamiltonian(
    n: int, h: np.ndarray, v: np.ndarray, atol: float = 1e-10
) -> tuple[np.ndarray, list[dict]]:
    """Numeric Jordan-Wigner transform of

        H = sum_pq h[p,q] a_p^dagger a_q
          + 1/2 sum_pqrs v[p,q,r,s] a_p^dagger a_q^dagger a_r a_s

    (Huggins convention, see module docstring). Returns this codebase's
    ``(coeffs, specs)`` representation -- ``coeffs`` real (a nonzero
    imaginary residue above ``atol`` raises, since that means h/v are not
    actually Hermitian-symmetric: need h[p,q]==conj(h[q,p]) and
    v[p,q,r,s]==conj(v[s,r,q,p])) and ``specs`` a list of
    ``pauli_utils``-style ``{qubit_index: 'X'|'Y'|'Z'}`` dicts, already
    re-indexed to match ``pauli_utils.spec_to_label``'s convention exactly
    (verified bit-for-bit -- see module docstring).
    """
    H: dict = defaultdict(complex)
    for p in range(n):
        for q in range(n):
            hpq = h[p, q]
            if hpq == 0:
                continue
            for P, c in _mul_ops(_creation(p, n), _annihilation(q, n)).items():
                H[P] += hpq * c
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    vpqrs = v[p, q, r, s]
                    if vpqrs == 0:
                        continue
                    term = _creation(p, n)
                    term = _mul_ops(term, _creation(q, n))
                    term = _mul_ops(term, _annihilation(r, n))
                    term = _mul_ops(term, _annihilation(s, n))
                    for P, c in term.items():
                        H[P] += 0.5 * vpqrs * c

    coeffs: list[float] = []
    specs: list[dict] = []
    for P, c in H.items():
        if abs(c) < atol:
            continue
        if abs(c.imag) > atol:
            raise ValueError(
                f"Pauli term {P!r} has non-negligible imaginary coefficient "
                f"{c!r} -- h/v are not Hermitian-symmetric (need "
                f"h[p,q]==conj(h[q,p]) and v[p,q,r,s]==conj(v[s,r,q,p]))"
            )
        spec = {n - 1 - i: ch for i, ch in enumerate(P) if ch != "I"}
        coeffs.append(c.real)
        specs.append(spec)
    return np.array(coeffs), specs


def demo_integrals(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """A small, simple, HAND-PICKED (not a real molecule -- no chemistry
    package used or required) set of numeric one- and two-electron
    integrals, chosen only to give ``jw_electronic_hamiltonian`` a concrete
    n-qubit instance to work with for exp_12:

      - on-site energies h[p,p] = p - (n-1)/2 -- a linear ladder symmetric
        about 0 (e.g. at n=6: -2.5,-1.5,-0.5,0.5,1.5,2.5), so the
        non-interacting ground state occupies roughly the lower half of
        the orbitals (a nontrivial, not-all-empty/not-all-full filling);
      - nearest-neighbour hopping h[p,p+1]=h[p+1,p] = -0.5;
      - nearest-neighbour density-density interaction: v[p,q,q,p] =
        v[q,p,p,q] = U = 0.4 for |p-q|==1 (zero otherwise), which makes the
        two-body term work out to sum_{p<q, |p-q|=1} U * n_p * n_q exactly
        (a standard nearest-neighbour Hubbard-style coupling -- see the
        identity a_p^dag a_q^dag a_q a_p = n_p n_q for p != q).

    Both are real and satisfy the standard chemist symmetries (h
    symmetric; v[p,q,r,s] = v[q,p,s,r] = v[r,s,p,q] = v[s,r,q,p]) needed
    for H to come out Hermitian -- ``jw_electronic_hamiltonian``'s ``atol``
    check will raise loudly if that symmetry is ever broken by a future
    edit here.
    """
    h = np.zeros((n, n))
    for p in range(n):
        h[p, p] = p - (n - 1) / 2.0
    for p in range(n - 1):
        h[p, p + 1] = h[p + 1, p] = -0.5

    v = np.zeros((n, n, n, n))
    U = 0.4
    for p in range(n):
        for q in range(n):
            if abs(p - q) == 1:
                v[p, q, q, p] = U
    return h, v


def _pauli_label_matrix(label: str) -> np.ndarray:
    M = np.array([[1.0]], dtype=complex)
    for ch in label:
        M = np.kron(M, _PAULI_2X2[ch])
    return M


def hamiltonian_matrix(coeffs: np.ndarray, specs: list[dict], n: int) -> np.ndarray:
    """Dense ``2**n x 2**n`` matrix for ``sum_i coeffs[i] * P_i``, built via
    ``pauli_utils.spec_to_label`` + a left-to-right Kronecker product --
    exactly qiskit's own ``Pauli(label).to_matrix()`` convention, so this
    matrix's basis ordering matches what ``ShadowEnsemble.evaluate_snapshots``
    assumes (verified bit-for-bit against ``spec_to_label`` -- see module
    docstring). Only practical up to about n~12-13 (2**n x 2**n dense).
    """
    dim = 2**n
    M = np.zeros((dim, dim), dtype=complex)
    for c, spec in zip(coeffs, specs):
        label = spec_to_label(spec, n)
        M += c * _pauli_label_matrix(label)
    return M


def ground_state(
    coeffs: np.ndarray, specs: list[dict], n: int
) -> tuple[Statevector, float]:
    """Exact ground state and ground energy of ``sum_i coeffs[i] * P_i`` by
    direct dense diagonalization (``numpy.linalg.eigh``, since the matrix
    from ``hamiltonian_matrix`` is exactly Hermitian for real ``coeffs``).
    Returns ``(Statevector, energy)`` -- the ``Statevector`` plugs directly
    into ``ShadowEnsemble.sample_snapshots`` exactly like every other state
    builder in ``common/states.py``.
    """
    M = hamiltonian_matrix(coeffs, specs, n)
    herm_err = float(np.max(np.abs(M - M.conj().T)))
    if herm_err > 1e-8:
        raise ValueError(
            f"Hamiltonian matrix is not Hermitian (max|H-H^dagger|={herm_err:.3e}) "
            "-- check that coeffs/specs come from jw_electronic_hamiltonian "
            "with Hermitian-symmetric h, v"
        )
    eigvals, eigvecs = np.linalg.eigh(M)
    ground_energy = float(eigvals[0])
    ground_vec = eigvecs[:, 0]
    return Statevector(ground_vec), ground_energy


def jw_long_range_kitaev(
    n: int,
    hopping: float = 1.0,
    mu: float = 1.0,
    delta: float = 1.0,
    alpha: float = 1.0,
    atol: float = 1e-10,
) -> tuple[np.ndarray, list[dict]]:
    """Jordan-Wigner transform of the open-boundary long-range p-wave
    pairing Kitaev chain (Vodola et al., Phys. Rev. Lett. 113, 156402
    (2014)):

        H = -hopping * sum_j (a_j^dagger a_{j+1} + h.c.)
            - mu * sum_j (n_j - 1/2)
            + (delta/2) * sum_{j<k} (a_j a_k + a_k^dagger a_j^dagger) / (k-j)^alpha

    Same Huggins-convention Jordan-Wigner transform as
    ``jw_electronic_hamiltonian`` (same ``_creation``/``_annihilation``
    primitives), but this Hamiltonian's pairing term (``a_j a_k`` and its
    Hermitian conjugate ``a_k^dagger a_j^dagger``) does NOT conserve
    particle number -- a genuinely different (BdG-type) algebraic
    structure than the number-conserving one/two-body electronic
    Hamiltonian, hence a separate function rather than a special case of
    ``h``/``v``.

    THE POINT OF THIS MODEL: a pairing term spanning sites ``j`` and
    ``k`` picks up a Z-string over every site strictly between them (via
    the same JW Z-string mechanism as ``jw_electronic_hamiltonian``), so
    its total Pauli weight is ``k-j+1`` -- genuine weights up to ``n``,
    unlike ``demo_integrals``' nearest-neighbour-only structure (which
    never exceeds weight 2 regardless of ``n``). This was prototyped and
    verified (real coefficients, exact Hermiticity, correct weight
    structure) in
    ``tests/JW_transformation_long_range_kitaev.ipynb``, and re-verified
    numerically (same conversation, plain numpy) before being ported
    here.

    PARAMETER REGIME CAVEAT -- READ BEFORE ASSUMING THIS BEATS PAULI: with
    this model's own naturally-quoted physics defaults
    (``hopping=mu=delta=1, alpha=1``), Pauli STILL wins overall (checked
    numerically): the O(1) hopping/onsite/short-range-pairing terms
    (weight <= 3) dominate the total coefficient-squared mass (>99.5% of
    it at ``alpha=1``) even though rare weight-up-to-``n`` terms exist,
    and Pauli is cheap exactly where that mass sits. To see SEEQST/
    Clifford actually beat Pauli in aggregate, the long-range pairing
    needs to genuinely DOMINATE the physics: weak/zero ``hopping``/``mu``,
    a SMALL ``alpha`` (slow decay -- ``alpha=0`` is fully uniform,
    all-to-all pairing with no decay at all), and a reasonably large
    ``n``. Checked example: ``n=10, hopping=0, mu=0, delta=1, alpha=0``
    with ``q=1/(n+1)`` gives roughly a 5.7x IMPROVEMENT over Pauli in the
    ``sum c_i^2 beta(P_i)`` proxy (and Clifford about 2.9x). Also note:
    an AGGRESSIVELY high-weight-tuned q (e.g. q close to 1, chosen as if
    "tuned to weight~n") can be dramatically WORSE than the "generic"
    untuned ``q=1/(n+1)`` here (checked: q=0.8 was over 400x worse than
    Pauli at n=10, alpha=0) -- this model's terms are spread across MANY
    different weights simultaneously, exactly the regime ``q=1/(n+1)``
    (Equation 7 / Example 2 of ``SEEQST_shadows4.pdf``) is meant for, not
    an aggressively high-weight-biased q.
    """
    H: dict = defaultdict(complex)
    identity_key = "I" * n

    for j in range(n - 1):
        for P, c in _mul_ops(_creation(j, n), _annihilation(j + 1, n)).items():
            H[P] += -hopping * c
        for P, c in _mul_ops(_creation(j + 1, n), _annihilation(j, n)).items():
            H[P] += -hopping * c

    for j in range(n):
        for P, c in _mul_ops(_creation(j, n), _annihilation(j, n)).items():
            H[P] += -mu * c
        H[identity_key] += 0.5 * mu

    for j in range(n):
        for k in range(j + 1, n):
            r = k - j
            coupling = delta / (2.0 * (r**alpha))
            for P, c in _mul_ops(_annihilation(j, n), _annihilation(k, n)).items():
                H[P] += coupling * c
            for P, c in _mul_ops(_creation(k, n), _creation(j, n)).items():
                H[P] += coupling * c

    coeffs: list[float] = []
    specs: list[dict] = []
    for P, c in H.items():
        if abs(c) < atol:
            continue
        if abs(c.imag) > atol:
            raise ValueError(
                f"Pauli term {P!r} has non-negligible imaginary coefficient "
                f"{c!r} -- this should not happen for real hopping/mu/delta/alpha "
                "(the hopping and pairing terms are each added together with "
                "their own exact Hermitian conjugate)"
            )
        spec = {n - 1 - i: ch for i, ch in enumerate(P) if ch != "I"}
        coeffs.append(c.real)
        specs.append(spec)
    return np.array(coeffs), specs


def seeqst_binomial_beta(spec: dict, n: int, q: float) -> float:
    """beta_q(P) for ``SEEQSTBinomialEnsemble`` as a pure function of
    ``(spec, n, q)`` -- mirrors
    ``ensembles.seeqst_binomial_ensemble.SEEQSTBinomialEnsemble.inverse_weight``
    exactly, without needing to instantiate an ensemble object, so it can
    be swept over many candidate q values cheaply (used by ``optimal_q``
    below)."""
    if not spec:
        return 1.0
    if is_pure_z_type(spec):
        j = len(spec)
        return 1.0 / (0.5 + 0.5 * (1 - 2 * q) ** j)
    k = xy_weight(spec)
    return 2.0 * q ** (-k) * (1 - q) ** (-(n - k))


def optimal_q(
    coeffs: np.ndarray,
    specs: list[dict],
    n: int,
    coarse_points: int = 2000,
    refine_iters: int = 6,
) -> float:
    """``argmin_q sum_i coeffs[i]^2 * beta_q(specs[i])`` over ``q in (0,1)``
    -- tunes ``SEEQSTBinomialEnsemble``'s q directly to a SPECIFIC given
    Hamiltonian, rather than guessing a value (or using the generic
    ``q=1/(n+1)``, or the single-fixed-weight ``q=m/n`` rule, both of
    which are special cases: a Hamiltonian whose terms all share one
    X/Y-weight ``m`` will have this function return exactly ``m/n``,
    since ``beta_q`` for non-pure-Z terms depends only on X/Y-weight, not
    on how much incidental Z-padding (e.g. from Jordan-Wigner locality) a
    term happens to carry).

    ``sum_i c_i^2 beta_q(P_i)`` is the natural per-term (uncorrelated)
    proxy for the combined-shadow estimator's variance under
    ``SEEQSTBinomialEnsemble(q)`` -- see the exp_12/exp_13 discussion for
    why this proxy is the right thing to minimize, and why an
    aggressively high-weight-tuned q can be dramatically WORSE than a
    well-chosen (or the generic ``1/(n+1)``) value for a Hamiltonian
    spanning many different weights at once.

    Deliberately dependency-free (a simple iteratively-narrowing grid
    search, not ``scipy.optimize``) since this codebase's actual runtime
    environment (qiskit installed on the user's machine) was not
    available to verify a scipy dependency against when this was
    written. Verified (same conversation, before being ported here): for
    the long-range Kitaev chain at ``n=10, hopping=mu=0, delta=1,
    alpha=0`` (every pairing term has X/Y-weight exactly 2, only the
    Z-string length varies with separation), this returns EXACTLY
    ``q=0.2 = 2/10``, recovering the textbook ``q=m/n`` rule for that
    common X/Y-weight -- the expected sanity check for this function,
    giving roughly a 9.9x improvement over Pauli in the variance proxy
    there (versus 5.7x at the generic ``q=1/(n+1)``).
    """
    nontrivial = [(c, s) for c, s in zip(coeffs, specs) if s]
    if not nontrivial:
        return 0.5  # H is pure identity (or empty); q is irrelevant

    def objective(q: float) -> float:
        return sum(c**2 * seeqst_binomial_beta(s, n, q) for c, s in nontrivial)

    lo, hi = 1e-6, 1 - 1e-6
    best_q = 0.5
    qs = np.array([best_q])
    for _ in range(refine_iters):
        qs = np.linspace(lo, hi, coarse_points)
        vals = np.array([objective(q) for q in qs])
        best_idx = int(np.argmin(vals))
        best_q = float(qs[best_idx])
        step = qs[1] - qs[0]
        lo = max(1e-6, best_q - 2 * step)
        hi = min(1 - 1e-6, best_q + 2 * step)
    return best_q
