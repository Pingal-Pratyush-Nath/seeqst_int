"""Physical Hamiltonians for the shadow benchmark (exp_5).

Currently: the lattice Schwinger model of Kokail, Maier, van Bijnen,
Brydges, Joshi, Jurcevic, Muschik, Silvi, Blatt, Roos & Zoller,
"Self-verifying variational quantum simulation of lattice models",
Nature 569, 355 (2019) (arXiv:1810.03421) -- the seed application Huang,
Kueng & Preskill's own SI (arXiv:2002.08953, Sec. F; main-text Fig. 5) uses
to illustrate classical shadows for local Hamiltonian estimation, and the
system Kokail et al. actually ran on up to 20 trapped ions. It is exactly
the kind of "real Hamiltonian shadow papers use" this benchmark's exp_0/
exp_1 (random Pauli observables) do not cover -- see the README's "What's
*not* covered here" list.

CONSTRUCTION, NOT A HAND-DERIVED CLOSED FORM
--------------------------------------------------------------------------
After a Kogut-Susskind encoding and Jordan-Wigner transform, and
integrating out the gauge field via Gauss's law, Kokail et al.'s Eq. (1)-(2)
give

    H = w * sum_{j=1}^{N-1} [sigma^+_j sigma^-_{j+1} + h.c.]
      + (m/2) * sum_{j=1}^{N} (-1)^j sigma^z_j
      + g * sum_{j=1}^{N-1} L_j^2                                     (*)

where L_j is the electric field on the link between sites j and j+1, fixed
by the discrete Gauss-law recursion (their Eq. 2, open boundary conditions,
vacuum -- i.e. zero field -- on both ends of the chain):

    L_j = L_{j-1} + (1/2)(sigma^z_j + (-1)^j),         L_0 = 0.

The first term above is exactly HKP SI Eq. S32's hopping piece, written
there as (w/2) sum (X_j X_{j+1} + Y_j Y_{j+1}) -- the standard identity for
a single XY-hopping bond, used directly here rather than re-derived. The
third term expands into a long-range Z_j Z_j' coupling plus a correction to
the on-site Z term; rather than hand-deriving that expansion's coefficients
(easy to get a sign or an off-by-one wrong), we build L_j directly as a
qiskit ``SparsePauliOp`` (a diagonal sum of Z's plus a c-number) from the
recursion above, square it with ``SparsePauliOp``'s own exact operator
algebra (``@``), and let ``.simplify()`` collect and cancel terms. This is
guaranteed correct by construction: there is no algebra above for us to get
wrong, only the recursion itself, which is copied verbatim from Kokail et
al.'s Eq. (2). ``tests/verify_schwinger_hamiltonian.py`` checks the result
against several independent structural facts (Hermiticity, the charge/CP
symmetry [H, sum Z_j] = 0, and that only the Pauli-string types HKP's Eq.
S32 allows -- nearest-neighbour XX/YY, single Z, and ZZ pairs with both
indices < N -- ever appear with a nonzero coefficient).

Default parameters w=1, m=0.9, g=1 are Kokail et al.'s own values for their
main 20-ion simulation (their Fig. 2a caption).

QUBIT INDEXING
--------------------------------------------------------------------------
Kokail/HKP label sites 1..N. We use 0-indexed qubits throughout, matching
every other module in this benchmark (``common/pauli_utils.py`` etc.):
physical site j (1-indexed) <-> qubit index j-1. All Pauli labels are built
via ``pauli_utils.spec_to_label``, so the qubit-0-is-rightmost convention
used everywhere else in this codebase is reused automatically rather than
re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from common.pauli_utils import is_pure_z_type, spec_to_label

TermGroup = Literal["z_single", "zz_long_range", "hopping"]


def _pauli_op(spec: dict[int, str], n: int, coeff: complex = 1.0) -> SparsePauliOp:
    return SparsePauliOp(spec_to_label(spec, n), coeff)


def _identity_op(n: int, coeff: complex = 1.0) -> SparsePauliOp:
    return SparsePauliOp("I" * n, coeff)


def _z_op(site: int, n: int, coeff: complex = 1.0) -> SparsePauliOp:
    return _pauli_op({site: "Z"}, n, coeff)


def electric_field_ops(n: int) -> list[SparsePauliOp]:
    """``L_1, ..., L_{n-1}`` (Kokail et al. Eq. 2, ``L_0 = 0``), as
    ``SparsePauliOp``s on ``n`` qubits. Returned in order, so element ``k``
    (0-indexed) is ``L_{k+1}`` in their 1-indexed notation. There are only
    ``n-1`` of these for ``n`` sites (open boundary conditions: a chain of
    n sites has n-1 links, and HKP's Eq. S32 correspondingly restricts every
    Z_jZ_j' term to j, j' <= n-1)."""
    ops: list[SparsePauliOp] = []
    running = _identity_op(n, 0.0)  # L_0 = 0
    for j in range(1, n):  # j = 1, ..., n-1 (Kokail's 1-indexed site label)
        site_idx = j - 1  # 0-indexed qubit for site j
        sign = (-1) ** j
        running = (running + 0.5 * (_z_op(site_idx, n) + sign * _identity_op(n))).simplify()
        ops.append(running)
    return ops


def schwinger_hamiltonian(n: int, w: float = 1.0, m: float = 0.9, g: float = 1.0) -> SparsePauliOp:
    """The lattice Schwinger model Hamiltonian (Kokail et al. Eq. 1-2 / HKP
    SI Eq. S32) on ``n`` qubits, built by direct operator construction (see
    module docstring). ``n`` must be even (Kokail et al.'s own requirement,
    for a chain with vacuum boundary conditions on both ends -- see
    ``tests/verify_schwinger_hamiltonian.py`` for the charge-neutrality
    check this relies on)."""
    if n < 2:
        raise ValueError("need at least 2 sites")
    if n % 2 != 0:
        raise ValueError(f"the lattice Schwinger model needs an even number of sites, got n={n}")

    h = _identity_op(n, 0.0)

    # Hopping: (w/2) sum_{j=1}^{n-1} (X_j X_{j+1} + Y_j Y_{j+1})  [HKP SI Eq. S32]
    for j in range(1, n):  # bond between sites j, j+1 (1-indexed)
        q1, q2 = j - 1, j  # 0-indexed qubits
        h = h + (w / 2.0) * _pauli_op({q1: "X", q2: "X"}, n)
        h = h + (w / 2.0) * _pauli_op({q1: "Y", q2: "Y"}, n)

    # Staggered mass: (m/2) sum_{j=1}^{n} (-1)^j Z_j
    for j in range(1, n + 1):
        h = h + (m / 2.0) * ((-1) ** j) * _z_op(j - 1, n)

    # Gauss/electric-field energy: g * sum_{j=1}^{n-1} L_j^2
    for l_op in electric_field_ops(n):
        h = h + g * (l_op @ l_op)

    return h.simplify()


def hamiltonian_terms(op: SparsePauliOp, n: int, drop_identity: bool = True) -> list[tuple[float, dict[int, str]]]:
    """Convert a ``SparsePauliOp`` into ``(coefficient, spec)`` pairs, using
    this codebase's ``spec`` convention (dict qubit index -> 'X'/'Y'/'Z',
    0-indexed, unlisted qubits implicitly identity -- see
    ``common/pauli_utils.py``). Coefficients are taken to be real (the
    Hamiltonian is Hermitian, so any imaginary part is numerical noise;
    asserted small rather than silently dropped). If ``drop_identity``, the
    all-identity term (a pure energy offset, trivially "measured" as
    <I>=1 by any protocol) is omitted from the returned list."""
    out: list[tuple[float, dict[int, str]]] = []
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        assert abs(coeff.imag) < 1e-9, f"Hamiltonian term {label} has non-negligible imaginary part {coeff}"
        # label is qiskit convention: leftmost char = highest qubit index (n-1), rightmost = qubit 0.
        spec = {n - 1 - pos: letter for pos, letter in enumerate(label) if letter != "I"}
        if not spec and drop_identity:
            continue
        out.append((float(coeff.real), spec))
    return out


def classify_term(spec: dict[int, str], n: int) -> TermGroup:
    """Classify a nonzero Hamiltonian term (as returned by
    ``hamiltonian_terms``) into one of the three structural types that
    Kokail et al.'s Eq. (1)-(2) / HKP SI Eq. S32 actually produce -- see the
    module docstring above and ``tests/verify_schwinger_hamiltonian.py``
    Check 5, which confirms these are the *only* three types that ever
    appear with a nonzero coefficient:

      "z_single"      -- a single on-site Z (the staggered mass term)
      "zz_long_range" -- a Z_j Z_j' pair, any range (the Gauss-law /
                         electric-field term -- "long range" because L_j^2
                         couples every pair of links up to j, not just
                         nearest neighbours)
      "hopping"        -- a nearest-neighbour X_jX_{j+1} or Y_jY_{j+1} pair

    Raises ``ValueError`` on anything else. That should never happen for a
    term actually produced by ``hamiltonian_terms(schwinger_hamiltonian(...))``
    -- if it does, that's a real bug (e.g. a construction error), not an
    expected edge case, so this deliberately raises rather than returning
    e.g. "other". ``exp_5`` uses this directly to report shadow performance
    broken down by physical term type -- the ensembles' inverse-map weights
    beta(P) depend on exactly this classification (pure-Z-type vs.
    everything else -- see ``ensembles/seeqst_ensemble.py``), so this is the
    grouping the benchmark's headline result (SEEQST favouring Z-type terms,
    Pauli favouring hopping terms) is actually about.

    Note this only classifies the TYPE, not whether the specific site
    indices are valid for a given n (e.g. that a ``zz_long_range`` pair
    never touches site n-1) -- that's a construction-correctness question,
    checked separately (and more strictly) by
    ``tests/verify_schwinger_hamiltonian.py``.
    """
    wt = len(spec)
    sites = sorted(spec.keys())
    if wt == 1 and is_pure_z_type(spec):
        return "z_single"
    if wt == 2 and is_pure_z_type(spec):
        return "zz_long_range"
    letters = set(spec.values())
    if wt == 2 and letters in ({"X"}, {"Y"}) and sites[1] == sites[0] + 1:
        return "hopping"
    raise ValueError(
        f"term {spec} is not a recognized Schwinger-model term type "
        f"(weight={wt}, letters={letters}) -- likely a Hamiltonian "
        f"construction bug, see tests/verify_schwinger_hamiltonian.py Check 5"
    )


def neel_state(n: int) -> Statevector:
    """The staggered ("Neel") computational-basis reference state: site j
    (Kokail et al.'s 1-indexing) is prepared in the qubit eigenstate with
    Z-eigenvalue ``(-1)**(j+1)`` -- i.e. qubit ``j-1`` (0-indexed) is ``|0>``
    for odd j, ``|1>`` for even j. In spin-chain / lattice-gauge-theory
    language this is the "bare" or strong-coupling vacuum: the classical
    product state that (by construction) makes every staggered-mass term
    ``(-1)^j Z_j`` as negative as possible, has exactly zero electric field
    at every link (``L_j`` is a classical number here, not fluctuating), and
    -- being an unentangled computational basis state -- has EXACTLY zero
    expectation value for every hopping term (X/Y flips a basis state to an
    orthogonal one). It is a standard reference/initial state for VQE-style
    simulations of this model (e.g. Kokail et al.'s own ansatz starts from
    it), which is why it's included in ``exp_5`` alongside the true ground
    state: a second physically-motivated state, at the opposite extreme of
    entanglement (none at all, vs. the ground state's genuine entanglement).

    Built directly from the qubit index (``bit(q) = q % 2``, then reversed
    into qiskit's leftmost-highest-qubit label convention) rather than via
    any "n - 1 - pos"-style position arithmetic, specifically to avoid that
    common off-by-one/reversal error class. ``tests/verify_schwinger_hamiltonian.py``
    Check 7 confirms ``<Z_j> = (-1)**(j+1)`` for every site directly against
    this definition, and that every hopping term's expectation value is
    exactly zero on this state.
    """
    label = "".join(str(q % 2) for q in reversed(range(n)))  # qubit 0 = rightmost char
    return Statevector.from_label(label)


@dataclass
class SchwingerSystem:
    """A fully-specified lattice Schwinger model instance: the Hamiltonian
    (as a list of (coefficient, spec) terms, identity dropped -- see
    ``hamiltonian_terms``), its identity-term offset, and its exact ground
    state / ground energy (dense diagonalization -- fine up to n ~ 12-14;
    see ``ground_state`` for the size this remains practical at)."""

    n: int
    w: float
    m: float
    g: float
    terms: list[tuple[float, dict[int, str]]]  # (coefficient, spec), identity dropped
    identity_offset: float
    ground_energy: float
    ground_state: Statevector


def ground_state(op: SparsePauliOp, n: int) -> tuple[float, Statevector]:
    """Exact ground energy and ground state of a Hermitian ``SparsePauliOp``
    via dense diagonalization. Dense is deliberate (not sparse Lanczos):
    this benchmark's system sizes (matching exp_0/exp_1's n=2..10 range)
    keep the 2^n x 2^n matrix small, and dense diagonalization gives every
    eigenvalue with no risk of Lanczos non-convergence -- simplicity over
    speed, since this is called once per (n, w, m, g), not in a hot loop.
    Remains practical to roughly n=12-14 (2^14 = 16384); do not push this
    much further without switching to ``scipy.sparse.linalg.eigsh``."""
    mat = op.to_matrix()
    eigvals, eigvecs = np.linalg.eigh(mat)
    idx = int(np.argmin(eigvals))
    # eigvecs[:, idx] is a column slice of a 2D array -- a non-contiguous
    # view (stride = the matrix's row length), which qiskit's Statevector
    # expectation-value machinery rejects ("array is not contiguous or is
    # misaligned"). np.ascontiguousarray copies it into a proper 1D buffer.
    vec = np.ascontiguousarray(eigvecs[:, idx])
    return float(eigvals[idx].real), Statevector(vec)


def build_schwinger_system(n: int, w: float = 1.0, m: float = 0.9, g: float = 1.0) -> SchwingerSystem:
    """Build the Hamiltonian, split off its identity offset, and diagonalize
    for the exact ground state -- the one call ``exp_5`` needs."""
    h = schwinger_hamiltonian(n, w=w, m=m, g=g)
    terms = hamiltonian_terms(h, n, drop_identity=True)
    identity_terms = [c for c, spec in hamiltonian_terms(h, n, drop_identity=False) if not spec]
    identity_offset = identity_terms[0] if identity_terms else 0.0
    energy, state = ground_state(h, n)
    return SchwingerSystem(
        n=n, w=w, m=m, g=g,
        terms=terms, identity_offset=identity_offset,
        ground_energy=energy, ground_state=state,
    )
