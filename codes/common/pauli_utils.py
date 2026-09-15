"""
Random k-local Pauli observables.

A Pauli observable is represented as a ``spec``: a dict mapping qubit index
(0-indexed) -> one of 'X', 'Y', 'Z' for the qubits it acts on non-trivially.
Unlisted qubits are implicitly identity. This is the same representation
used by Huang's ``predicting-quantum-properties`` repo (list of
``(pauli_XYZ, position)`` pairs), just as a dict for convenience.

Qubit-index convention (must match everywhere state simulation happens):
qiskit's ``Pauli`` label strings and ``Statevector`` computational-basis
labels both put qubit 0 as the *rightmost* character. ``spec_to_label``
below builds labels consistent with that convention.
"""

from __future__ import annotations

import numpy as np

PAULI_LETTERS = ("X", "Y", "Z")


def random_pauli_spec(n: int, k: int, rng: np.random.Generator) -> dict[int, str]:
    """A uniformly random weight-k Pauli string on n qubits.

    Picks k distinct qubit positions uniformly and assigns each an
    independent uniformly random letter from {X, Y, Z}.
    """
    if not (0 <= k <= n):
        raise ValueError(f"locality k={k} must satisfy 0 <= k <= n={n}")
    positions = rng.choice(n, size=k, replace=False)
    letters = rng.choice(PAULI_LETTERS, size=k)
    return {int(pos): str(letter) for pos, letter in zip(positions, letters)}


def random_full_pauli_spec(n: int, rng: np.random.Generator) -> dict[int, str]:
    """A uniformly random NON-IDENTITY Pauli string on n qubits, with no
    locality constraint: each qubit independently gets I, X, Y or Z with
    equal probability 1/4, conditioned on the overall string not being the
    all-identity string (which has trivial expectation value 1 and would
    not be a meaningful "random observable" to test).

    This is the generator used by ``exp_1`` ("M random observables, which
    are random Pauli strings") -- unlike ``random_pauli_spec`` above, the
    weight/locality k is itself random here, distributed as
    Binomial(n, 3/4) conditioned on k >= 1.
    """
    letters_per_qubit = ("I", "X", "Y", "Z")
    while True:
        draw = rng.choice(letters_per_qubit, size=n)
        spec = {i: letter for i, letter in enumerate(draw) if letter != "I"}
        if spec:  # reject the all-identity draw and try again
            return spec


def spec_to_label(spec: dict[int, str], n: int) -> str:
    """Convert a Pauli spec into a qiskit-style label string of length n."""
    chars = ["I"] * n
    for pos, letter in spec.items():
        chars[pos] = letter
    # qiskit convention: leftmost character = highest-indexed qubit.
    return "".join(reversed(chars))


def weight(spec: dict[int, str]) -> int:
    return len(spec)


def is_pure_z_type(spec: dict[int, str]) -> bool:
    """True iff every non-identity factor of the (non-trivial) Pauli is Z.

    Used for the SEEQST inverse-map coefficients, which distinguish
    {I, Z}^n strings from everything else.
    """
    return all(letter == "Z" for letter in spec.values())


def bitstring_sign_product(spec: dict[int, str], outcome: str, n: int) -> int:
    """Given a computational-basis outcome string (qiskit convention, qubit 0
    rightmost), return prod_{q in support(spec)} (-1)^{bit_q}.

    Only meaningful when the outcome was obtained after rotating every
    qubit in ``support(spec)`` into the eigenbasis of ``spec``'s Pauli
    letter at that qubit (as done by the local random-Pauli ensemble).
    """
    sign = 1
    for pos in spec:
        bit = outcome[n - 1 - pos]  # undo the "reversed" convention
        if bit == "1":
            sign *= -1
    return sign


def xy_weight(spec: dict[int, str]) -> int:
    """Number of qubits where ``spec`` acts as X or Y (i.e. the size of the
    Pauli's "X/Y support" in the symplectic sense -- the qubits that are NOT
    pure-Z-type). Used by the bounded-X/Y-weight observable family below and
    by the SEEQST Binomial/uniform-size-capped ensembles' inverse-map
    formulas, whose eigenvalues (for a non-Z-type string) depend only on
    this count, not on any additional Z content elsewhere (see
    ``ensembles/seeqst_binomial_ensemble.py`` / ``seeqst_unifsize_ensemble.py``).
    """
    return sum(1 for letter in spec.values() if letter in ("X", "Y"))


def random_bounded_xy_spec(
    n: int, l: int, rng: np.random.Generator, min_xy: int = 1
) -> dict[int, str]:
    """A random FULL-WEIGHT (all n qubits non-identity) Pauli string with at
    most ``l`` X/Y factors and Z on every other qubit.

    Used by ``exp_6`` to isolate exactly the tunable knob the SEEQST
    threshold theorem (``QIP_notes/SEEQST_shadows_threshold.tex``) is about:
    the X/Y-support size ``a`` that SEEQST's inverse-map eigenvalue depends
    on (Theorem 2 there), with everything else pinned to Z so total Pauli
    weight is always exactly ``n`` -- the most conservative case for SEEQST
    relative to Pauli shadows (see that note's Remark 2), and one that keeps
    every generated observable inside the support of every ensemble under
    test in exp_6 (no ensemble ever sees a zero-probability Pauli).

    The actual X/Y count is drawn uniformly from {min_xy, ..., l} (not
    pinned to exactly l) to match "at most l X's and Y's" literally.
    ``min_xy=1`` (the default) guarantees at least one X/Y factor, which
    keeps every draw out of the pure-Z case -- deliberately, since
    ``SEEQSTUniformSizeEnsemble``'s inverse map has no closed form on that
    sector (see its module docstring) and this generator is the only
    observable source exp_6 uses.
    """
    if not (min_xy <= l <= n):
        raise ValueError(f"require min_xy={min_xy} <= l={l} <= n={n}")
    k = int(rng.integers(min_xy, l + 1))  # uniform in {min_xy, ..., l}
    xy_positions = rng.choice(n, size=k, replace=False)
    xy_letters = rng.choice(("X", "Y"), size=k)
    spec = {int(pos): str(letter) for pos, letter in zip(xy_positions, xy_letters)}
    for q in range(n):
        if q not in spec:
            spec[q] = "Z"
    return spec


def random_exact_xy_spec(
    n: int, m: int, rng: np.random.Generator, rest_mode: str = "random"
) -> dict[int, str]:
    """A random Pauli string with EXACTLY ``m`` X/Y factors and the other
    ``n - m`` qubits set according to ``rest_mode``:

      "random"    -- each non-XY qubit independently, uniformly Z or I
                      (fresh coin flip per qubit per draw). Default.
      "z"         -- every non-XY qubit is Z (exp_6's convention, i.e. this
                      reduces to ``random_bounded_xy_spec(n, l=m, rng,
                      min_xy=m)`` -- full weight n always).
      "identity"  -- every non-XY qubit is left as identity, i.e. total
                      Pauli weight is exactly m (the convention used
                      throughout ``notes/main_theorem/main_theorem.tex``).

    This is exp_7's "exact-m" observable family -- the regime Theorem 1 /
    Corollary 5 of that note are stated for: a SINGLE, FIXED, KNOWN X/Y
    locality m. It exists alongside (not as a replacement for)
    ``random_bounded_xy_spec`` above, which remains exp_7's generator for
    the "at least m" family via its existing ``min_xy`` parameter --
    ``random_bounded_xy_spec(n, l, rng, min_xy=m)`` already does exactly
    "weight between m and l, Z-padded" with no changes needed.

    Why ``rest_mode="random"`` is the default: by the general SEEQST
    eigenvalue theorem, every SEEQST ensemble's cost on a spec depends only
    on its X/Y-support (the symplectic ``a``), never on the additional
    Z-content ``b`` -- so under "random", every draw at fixed m has the
    EXACT SAME SEEQST beta regardless of how the Z/I split landed, while
    ``PauliEnsemble``'s own cost (``3 ** len(spec)``) is NOT b-independent
    (Z counts toward its weight, I doesn't) and so genuinely varies with
    the split. Generating specs this way turns that theoretical
    b-independence claim into something directly visible in exp_7's data,
    rather than only ever exercising the b=0 slice.

    Raises ValueError if ``m`` is not in ``[0, n]`` or ``rest_mode`` is not
    one of the three above. (exp_7.py additionally requires ``1 <= m <=
    n - 1`` at the CLI/config level, since that is what keeps the *tuned*
    q* = m/n strictly inside (0, 1) for ``SEEQSTBinomialEnsemble`` -- this
    function itself stays as permissive as ``random_pauli_spec`` above.)
    """
    if not (0 <= m <= n):
        raise ValueError(f"m={m} must satisfy 0 <= m <= n={n}")
    if rest_mode not in ("random", "z", "identity"):
        raise ValueError(
            f"rest_mode={rest_mode!r} must be one of 'random', 'z', 'identity'"
        )
    xy_positions = rng.choice(n, size=m, replace=False)
    xy_letters = rng.choice(("X", "Y"), size=m)
    spec = {int(pos): str(letter) for pos, letter in zip(xy_positions, xy_letters)}
    for q in range(n):
        if q in spec:
            continue
        if rest_mode == "z":
            spec[q] = "Z"
        elif rest_mode == "random" and rng.random() < 0.5:
            spec[q] = "Z"
        # "identity" (or the random coin landing on I): leave q out of spec entirely
    return spec


def random_capped_xy_spec(
    n: int, m: int, rng: np.random.Generator, min_xy: int = 0, rest_mode: str = "random"
) -> dict[int, str]:
    """A random Pauli string with X/Y-weight (symplectic X/Y-support size
    ``a``) drawn uniformly from ``{min_xy, ..., m}`` -- i.e. AT MOST ``m``
    -- and the other ``n - |a|`` qubits set according to ``rest_mode``, the
    SAME three conventions ``random_exact_xy_spec`` uses:

      "random"    -- each non-XY qubit independently, uniformly Z or I
                      (fresh coin flip per qubit per draw). Default.
      "z"         -- every non-XY qubit is Z (full weight n always; this
                      reduces to exactly ``random_bounded_xy_spec(n, m,
                      rng, min_xy)``, kept as a separate function for
                      backward compatibility -- exp_8/exp_10 both call it
                      directly).
      "identity"  -- every non-XY qubit is left as identity (total Pauli
                      weight = |a| <= m).

    This generalizes ``random_bounded_xy_spec`` (hard-coded to
    ``rest_mode="z"``) to the FULL ``S``-sparsity condition of Definition
    5.3 in ``SEEQST_shadows4.pdf`` / ``notes/working/seeqst_sparse_tuning.tex``:
    that definition places NO constraint at all on a term's Z-content (the
    symplectic ``b``) once its X/Y-support ``a`` lies in the allowed set
    ``S_m = {a : |a| <= m}`` -- only ``rest_mode="z"`` reproduces the
    single, deterministic ``b`` that ``random_bounded_xy_spec`` always
    used. ``rest_mode="random"`` is the default here for the same reason
    it's ``random_exact_xy_spec``'s default: it lets ``b`` actually vary
    from draw to draw instead of only ever exercising the ``b=Z^{n-|a|}``
    slice, which matters once you're summing many such terms into one
    operator (see ``random_sparse_operator`` below) rather than tracking
    them as separate single-term observables.

    Used by ``random_sparse_operator`` to draw each individual term of a
    genuine summed ``S_m``-sparse operator.
    """
    if not (0 <= min_xy <= m <= n):
        raise ValueError(f"require 0 <= min_xy={min_xy} <= m={m} <= n={n}")
    if rest_mode not in ("random", "z", "identity"):
        raise ValueError(
            f"rest_mode={rest_mode!r} must be one of 'random', 'z', 'identity'"
        )
    k = int(rng.integers(min_xy, m + 1))  # uniform in {min_xy, ..., m}
    xy_positions = rng.choice(n, size=k, replace=False)
    xy_letters = rng.choice(("X", "Y"), size=k)
    spec = {int(pos): str(letter) for pos, letter in zip(xy_positions, xy_letters)}
    for q in range(n):
        if q in spec:
            continue
        if rest_mode == "z":
            spec[q] = "Z"
        elif rest_mode == "random" and rng.random() < 0.5:
            spec[q] = "Z"
        # "identity" (or the random coin landing on I): leave q out of spec entirely
    return spec


def random_sparse_operator(
    n: int,
    m: int,
    num_terms: int,
    rng: np.random.Generator,
    min_xy: int = 0,
    rest_mode: str = "random",
    coeff_dist: str = "normal",
) -> tuple[np.ndarray, list[dict[int, str]]]:
    """A random ``S_m``-sparse *combined* operator ``A = sum_i c_i P_i``,
    as ``num_terms`` random real coefficients paired with ``num_terms``
    random Pauli specs (each drawn from ``random_capped_xy_spec``, so every
    term's X/Y-support size is in ``{min_xy, ..., m}``, i.e. in
    ``S_m = {a : |a| <= m}``). Returns ``(coeffs, specs)`` -- a plain,
    qiskit-free representation -- rather than a built operator object; see
    "WHY (coeffs, specs) AND NOT A BUILT OPERATOR" below.

    WHY THIS IS GUARANTEED ``S_m``-SPARSE (Definition 5.3 of
    ``SEEQST_shadows4.pdf`` / ``notes/working/seeqst_sparse_tuning.tex``):
    S-sparsity is a purely LINEAR condition -- "``a`` not in ``S`` implies
    ``Tr(A P_{a,b}) = 0`` for all ``b``" -- so the set of ``S``-sparse
    operators is EXACTLY the linear span of ``{P_{a,b} : a in S, b
    arbitrary}``. Any real-coefficient sum of terms each individually drawn
    with its X/Y-support in ``S`` is therefore automatically ``S``-sparse,
    with no further argument needed -- this is exactly your "generate
    eligible strings, then take a random superposition of them" idea, and
    it's correct regardless of which eligible terms you include or what
    coefficients you use.

    WHY SAMPLE ``num_terms`` TERMS RATHER THAN ENUMERATE *EVERY* ELIGIBLE
    TERM: the full ``S_m``-sparse basis has ``2**n * sum_{k=0}^{m} C(n,
    k)`` elements (every ``a`` with ``|a|<=m``, times EVERY ``b`` in
    ``{0,1}^n`` independently of ``a``) -- already ``2**10 * sum_{k=0}^{5}
    C(10,k) = 1024 * 638 ~= 6.5e5`` terms at just ``n=10, m=5``, and grows
    exponentially in ``n``. Enumerating that full basis is wasteful (most
    physically interesting sparse operators -- Hamiltonians, few-body
    observables, e.g. ``common/hamiltonians.py``'s Schwinger model -- have
    a small, FIXED number of terms, not "every possible term with bounded
    X/Y-support") and gives you no control over how large the resulting
    operator actually is. Sampling ``num_terms`` random eligible terms is
    the practical way to draw from this family at a size YOU choose, and,
    by the linearity argument above, is exactly as valid an ``S_m``-sparse
    operator as using the full basis would be -- just far cheaper, and
    closer to what a "sparse Hamiltonian" means in practice.

    Duplicate ``(a, b)`` draws (the same full Pauli string drawn twice) are
    left as two separate ``(coeff, spec)`` entries rather than merged --
    harmless for every linear use of the result (evaluating ``A``'s true
    expectation value, or estimating it via classical shadows term-by-term
    and combining with the same coefficients, both just sum contributions
    and get the right answer either way); merge them yourself first (e.g.
    via a ``dict`` keyed on ``spec_to_label``) if you specifically need a
    canonical, duplicate-free term list.

    WHY (coeffs, specs) AND NOT A BUILT OPERATOR: this module has no
    qiskit dependency (deliberately -- see the module docstring), so this
    function returns the same plain, portable representation
    ``common/hamiltonians.py::hamiltonian_terms`` converts INTO from a
    ``SparsePauliOp``, rather than pulling qiskit into this module to
    convert the other way. If you need an actual operator object:

        from qiskit.quantum_info import SparsePauliOp
        labels = [spec_to_label(spec, n) for spec in specs]
        A = SparsePauliOp(labels, coeffs).simplify()   # merges duplicates too

    and if you only need ``A``'s true expectation value on a state (no
    operator object required at all, by linearity):

        from qiskit.quantum_info import Pauli
        true_val = sum(
            c * state.expectation_value(Pauli(spec_to_label(spec, n))).real
            for c, spec in zip(coeffs, specs)
        )

    ``coeff_dist``: ``"normal"`` (iid standard normal, the default -- makes
    no particular assumption about coefficient scale, matching how
    ``notes/working/seeqst_sparse_tuning.tex``'s bound is stated purely in
    terms of ``||A||_infty`` rather than any specific coefficient
    distribution) or ``"uniform_pm1"`` (iid Uniform{-1, +1}, a random
    signed sum where every term contributes with equal magnitude).
    """
    if num_terms < 1:
        raise ValueError(f"num_terms={num_terms} must be >= 1")
    if coeff_dist not in ("normal", "uniform_pm1"):
        raise ValueError(f"coeff_dist={coeff_dist!r} must be 'normal' or 'uniform_pm1'")

    if coeff_dist == "normal":
        coeffs = rng.standard_normal(num_terms)
    else:
        coeffs = rng.choice((-1.0, 1.0), size=num_terms)

    specs = [
        random_capped_xy_spec(n, m, rng, min_xy=min_xy, rest_mode=rest_mode)
        for _ in range(num_terms)
    ]
    return coeffs, specs
