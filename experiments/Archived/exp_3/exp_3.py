"""Derandomized SEEQST: given a specific Pauli string P, return the SEEQST
circuit that measures P *deterministically* (no "hit or miss").

STATUS NOTE: this module lives under ``experiments/Archived/`` because it
was never a standalone run file (no CLI, no results/ folder of its own) --
but unlike the rest of ``Archived/``, it is NOT deprecated: it is actively
imported by ``tasks/exp_2_derandomized_scaling.py``,
``tasks/exp_4_derandomized_comparison.py``, and
``tests/verify_derandomized_seeqst.py``. If you're considering deleting or
further relocating this file, check those three call sites first.

Background
--------------------------------------------------------------------------------
The SEEQST unitary ensemble S = {U_{I,E}, U_{I,O}}_{I subseteq [n]} (Definition
3, ``papers/SEEQST_shadows.pdf``) is built around a subset I subseteq [n]:
qubits in I get entangled into one of two conjugate GHZ bases (GHZ_E, with
``omega in {+1,-1}``, or GHZ_O, with ``omega in {+i,-i}`` -- Definition 2),
while qubits NOT in I are measured directly in the computational (Z) basis.

For a random draw from the full ensemble, a given Pauli P is only measured
*exactly* (i.e. U P U^dagger comes out diagonal in the computational basis,
so the single-shot estimate is +-beta(P) with certainty, instead of 0 on a
"miss") for SOME of the 2^{n+1} choices of (I, branch) -- see
``ensembles/seeqst_ensemble.py``'s ``inverse_weight`` / the paper's
Proposition 9. This module answers the reverse question: given P, which
(I, branch) -- and hence which circuit -- makes P measurable? This is exactly
the "derandomization" (Huang-Kueng-Preskill terminology) ingredient exp_2
needs, in place of drawing (I, branch) uniformly at random the way
``SEEQSTEnsemble.sample_snapshot`` does. Built from scratch here (fresh
logic, no dependency on the ``SEEQSTEnsemble`` class) as requested.

The rule, derived directly from Propositions 6-9 of the draft
--------------------------------------------------------------------------------
Write P as a ``spec``: a dict ``{qubit: 'X'|'Y'|'Z'}`` (identity qubits
omitted, same convention as ``common/pauli_utils.py``). Let
``J = {qubits where P is X or Y}``.

Case 1 -- P has at least one X/Y factor (J non-empty). Proposition 9's proof
shows the *unique* valid subset is I = J exactly:
  - every qubit with an X/Y factor must be inside I, because qubits OUTSIDE I
    are measured directly in the Z basis and can only tolerate {I,Z} factors
    (Proposition 6) -- so J subseteq I;
  - I can't be any larger than J either: an extra qubit inside I where P acts
    as identity would break the "P restricted to I is entirely X/Y, none
    identity" condition that Propositions 7/8 require for a non-zero result
    -- so I subseteq J.
Having fixed I = J, exactly one of the two branches works (Propositions 7 vs
8 differ only in requiring even vs. odd Y-parity of P restricted to I):
  - even number of Y's in J -> branch E
  - odd number of Y's in J  -> branch O

Case 2 -- P is pure {I,Z}-type (J empty; includes P = identity). Now
Propositions 7 and 8 give the *identical* condition (doesn't depend on
Y-parity at all): P is measurable for ANY subset I with an even number of
P's Z's inside it, and for BOTH branches. The simplest such choice is
I = empty set (0 of P's Z's inside it, trivially even) -- i.e. no entangling
gates at all, just measure every qubit directly in the computational basis.
That's the canonical choice used below.

Which branch index is E and which is O?
--------------------------------------------------------------------------------
Definition 3 in the current draft leaves the explicit U_{I,E}/U_{I,O} gate
sequences as "TODO", so the E/O <-> RY90/RX90 correspondence isn't written
down anywhere in the paper. We pinned it down numerically instead: built
both branches' circuits for a couple of small examples via
``build_parallel_entangler_blocks`` and checked which one diagonalizes an
even-Y-parity Pauli (Proposition 7 / GHZ_E) vs. an odd-Y-parity one
(Proposition 8 / GHZ_O). Result: branch index 0, the RY90-based sequence, is
GHZ_E; branch index 1, the RX90-based sequence, is GHZ_O. This matches the
physical intuition that RY rotations produce real-valued superpositions
(matching GHZ_E's omega in {+1,-1}), while RX rotations produce
complex/imaginary ones (matching GHZ_O's omega in {+i,-i}).

We reuse ``build_parallel_entangler_blocks`` from
``codes/SEEQST/tools/setup.py`` UNCHANGED (same as
``ensembles/seeqst_ensemble.py`` does) -- that is still the actual
definition of the gate sequence; we're only choosing WHICH (I, branch) to
build a circuit for, from scratch, rather than sampling one at random or
going through the ``SEEQSTEnsemble`` class.

Bit-convention note (easy to get wrong, so spelled out explicitly)
--------------------------------------------------------------------------------
``build_parallel_entangler_blocks(blocks, n)`` takes each "block" as an
integer and recovers the active-qubit subset via
``format(block, f'0{n}b')`` read LEFT TO RIGHT, i.e. string index i
corresponds to bit position ``n-1-i`` of the integer -- NOT "LSB = qubit 0"
as that repo's own comment claims (verified directly: block=8 with n=4 gives
active_qubits=[0], not [3]). Since every other module in this codebase uses
the opposite convention for qubit indices (qiskit's rightmost-char-is-qubit-0
-- see ``common/pauli_utils.spec_to_label``), we have to convert: the
integer to pass in for a target subset I is

    block = sum(2**(n - 1 - q) for q in I)

(equal to the "reverse the bit order" map -- confirmed empirically to
recover the intended qubit subset for both single-qubit and multi-qubit I).
"""

from __future__ import annotations

import sys
from pathlib import Path

# This module lives at shadow_benchmark/experiments/Archived/exp_3/exp_3.py
# -- three levels below the shadow_benchmark root (common/ lives there).
# Callers that already have the root on sys.path (exp_2.py, exp_4.py,
# tests/*.py) make this a no-op; this line just makes the module
# self-sufficient if it's ever imported or run some other way.
# parents[0]=exp_3/, parents[1]=Archived/, parents[2]=experiments/,
# parents[3]=shadow_benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from qiskit import QuantumCircuit

from common.external_paths import import_seeqst_setup

_seeqst_setup = import_seeqst_setup()


def _block_index(subset: frozenset[int], n: int) -> int:
    """Integer 'block' argument for ``build_parallel_entangler_blocks`` that
    corresponds to qubit subset ``subset``, under THIS codebase's
    qubit-index convention (see module docstring's "Bit-convention note")."""
    return sum(2 ** (n - 1 - q) for q in subset)


def _parse_circuit_text(text: str, n: int) -> QuantumCircuit:
    """Minimal, local re-implementation of setup.py's gate-token parsing
    (independent copy of ``ensembles/seeqst_ensemble.py``'s
    ``_parse_circuit_text``, so this module has no dependency on the
    ``SEEQSTEnsemble`` class, per the request to build this from scratch).
    Skips ``setup.parse_circuit``'s terminal ``measure_all()``, which would
    make the circuit non-unitary."""
    qc = QuantumCircuit(n)
    for op in text.split(")"):
        op = op.strip().strip("(")
        if not op:
            continue
        gate_name, _, qubit_str = op.partition(":")
        qubit_indices = [int(x) for x in qubit_str.split(",")]
        if gate_name == "RX90":
            qc.rx(np.pi / 2, qubit_indices[0])
        elif gate_name == "RY90":
            qc.ry(np.pi / 2, qubit_indices[0])
        elif gate_name == "CNOT":
            qc.cx(qubit_indices[0], qubit_indices[1])
        else:
            raise ValueError(f"Unsupported gate in SEEQST circuit text: {gate_name}")
    return qc


def select_subset_and_branch(spec: dict[int, str], n: int) -> tuple[frozenset[int], str]:
    """The derandomization rule itself (see module docstring): given a Pauli
    ``spec`` (``{qubit: 'X'|'Y'|'Z'}``), return the ``(I, branch)`` --
    ``branch in {'E', 'O'}`` -- for which the SEEQST circuit U_{I,branch}
    measures ``spec`` exactly (i.e. U P U^dagger is diagonal in the
    computational basis)."""
    xy_qubits = frozenset(q for q, letter in spec.items() if letter in ("X", "Y"))
    if xy_qubits:
        # Case 1: I = J = support of the X/Y factors; branch by Y-parity.
        y_count = sum(1 for q in xy_qubits if spec[q] == "Y")
        branch = "E" if y_count % 2 == 0 else "O"
        return xy_qubits, branch
    # Case 2: pure {I,Z}-type (including spec == {}, the identity). Any
    # subset with an even number of this P's Z's inside it works, for either
    # branch -- the empty subset (no entangling gates at all) is always
    # valid, and the simplest possible choice.
    return frozenset(), "E"


def circuit_for_subset_branch(subset: frozenset[int], branch: str, n: int) -> QuantumCircuit:
    """Build the SEEQST circuit U_{subset,branch} directly from an
    already-chosen ``(subset, branch)`` pair (``branch in {'E', 'O'}``),
    without re-deriving it from a Pauli spec. Split out from
    ``circuit_for_pauli`` so callers that group many Pauli specs onto a
    smaller number of distinct ``(subset, branch)`` circuits (e.g. exp_2)
    can build each distinct circuit exactly once."""
    block = _block_index(subset, n)
    texts = _seeqst_setup.build_parallel_entangler_blocks([block], n)[0]
    branch_idx = 0 if branch == "E" else 1
    text = texts[branch_idx] if len(texts) > 1 else texts[0]
    return _parse_circuit_text(text, n)


def circuit_for_pauli(spec: dict[int, str], n: int) -> QuantumCircuit:
    """The SEEQST circuit U_{I,branch} (built from scratch: fresh logic, not
    routed through ``SEEQSTEnsemble``/``enumerate_unitaries``) that measures
    the Pauli ``spec`` exactly: U P U^dagger is diagonal in the
    computational basis, so a single shot with this circuit gives the
    single-shot shadow estimate +-beta(P) with certainty rather than a
    "hit or miss" outcome. See ``select_subset_and_branch`` for how
    ``(I, branch)`` is chosen, and the module docstring for the derivation.
    """
    subset, branch = select_subset_and_branch(spec, n)
    return circuit_for_subset_branch(subset, branch, n)
