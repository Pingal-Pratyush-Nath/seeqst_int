"""Shared SEEQST circuit-construction machinery, used by every SEEQST-family
ensemble (``seeqst_ensemble.py``, ``seeqst_binomial_ensemble.py``,
``seeqst_unifsize_ensemble.py``).

This is pure refactor-for-reuse: originally ``seeqst_ensemble.py`` owned all
of this itself (see git history / the module docstring there for the full
provenance of ``build_parallel_entangler_blocks``). Once a second and third
SEEQST-family ensemble needed the same "which qubits get GHZ-entangled" gate
sequences, keeping three independent copies of the parsing + caching logic
became a real risk, not just duplication: see ``qubit_subset_to_block``
below, which is the one place a convention mismatch between ensembles would
silently corrupt an inverse-map weight (see its own docstring). Centralizing
it means every SEEQST-family ensemble shares both the SAME conversion
function and the SAME circuit cache (keyed on ``(n, block, branch)``, which
does not depend on which ensemble/probability-distribution asked for it, so
sharing the cache across ensembles strictly improves hit rate over one cache
per ensemble instance).

``SEEQSTEnsemble`` (the original flat/uniform-over-all-subsets ensemble) was
refactored to import from here with NO intended behavior change; this is
checked directly by ``tests/verify_seeqst_binomial_and_unifsize.py`` (a
fixed-seed before/after snapshot comparison).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from common.external_paths import import_seeqst_setup

_seeqst_setup = import_seeqst_setup()

# Module-level (i.e. process-wide singleton) caches, intentionally shared by
# every ensemble instance/class that imports this module -- see the module
# docstring above for why that's a deliberate choice, not an oversight.
_text_cache: dict[int, list[list[str]]] = {}
_circuit_cache: dict[tuple[int, int, int], QuantumCircuit] = {}


def parse_circuit_text(text: str, n: int) -> QuantumCircuit:
    """Local re-implementation of ``codes/SEEQST/tools/setup.py``'s
    ``parse_circuit`` gate parsing, minus the terminal ``measure_all()`` --
    see ``seeqst_ensemble.py``'s original module docstring (preserved there)
    for why that terminal call has to be dropped for statevector ``evolve``
    to work.
    """
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
        elif gate_name == "H":
            qc.h(qubit_indices[0])
        else:
            raise ValueError(f"Unsupported gate in SEEQST circuit text: {gate_name}")
    return qc


def texts_for_n(n: int) -> list[list[str]]:
    """All 2**n blocks' circuit-text pairs for this ``n``, built once (via
    the external ``build_parallel_entangler_blocks``) and cached forever --
    shared by every SEEQST-family ensemble, not just the caller."""
    if n not in _text_cache:
        blocks = list(range(2**n))
        _text_cache[n] = _seeqst_setup.build_parallel_entangler_blocks(blocks, n)
    return _text_cache[n]


def circuit_for(n: int, block: int, branch: int) -> QuantumCircuit:
    """The (parsed, cached) circuit for entangler block ``block`` (see
    ``qubit_subset_to_block``), branch 0 or 1 (branch is meaningless -- and
    collapses to the single available text -- when the block's active-qubit
    set is empty; see ``build_parallel_entangler_blocks``)."""
    key = (n, block, branch)
    if key not in _circuit_cache:
        texts = texts_for_n(n)[block]
        text = texts[branch] if len(texts) > 1 else texts[0]
        _circuit_cache[key] = parse_circuit_text(text, n)
    return _circuit_cache[key]


def num_branches(n: int, block: int) -> int:
    return len(texts_for_n(n)[block])


def qubit_subset_to_block(subset, n: int) -> int:
    """Convert a REAL qubit subset (an iterable of qubit indices in
    ``0..n-1`` -- e.g. the X/Y-support of a Pauli spec, or a subset drawn by
    an ensemble's own sampling distribution) into the integer ``block``
    index that ``build_parallel_entangler_blocks``/``texts_for_n`` expects.

    This is THE one place that has to get ``codes/SEEQST/tools/setup.py``'s
    bit convention exactly right, and it is worth spelling out why: that
    function does ``bin_str = format(block, f'0{num_qubits}b')`` (a
    most-significant-bit-first string) and then ``active_qubits = [i for i,
    bit in enumerate(bin_str[::1]) if bit == '1']`` -- note ``[::1]`` is a
    no-op slice (NOT a reversal), so despite that code's own "LSB = qubit 0"
    comment, position ``i`` in the MSB-first string decodes to place value
    ``2**(n-1-i)``, i.e. qubit ``i`` <-> bit position ``n-1-i`` of the
    integer. Hence the inverse here: qubit ``i`` contributes ``2**(n-1-i)``.

    Getting the MSB/LSB choice itself wrong would NOT silently break
    anything for a per-qubit-EXCHANGEABLE sampling distribution (Binomial(q)
    with one shared q, or "uniform among subsets of a given size") -- any
    self-consistent bit-position permutation still samples a real-qubit
    subset with the correct marginal probability, since those distributions
    depend on a subset only through its cardinality. What would NOT be
    safe is two call sites disagreeing with EACH OTHER on the convention
    (they'd collide on ``circuit_for``'s cache keys while intending
    different real-qubit sets) -- which is exactly why every SEEQST-family
    ensemble goes through this single shared function rather than each
    reimplementing the conversion locally. See
    ``tests/verify_seeqst_binomial_and_unifsize.py`` for the Monte-Carlo
    check that empirical sampled-set frequencies match the closed-form
    probabilities this depends on.
    """
    return sum(2 ** (n - 1 - i) for i in subset)
