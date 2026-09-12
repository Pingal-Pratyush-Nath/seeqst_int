"""
Wiring into the two source repositories, without modifying them.

This module only *reads* from the repos below. Nothing here writes to,
patches, or otherwise touches their contents.

Repos (siblings of this `shadow_benchmark/` folder, under `codes/`):

- ``codes/predicting-quantum-properties``
    Hsin-Yuan Huang's reference implementation for
    "Predicting Many Properties of a Quantum System from Very Few
    Measurements" (arXiv:2002.08953) and the derandomization follow-up
    (arXiv:2103.07510). We reuse two pieces of it directly (unmodified):
    ``prediction_shadow.estimate_exp``, the textbook single-qubit
    random-Pauli ("local Clifford") hit-or-miss estimator -- our own
    ``ensembles/pauli_ensemble.py`` implements the same estimator directly
    (vectorized over many observables/shots), so this import is optional and
    used only in cross-check tests -- and
    ``data_acquisition_shadow.derandomized_classical_shadow``, their
    derandomization algorithm (paper's Algorithm 1: a deterministic greedy
    procedure that, given a fixed set of target Pauli observables, outputs a
    fixed schedule of single-qubit measurement bases). This one is NOT
    reimplemented locally -- exp_4 calls it directly, as the "derandomized
    Pauli" baseline compared against our own SEEQST-derandomized protocol
    (exp_2/exp_3/exp_4).

- ``codes/SEEQST``
    The SEEQST measurement-ensemble construction code. We import
    ``build_parallel_entangler_blocks`` from ``tools/setup.py`` directly:
    this is the function that builds the GHZ-block entangling circuits
    (RY90/RX90 + CNOT gate sequences) defining the SEEQST unitary ensemble
    ``S = {U_{I,E}, U_{I,O}}`` from Definition 3 of the SEEQST draft
    (papers/SEEQST_shadows.pdf). We do NOT reuse ``tools/setup.py``'s
    ``parse_circuit`` as-is, because it appends a terminal ``measure_all()``
    to every circuit (it's written for submission to real hardware), which
    makes the resulting ``QuantumCircuit`` non-unitary and unusable with
    Qiskit's statevector ``evolve``. Instead
    ``ensembles/seeqst_ensemble.py`` has a small local re-implementation of
    the same RY90/RX90/CNOT gate parsing, applied to the *unmodified* circuit
    text strings produced by ``build_parallel_entangler_blocks``.
"""

import sys
from pathlib import Path

# .../SEEQST/shadow_benchmark/common/external_paths.py -> .../SEEQST
_SEEQST_ROOT = Path(__file__).resolve().parents[2]
_CODES_DIR = _SEEQST_ROOT / "codes"

SEEQST_TOOLS_DIR = _CODES_DIR / "SEEQST" / "tools"
HUANG_SHADOWS_DIR = _CODES_DIR / "predicting-quantum-properties"

for _p in (SEEQST_TOOLS_DIR, HUANG_SHADOWS_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def import_seeqst_setup():
    """Import codes/SEEQST/tools/setup.py as a module (read-only)."""
    if not SEEQST_TOOLS_DIR.exists():
        raise FileNotFoundError(
            f"Expected SEEQST repo at {SEEQST_TOOLS_DIR}, but it was not found. "
            "This benchmark folder assumes it lives alongside the untouched "
            "'codes/SEEQST' and 'codes/predicting-quantum-properties' repos."
        )
    import setup as seeqst_setup  # noqa: WPS433 (intentional dynamic import)

    return seeqst_setup


def import_huang_prediction_shadow():
    """Import codes/predicting-quantum-properties/prediction_shadow.py (read-only)."""
    if not HUANG_SHADOWS_DIR.exists():
        raise FileNotFoundError(
            f"Expected Huang shadows repo at {HUANG_SHADOWS_DIR}, but it was not found."
        )
    import prediction_shadow  # noqa: WPS433

    return prediction_shadow


def import_huang_data_acquisition_shadow():
    """Import codes/predicting-quantum-properties/data_acquisition_shadow.py
    (read-only). Exposes ``derandomized_classical_shadow`` (used by exp_4)
    and ``randomized_classical_shadow`` (unused here -- our own ensembles
    implement randomized sampling directly)."""
    if not HUANG_SHADOWS_DIR.exists():
        raise FileNotFoundError(
            f"Expected Huang shadows repo at {HUANG_SHADOWS_DIR}, but it was not found."
        )
    import data_acquisition_shadow  # noqa: WPS433

    return data_acquisition_shadow
