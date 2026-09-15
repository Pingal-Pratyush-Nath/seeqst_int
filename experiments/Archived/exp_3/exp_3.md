# exp_3 — Derandomizing SEEQST: circuit selection for a target Pauli

**Module:** `experiments/Archived/exp_3/exp_3.py` (renamed from `derandomized_seeqst.py`; imported by `experiments/exp_2/exp_2.py`, `experiments/exp_4/exp_4.py`, `tasks/exp_2_derandomized_scaling.py`, and `tasks/exp_4_derandomized_comparison.py` — not run directly) · **Verification:** `tests/verify_derandomized_seeqst.py`

> **Not deprecated.** This lives under `Archived/` purely for folder
> organization — exp_3 never had its own CLI or `results/` folder (unlike
> exp_0/1/2/4), so it doesn't read as a "peer" run file next to those. But
> it is a live, actively-imported dependency of both exp_2 and exp_4 (see
> the import lines in `tasks/exp_2_derandomized_scaling.py` and
> `tasks/exp_4_derandomized_comparison.py`). Moving or deleting this file
> will break both of those.

## What it does

exp_3 is not a run file — it's the derandomization utility that exp_2's
protocol is built on. Given a specific Pauli string `P` (as a `spec`: a dict
`{qubit: 'X'|'Y'|'Z'}`, identity qubits omitted — same convention as
`common/pauli_utils.py`), it answers: **which SEEQST circuit measures `P`
exactly?** I.e. which `(I, branch)` — subset of qubits `I` to entangle into a
GHZ block, and which of the two conjugate branches (`E` or `O`) — makes
`U P U†` diagonal in the computational basis, so a single shot gives the
deterministic value `±β(P)` with certainty instead of a "hit or miss"
outcome. This is exactly the "derandomization" ingredient (Huang-Kueng-Preskill
terminology) that exp_2 needs, used in place of drawing `(I, branch)` uniformly
at random the way `SEEQSTEnsemble.sample_snapshot` does.

## Background: the SEEQST ensemble

The SEEQST unitary ensemble `S = {U_{I,E}, U_{I,O}}_{I ⊆ [n]}` (Definition 3,
`papers/SEEQST_shadows.pdf`) is built around a subset `I ⊆ [n]`: qubits in `I`
get entangled into one of two conjugate GHZ bases (`GHZ_E`, with
`ω ∈ {+1,−1}`, or `GHZ_O`, with `ω ∈ {+i,−i}` — Definition 2), while qubits
*not* in `I` are measured directly in the computational (Z) basis. For a
random draw from the full ensemble, a given Pauli `P` is only measured
*exactly* for **some** of the `2^(n+1)` choices of `(I, branch)` — see
`ensembles/seeqst_ensemble.py`'s `inverse_weight` / the paper's Proposition 9.
exp_3 answers the reverse question: given `P`, which `(I, branch)` works?

Built from scratch (fresh logic, not routed through the `SEEQSTEnsemble`
class) — it reuses `build_parallel_entangler_blocks` from
`codes/SEEQST/tools/setup.py` unchanged (same as
`ensembles/seeqst_ensemble.py` does) for the actual gate sequences, but
chooses *which* `(I, branch)` to build a circuit for independently.

## The derandomization rule

Derived directly from Propositions 6–9 of the SEEQST draft. Write
`J = {qubits where P is X or Y}`.

**Case 1 — `P` has at least one X/Y factor (`J` non-empty).** Proposition 9's
proof shows the *unique* valid subset is `I = J` exactly:

- every qubit with an X/Y factor must be inside `I`, because qubits outside
  `I` are measured directly in the Z basis and can only tolerate `{I, Z}`
  factors (Proposition 6) — so `J ⊆ I`;
- `I` can't be any larger than `J` either: an extra qubit inside `I` where `P`
  acts as identity would break the "`P` restricted to `I` is entirely X/Y,
  none identity" condition that Propositions 7/8 require for a non-zero
  result — so `I ⊆ J`.

Having fixed `I = J`, exactly one of the two branches works (Propositions 7
vs. 8 differ only in requiring even vs. odd Y-parity of `P` restricted to
`I`):

- even number of Y's in `J` → branch `E`
- odd number of Y's in `J` → branch `O`

**Case 2 — `P` is pure `{I, Z}`-type (`J` empty; includes `P` = identity).**
Propositions 7 and 8 give the *identical* condition here (doesn't depend on
Y-parity at all): `P` is measurable for *any* subset `I` with an even number
of `P`'s Z's inside it, and for *both* branches. The simplest such choice is
`I = ∅` (0 of `P`'s Z's inside it, trivially even) — i.e. no entangling gates
at all, just measure every qubit directly in the computational basis. That's
the canonical choice `select_subset_and_branch` returns for this case.

## Which branch index is E and which is O?

Definition 3 in the draft leaves the explicit `U_{I,E}`/`U_{I,O}` gate
sequences as "TODO", so the E/O ↔ RY90/RX90 correspondence isn't written down
in the paper. It was pinned down numerically instead: build both branches'
circuits for a couple of small examples via `build_parallel_entangler_blocks`
and check which one diagonalizes an even-Y-parity Pauli (Proposition 7 /
`GHZ_E`) vs. an odd-Y-parity one (Proposition 8 / `GHZ_O`). Result: branch
index 0 (the RY90-based sequence) is `GHZ_E`; branch index 1 (the RX90-based
sequence) is `GHZ_O`. This matches the physical intuition that RY rotations
produce real-valued superpositions (matching `GHZ_E`'s `ω ∈ {+1,−1}`), while
RX rotations produce complex/imaginary ones (matching `GHZ_O`'s `ω ∈ {+i,−i}`).

## Bit-convention note

`build_parallel_entangler_blocks(blocks, n)` takes each "block" as an integer
and recovers the active-qubit subset via `format(block, f'0{n}b')` read left
to right, i.e. string index `i` corresponds to bit position `n-1-i` of the
integer — *not* "LSB = qubit 0" as that repo's own comment claims (verified
directly: `block=8` with `n=4` gives `active_qubits=[0]`, not `[3]`). Since
every other module in this codebase uses the opposite convention for qubit
indices (qiskit's rightmost-char-is-qubit-0 — see
`common/pauli_utils.spec_to_label`), the integer to pass in for a target
subset `I` is

```
block = sum(2**(n - 1 - q) for q in I)
```

(the "reverse the bit order" map — confirmed empirically to recover the
intended qubit subset for both single-qubit and multi-qubit `I`).

## Functions

- **`select_subset_and_branch(spec, n) -> (subset, branch)`** — the
  derandomization rule itself: given a Pauli `spec`, returns the
  `(I, branch)` pair for which `U_{I,branch}` measures `spec` exactly.
- **`circuit_for_subset_branch(subset, branch, n) -> QuantumCircuit`** —
  builds `U_{subset,branch}` directly from an already-chosen pair, without
  re-deriving it from a Pauli spec. Split out so callers that group many
  Pauli specs onto a smaller number of distinct `(subset, branch)` circuits
  (exp_2) can build each distinct circuit exactly once instead of once per
  observable.
- **`circuit_for_pauli(spec, n) -> QuantumCircuit`** — convenience wrapper:
  `select_subset_and_branch` then `circuit_for_subset_branch` in one call.
- **`_block_index(subset, n)`** — internal: converts a qubit subset to the
  integer block argument, per the bit-convention note above.
- **`_parse_circuit_text(text, n)`** — internal: minimal local
  re-implementation of `setup.py`'s gate-token parsing (independent copy of
  `ensembles/seeqst_ensemble.py`'s version, so this module has no dependency
  on the `SEEQSTEnsemble` class). Skips `setup.parse_circuit`'s terminal
  `measure_all()`, which would make the circuit non-unitary.

## How exp_2 uses it

`tasks/exp_2_derandomized_scaling.py` calls `select_subset_and_branch` once
per drawn observable to get its `(subset, branch)` key, groups observables
that share a key (so they share a circuit), and calls
`circuit_for_subset_branch` once per *distinct* key — not once per
observable — before drawing the shared shot stream. See `exp_2.md` for the
full protocol this feeds into.

## Verification

`tests/verify_derandomized_seeqst.py` sweeps random Pauli specs (all weights,
both pure-Z-type and X/Y-containing) across `n = 2..6` and checks, for every
one, that `U P U†` is exactly diagonal (off-diagonal Frobenius norm ≈ 0) with
every diagonal entry exactly `±1`. Last run: **115/115 specs passed**, max
off-diagonal norm `1.52e-15` (floating-point noise). Run it with:

```bash
cd shadow_benchmark
python tests/verify_derandomized_seeqst.py
```

Note on what "exactly measurable" means here: every conjugated Pauli is
unitarily similar to `P`, so its eigenvalues are still `±1` regardless — the
"exactly measurable" property is about `U P U†` being *diagonal at all*, not
about eigenvalue magnitude. The separate `β(P)` factor
(`SEEQSTEnsemble.inverse_weight`: 2 for pure-Z-type, `2^(n+1)` for anything
with an X/Y factor) rescales a raw ±1 outcome into an unbiased single-shot
estimate of `Tr(P ρ)` under the *random*-sampling protocol (exp_1); it is not
used by the derandomized protocol at all (see `exp_2.md`'s "no beta
rescaling" section) and is not the eigenvalue of `U P U†` itself.
