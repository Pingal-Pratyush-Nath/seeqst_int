# exp_12 — estimating a real Hamiltonian's ground-state energy from shadows

## What's different from exp_9–exp_11

Every earlier experiment draws many random synthetic observables against many
random states to build up a statistical picture of typical ensemble behavior.
exp_12 instead targets one fixed, physically meaningful problem:

1. Build a concrete **numeric** Jordan-Wigner electronic Hamiltonian via the
   new `common/jw_hamiltonian.py` module — the exact same algebra verified
   against `tests/JW_transformation.ipynb` (canonical anticommutation
   relations, Hermiticity for symmetric integrals, exact match to the known
   free-fermion spectrum — see that module's docstring and the conversation
   where it was checked).
2. Diagonalize it exactly (`numpy.linalg.eigh`, cheap at `n<=6`: a 64×64
   dense matrix) to get the true ground state and `E_true`.
3. Measure the ground state with each ensemble's classical shadows, combine
   every Pauli term's per-shot estimate with the Hamiltonian's own
   coefficients (exp_11's "combine-then-estimate" design), and see how
   quickly `|E_hat(N_sample) - E_true|` shrinks.

## The Hamiltonian

`--integrals demo` (the only option so far) calls `jw_hamiltonian.demo_integrals`:
a small, **hand-picked, not-a-real-molecule** instance chosen only to give the
pipeline something concrete and reproducible to run on, at whatever `n` you
ask for:

- on-site energies `h[p,p] = p - (n-1)/2` — a linear ladder centered at 0, so
  the non-interacting ground state fills roughly the lower half of the
  orbitals (a nontrivial filling, not all-empty or all-full);
- nearest-neighbour hopping `h[p,p+1] = h[p+1,p] = -0.5`;
- nearest-neighbour density-density interaction `v[p,q,q,p] = v[q,p,p,q] = 0.4`
  for `|p-q|==1` — via the identity `a_p^dagger a_q^dagger a_q a_p = n_p n_q`
  (`p != q`), this two-body term works out to exactly
  `sum_{p<q, |p-q|=1} 0.4 * n_p * n_q`, a standard nearest-neighbour
  Hubbard-style coupling.

Both `h` and `v` satisfy the standard chemist symmetries needed for `H` to be
Hermitian by construction; `jw_electronic_hamiltonian` raises loudly if that
is ever broken (checks every term's coefficient is real to within `atol`).
At `n=6` this produces the same kind of term-count/weight-distribution
structure discussed for `JW_transformation.ipynb` (a few hundred nonzero
Pauli terms, XY-weight only ever 0/2/4).

## Ensembles

```
pauli                  PauliEnsemble()                (beta = 3^weight(P))
clifford               CliffordEnsemble()              (beta = 2^n+1, exact)
seeqst_uniform         SEEQSTEnsemble()                (flat/uniform SEEQST, q=1/2)
seeqst_binomial_q<q>   SEEQSTBinomialEnsemble(q)       (one per --q-values entry)
```

Unlike exp_9–exp_11, there is **no** automatic `q=m/n` tuned ensemble here —
`H` mixes many different X/Y-weights simultaneously (0, 2, and 4 all appear
with substantial term counts), so there's no single "the" `m` to tune to.
Instead, `--q-values` takes any list of `q`'s you want to compare directly.
Default: `4/6` and `2/6`, matching the original request.

## Repeats, not state/observable layers

Because both the state (the ground state) and the observable (`H` itself)
are fixed, there's nothing left to randomize except which measurement
outcomes happen to come up. `--num-repeats` (default 20) replaces
exp_9–exp_11's `num_state_repeats * num_observable_repeats` as the sole
source of the error bars in the output plot — each repeat draws a fresh
`max(n_samples)` shadow snapshots per ensemble and computes its own
checkpointed energy trajectory.

## Metric

With only one observable, RMSE/MaxError collapse to a single number:
`abs_error(N_sample) = |E_hat(N_sample) - E_true|`, averaged (± std-dev)
across repeats. Plotted log-log vs. `N_sample`, one line per ensemble, in
`energy_error_vs_nsample.png`.

## Usage

```bash
python experiments/exp_12/exp_12.py --quick
python experiments/exp_12/exp_12.py --n 6 --q-values 0.6666666667 0.3333333333
python experiments/exp_12/exp_12.py --n 6 --q-values 0.6666666667 0.3333333333 --num-repeats 30
python experiments/exp_12/exp_12.py --n 6 --estimator median_of_means
```

## Validation

qiskit is not installable in either the cloud sandbox or the local device
sandbox this was authored from, so the qiskit-touching pieces (building the
`Statevector`, `evaluate_snapshots`) could not be executed end-to-end from
there. What *was* checked, with plain numpy (no qiskit/sympy needed):

- the underlying Jordan-Wigner algebra: canonical anticommutation relations
  for the resulting ladder operators, exact Hermiticity of `H` for
  symmetric integrals, and exact agreement with the analytically known
  non-interacting ("subset-sum of single-particle eigenvalues") spectrum
  for `n=3,4,5`;
- the spec re-indexing (`common/jw_hamiltonian.py`'s internal left-to-right
  string → this codebase's `spec[i]` convention) checked bit-for-bit
  against `pauli_utils.spec_to_label`'s actual output, by building the same
  dense matrix two independent ways and comparing element-wise (`n=3,4,5`);
- `demo_integrals`'s Hermitian symmetry (`h` symmetric, `v`'s 4-fold chemist
  symmetry) holds by construction;
- `_checkpoint_estimates` (mean and median-of-means) and the
  `term_shots @ coeffs` combine step were cross-checked against independent
  from-scratch reference implementations using fake random `term_shots`
  data (same style of validation exp_11 used).

Please run `--quick` first on your machine and sanity-check that the printed
`E_true` looks like a reasonable ground energy for the Hamiltonian size
before committing to a full run.
