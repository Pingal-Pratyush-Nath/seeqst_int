# exp_11 — exp_9's hyperparameters, estimating COMBINED *S_m*-sparse operators

**Script:** `experiments/exp_11/exp_11.py` · **Logic:** `tasks/exp_11_sparse_operator_mom.py` ·
**Output:** `results/exp_11/v<N>/`

## What this is

`exp_11` uses **exp_9's exact hyperparameter set** — same four-ensemble
roster, same `q=m/n` tuning rule, same selectable point estimator, same
two-layer randomization design, same RMSE/MaxError metrics — applied to a
genuinely different observable. Where exp_9 tracks `num_observables`
*separate single-Pauli-string* observables, exp_11 tracks `num_observables`
independent random **combined operators**:

```
A = Σ_{i=1}^{num_terms} c_i P_i,     each P_i with X/Y-weight in {min_xy,...,m}
```

via `pauli_utils.random_sparse_operator` (added in the previous session
specifically for this). Any such real-coefficient superposition of terms
whose X/Y-support lies in `S_m = {a : |a| ≤ m}` is automatically
`S_m`-sparse (Definition 5.3 of `SEEQST_shadows4.pdf` /
`notes/working/seeqst_sparse_tuning.tex`) — a purely linear fact, since
`S`-sparsity is exactly the condition "every nonzero Pauli coefficient has
`a ∈ S`", and that set of operators is a linear subspace. This is the
first experiment in this project to estimate that actual object, rather
than a *list* of separately-scored weight-`≤m` observables (exp_8, exp_10)
or a single fixed-weight string (exp_7, exp_9).

## How a combined operator is estimated from shadows

Every `ShadowEnsemble.evaluate_snapshots` call already returns, per
snapshot `s` and Pauli term `P_i`, a single-shot estimate `ô_i^(s)` with
`E[ô_i^(s)] = tr(P_i ρ)` exactly. Since expectation is linear,

```
E[ Σ_i c_i · ô_i^(s) ] = Σ_i c_i · tr(P_i ρ) = tr(A ρ)
```

so the *same* per-shot linear combination the operator itself uses is
already an unbiased single-shot estimator of `A` — one physical
measurement round predicts the whole combined operator at once, same as
classical shadows always do for many observables. Concretely, `exp_11`:

1. Flattens every term of every combined operator into one spec list and
   calls `evaluate_snapshots` once per ensemble (shape
   `(max_n_sample, num_observables · num_terms)`).
2. For each operator, forms **one** per-shot column as the weighted sum
   (its own `coeffs`) of its own term columns — shape
   `(max_n_sample, num_observables)`, the same shape exp_9/exp_10 feed into
   the point estimator.
3. Passes that combined-per-shot array into `_checkpoint_estimates`
   **unchanged** from exp_9/exp_10.

Step 3 is a deliberate choice, not a simplification: for `"mean"`,
combine-then-average and average-then-combine are exactly equal (linearity
of the mean). For `"median_of_means"` they are **not** — median is
nonlinear — and combining first (as done here) is the standard
classical-shadows treatment of a linear combination: HKP's own
median-of-means estimator acts on a single scalar random variable per
checkpoint, and that scalar is already the per-shot estimator of the full
combined operator, not of any individual term. This also lets
median-of-means exploit cancellation between terms of opposite sign,
rather than pessimistically robustifying each term in isolation before
summing.

## A tuning caveat worth knowing before reading results

`notes/working/seeqst_sparse_tuning.tex`'s `q*=m/n` theorem (for `m≤n/2`,
see `experiments/exp_8/exp_8.md`'s two-regime derivation) is about the
shadow-norm bound of the operator formed by summing **every** eligible term
in the full `S_m` basis — not a random subset of it. `exp_11`'s operators
sum only `num_terms` randomly sampled eligible terms. Reusing exp_9/exp_10's
`q=m/n` rule here (`build_ensembles` is byte-for-byte exp_9's) is a
reasonable, theory-motivated default — every term that *could* appear still
has X/Y-weight `≤ m`, governed by the same `β_q(k)` endpoint analysis
`exp_8.md` works out — but it is **not** a claim that `q=m/n` is proven
minimax-optimal for this exact random-subset-of-terms construction.
`hyperparameters.json` records `q_tuned_is_minimax_optimal_full_basis`
(`= 2·m ≤ n`) as a breadcrumb, named explicitly to flag that the claim it
encodes is about the full-basis object, not `exp_11`'s sampled one.

## New hyperparameters (none of these exist in exp_9)

| Flag | Meaning | Default |
|---|---|---|
| `--num-terms` | number of Pauli terms summed into EACH combined operator | 5 |
| `--min-xy` | floor on each term's X/Y-weight | 1 (matches exp_8/exp_10's own generator convention — keeps every term genuinely X/Y-bearing) |
| `--rest-mode` | `random`\|`z`\|`identity` — how each term's non-XY qubits are set | `random` (unlike exp_9's `--rest-mode`, which governed ONE exact-weight observable, this applies per-term inside the sum — see `pauli_utils.random_capped_xy_spec`) |
| `--coeff-dist` | `normal`\|`uniform_pm1` — each operator's real term coefficients | `normal` |
| `--q-binomial-untuned` | per-qubit Bernoulli inclusion probability `q` for the EXTRA `seeqst_binomial_untuned` ensemble (below) | `1/(n+1)` (exp_6/exp_8's own default untuned `q`) |

Every other flag (`--n`, `--m`, `--estimator`, `--mom-num-groups`,
`--mom-delta`, `--num-observables`, `--n-samples`, `--num-state-repeats`,
`--num-observable-repeats`, `--states`, `--seed`, `--version`, `--quick`,
`--quiet`) is exp_9's own CLI, unchanged, including all defaults.

## Selecting the binomial parameter directly

exp_9/exp_10/exp_11's `seeqst_binomial_tuned` never took `q` as a free
input — it's always computed as `q = m/n` inside `build_ensembles`, so the
only way to move it was indirectly through `--n`/`--m`. exp_11 now adds a
**fifth ensemble**, `seeqst_binomial_untuned`, which is the same
`SEEQSTBinomialEnsemble` class at a `q` you set directly with
`--q-binomial-untuned` — exactly the pattern `exp_8` already established
for its own untuned contrast ensemble. `seeqst_binomial_tuned` itself is
untouched (still `q=m/n`, not overridable) — this is a genuinely separate
ensemble line in the output, not a replacement, so you can compare a
freely-chosen `q` against the algorithmically-tuned one in the same run.
`--q-binomial-untuned` must be strictly between 0 and 1 (`q=0` or `1` makes
any X/Y-containing observable's inverse weight divide by zero — the same
constraint `SEEQSTBinomialEnsemble.__init__` itself enforces).

```bash
# q pinned to 0.5 (independent of n, m) for direct comparison against the tuned ensemble
python experiments/exp_11/exp_11.py --n 6 --m 5 --q-binomial-untuned 0.5

# default untuned q = 1/(n+1) = 1/7, alongside the tuned q = m/n = 5/6
python experiments/exp_11/exp_11.py --n 6 --m 5
```

## Validation

As with `exp_10.md`, qiskit is not installable in either this session's
cloud workspace or the local sandbox `device_bash` runs in, so an
end-to-end `python exp_11.py --quick` could not be run. What was checked
directly, without qiskit:

- `python3 -m py_compile experiments/exp_11/exp_11.py
  tasks/exp_11_sparse_operator_mom.py` — both parse cleanly.
- `pauli_utils.random_sparse_operator` itself was already verified in the
  previous session by decomposing a random combined operator's dense
  matrix over the full `4^n` Pauli basis and confirming every nonzero
  coefficient has X/Y-support `≤ m` (see `pauli_utils.py`'s docstring for
  the argument; the check was numerical, not just the linearity proof).
- The **one genuinely new mechanical step** — flattening every operator's
  terms into one spec list, calling the (qiskit-dependent)
  `evaluate_snapshots` once, then slicing and recombining each operator's
  own columns with its own coefficients — was extracted and run against a
  **fake** `term_shots` array (`np.random.standard_normal`, no qiskit
  needed) at `n=6, m=3, num_terms=4, M=5`, and cross-checked element-for-
  element against an independently-written second implementation of the
  same recombination. Shapes and values matched exactly
  (`(50, 20) -> (50, 5)` as expected, `np.allclose` on the two
  recombinations).
- `_checkpoint_estimates` is byte-for-byte exp_9/exp_10's already-validated
  version, unchanged.
- The new `--q-binomial-untuned` resolution/validation logic (default
  `1/(n+1)` when omitted, reject anything outside `(0,1)`) was extracted
  and unit-checked standalone: correct default at `n=6` (`1/7`), an
  explicit value passes through unchanged, and `0.0`, `1.0`, negative, and
  `>1` values all raise as expected.
- **Not yet run**: the actual `--quick` smoke test end-to-end, and any
  comparison of `seeqst_binomial_tuned`'s combined-operator performance
  against exp_9/exp_10's single/list-observable numbers at the same `m`.
  Run these once a Python environment with `qiskit==2.4.1` is available.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_11/exp_11.py --quick
python experiments/exp_11/exp_11.py --n 6 --m 5
python experiments/exp_11/exp_11.py --n 6 --m 5 --num-terms 8 --coeff-dist uniform_pm1
python experiments/exp_11/exp_11.py --n 6 --m 5 --estimator median_of_means
python experiments/exp_11/exp_11.py --n 7 --m 3 --version my_run_name
```


cd shadow_benchmark
python experiments/exp_11/exp_11.py \
    --n 6 \
    --m 5 \
    --num-terms 5 \
    --min-xy 1 \
    --rest-mode random \
    --coeff-dist normal \
    --estimator mean \
    --num-observables 10 \
    --n-samples 10 30 100 300 1000 3000 10000 \
    --num-state-repeats 3 \
    --num-observable-repeats 3 \
    --states haar_random ghz random_stabilizer \
    --seed 0 \
    --version n6_m5_sparse_operator