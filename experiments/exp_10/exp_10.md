# exp_10 — exp_9 with the observable family swapped: sparse-*m* instead of exact-*m*

**Script:** `experiments/exp_10/exp_10.py` · **Logic:** `tasks/exp_10_sparse_locality_mom.py` ·
**Output:** `results/exp_10/v<N>/`

## What this is

`exp_10` is `exp_9` (see `exp_9.md`) with exactly one change: which
observable family gets drawn.

| | exp_9 | exp_10 |
|---|---|---|
| observable family | **exact-*m*** — X/Y-weight always exactly `m` | **sparse-*m*** — X/Y-weight drawn uniformly from `{1,...,m}`, never above `m` |
| generator | `pauli_utils.random_exact_xy_spec(n, m, rng, rest_mode)` | `pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)` (exp_8's generator, reused verbatim) |
| non-XY qubits | `--rest-mode`: `"random"` (iid Z/I) or `"z"` (all Z) | always Z (the generator has no `rest_mode` knob) |
| ensembles | `pauli`, `clifford`, `seeqst_uniform`, `seeqst_binomial_tuned(q=m/n)` | **same four**, unchanged |
| estimator | `--estimator {mean, median_of_means}` | **same option**, unchanged |
| metrics | RMSE, MaxError | **same two**, unchanged |

Everything not in that first row — the two-layer randomization design, the
snapshot-stream reuse per `(state, ensemble)`, the selectable point
estimator and its `K`-clamping, the four-ensemble roster and its `q=m/n`
tuning rule, and the RMSE/MaxError metrics — is carried over from `exp_9`
unchanged. See `exp_9.md` / `exp_7.md` for the full background on all of
that (`notes/main_theorem/main_theorem.tex`, Theorem 1 / Corollary 5).

## Why "sparse-*m*" and not "exact-*m*"

`exp_9`'s single-Pauli-string observables sit exactly at the weight the
`q=m/n` tuning is derived for. `exp_10` instead draws the operational,
list-of-separately-tracked-observables analogue of the **m-sparse
operators** studied in `notes/working/seeqst_sparse_tuning.tex`
(`S_m = {s ⊆ [n] : |s| ≤ m}`, the set of X/Y-support patterns allowed to
appear) — this is the exact same family and generator `exp_8` already
introduced (`exp_8.md`), just run here through `exp_9`'s harness (four
ensembles instead of six, plus the selectable estimator) instead of
`exp_7`'s.

Because the true weight is `≤ m`, not `= m`, the `q=m/n` tuning used by
`seeqst_binomial_tuned` is **only minimax-optimal for `m ≤ n/2`** —
`exp_8.md`'s "Why `q=m/n` is right here *only for* `m ≤ n/2`" section
derives this in full (a two-branch argument: `β_q(·)` is monotonic in
weight `k` on any fixed range, so its max over `k ∈ {1,...,m}` sits at an
endpoint, and past `m=n/2` the constrained minimax `q` collapses to `1/2`,
i.e. `seeqst_uniform` itself). That derivation carries over to `exp_10`
unchanged, since the observable generator and the `q=m/n` rule are
identical to `exp_8`'s. Unlike `exp_8.py`, `exp_10.py` does **not** print a
runtime warning when `--m > n/2` and does **not** add a `seeqst_uniform`-at-
`q=1/2` fallback ensemble — it stays a minimal "`exp_9` with one swapped
observable family" diff, matching `exp_9`'s exact four-ensemble roster and
CLI shape. The caveat is real regardless of whether the script warns about
it: `hyperparameters.json` still records `q_tuned_is_minimax_optimal`
(`= 2*m <= n`) so a `--m > n/2` run is at least flagged in its own output,
the same field `exp_8`'s output uses.

Because the true weight is `≤ m` rather than pinned at `m`, expect
`seeqst_binomial_tuned`'s worst case here (realized at `k=m`, i.e. matching
`exp_9`'s `exact`-mode number at the same `m`) to sit *above* its
average-case behavior — most draws have `k < m` and so a strictly smaller
`β`, more so than in `exp_9` where every draw sits at exactly `k=m`. RMSE
(an average-case metric) should therefore come in noticeably below what a
naive worst-case read of `MaxError` implies, the same pattern `exp_8.md`
documents for its own six-ensemble run.

## Validation

Qiskit is not installable in either this session's cloud workspace or the
local sandbox `device_bash` runs in (both attempted; both proxies reject
`qiskit==2.4.1`, the project's pinned version, with no matching
distribution), so an end-to-end `python exp_10.py --quick` run could not be
executed as part of writing this experiment — unlike `exp_9.md`'s own
validation section, which reports one. What *was* checked directly:

- `python3 -m py_compile experiments/exp_10/exp_10.py
  tasks/exp_10_sparse_locality_mom.py` — both files parse cleanly.
- The one genuinely new piece of logic — `_draw_observables`'s call into
  `pauli_utils.random_bounded_xy_spec` — has no qiskit dependency
  (`common/pauli_utils.py` imports only `numpy`), so it was exercised
  directly: 20000 draws at `n=6, m=4` gave X/Y-weight counts of
  `{1: 5017, 2: 5006, 3: 5013, 4: 4964}` (uniform over `{1,...,4}` as
  expected, mean `2.496` vs. the predicted `2.5`), every draw had full
  weight `n` (every qubit `X`, `Y`, or `Z`, never identity), and every
  non-XY letter was `Z` — matching `exp_8`'s generator semantics exactly.
- Every other function in `tasks/exp_10_sparse_locality_mom.py`
  (`_checkpoint_estimates`, `run_exp10`'s two-layer loop, `summarize`) is
  byte-for-byte `exp_9`'s already-validated code (see `exp_9.md`'s own
  Validation section) with only the observable-generator call site and the
  dropped `rest_mode` field changed; `experiments/exp_10/exp_10.py` is
  likewise `exp_9.py` with `--rest-mode` removed and the docstring/imports
  updated for the new module names.
- **Not yet run**: the actual `--quick` smoke test, a `--n 6 --m 5`-scale
  comparison against `exp_9`'s numbers at the same `m`, and confirmation
  that `seeqst_binomial_tuned`'s advantage narrows or reverses somewhere
  past `m=n/2` (the pattern `exp_8.md` predicts and observed for its own
  six-ensemble run). Whoever next has a Python environment with
  `qiskit==2.4.1` available should run these before treating this
  experiment's *results* (as opposed to its code) as validated — the code
  path itself was checked as thoroughly as this environment allows.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_10/exp_10.py --quick
python experiments/exp_10/exp_10.py --n 6 --m 5
python experiments/exp_10/exp_10.py --n 6 --m 5 --estimator median_of_means
python experiments/exp_10/exp_10.py --n 6 --m 5 --estimator median_of_means --mom-num-groups 5
python experiments/exp_10/exp_10.py --n 7 --m 3 --version my_run_name
```
