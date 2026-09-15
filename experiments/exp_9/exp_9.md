# exp_9 — Core comparison: Pauli, Clifford, SEEQST-uniform, SEEQST-binomial-tuned, mean vs. median-of-means

**Script:** `experiments/exp_9/exp_9.py` · **Logic:** `tasks/exp_9_tuned_locality_mom.py` ·
**Output:** `results/exp_9/v<N>/`

## What this is

`exp_9` is `exp_7` trimmed to exactly what the project is comparing right now,
plus one new capability. It reuses `exp_7`'s "exact-*m*" observable family
and two-layer randomization design unchanged (see `tasks/exp_7_tuned_locality.py`
and `notes/main_theorem/main_theorem.tex`, Theorem 1 / Corollary 5, for the
full theoretical background) — nothing here is new physics, it's a narrower,
cleaner comparison plus a new estimator option.

## What's dropped from exp_7

| exp_7 had | exp_9 keeps |
|---|---|
| `--observable-type` (`exact` / `at_least`) | only `exact` (X/Y-weight is always exactly `m`) |
| `--rest-mode` (`random` / `z` / `identity`) | only `random` and `z` |
| 6 ensembles (`pauli`, `clifford`, `seeqst_uniform`, `seeqst_binomial_tuned`, `seeqst_binomial_untuned`, `seeqst_unifsize_tuned`) | 4 ensembles: `pauli`, `clifford`, `seeqst_uniform`, `seeqst_binomial_tuned` |
| a `∝ 1/√N_sample` dashed reference line on every plot | removed — it's only a claim about the *mean* estimator's own convergence rate, and would be misleading plotted against a median-of-means run |

`seeqst_binomial_untuned` and `seeqst_unifsize_tuned` were both apparatus
specific to exp_7's `at_least`/floor-robustness study, which exp_9 doesn't
run, so they're gone rather than just unused. (`seeqst_binomial_untuned` is
also the ensemble that produced the zero-inflated, heavy-tailed blow-up
documented for `n=6, m=5` in `results/exp_7/n6_m5_exact_random` /
`n6_m5_exact_zex` — part of the motivation for the new estimator option
below, even though exp_9 itself no longer runs that ensemble.)

## New: `--estimator {mean, median_of_means}`

Every experiment before this one always turned a stream of shots into a
point estimate with a plain running mean. `exp_9` adds HKP's own estimator
as a second option (Huang, Kueng & Preskill, informal Thm. 1 / SI Thm. S1 —
see `notes/working/seeqst_sample_complexity.tex`):

- **`mean`** (default) — the running empirical mean over the first
  `N_sample` shots, unchanged from every prior experiment.
- **`median_of_means`** — split the first `N_sample` shots into `K` equal
  groups, average within each group, then take the per-observable *median*
  of those `K` group-means.

`K` (`--mom-num-groups`) is fixed for the whole run, exactly as in HKP's
theorem (it depends on `M` and `δ`, not on `N_sample` or `ε`), and defaults
to HKP's own formula `K = 2⌈ln(2M/δ)⌉` (`--mom-delta`, default `0.1`) when
not given explicitly. If a requested `--n-samples` checkpoint is smaller
than `K`, that checkpoint clamps `K` down to its own `N_sample` (groups of
size 1 — a plain per-shot median) rather than crashing; a note is printed
when this happens. See `tasks/exp_9_tuned_locality_mom.py::_checkpoint_estimates`
for the implementation — the `mean` path keeps exp_7's `O(max_n_sample)`
single-cumsum trick; `median_of_means` reshapes the first `N_sample` shots
fresh per checkpoint (cheap next to the snapshot sampling itself, since `K`
is fixed but the group size grows with `N_sample`).

Both estimators consume the exact same underlying stream of
`max(n_samples)` classical-shadow snapshots per `(state, ensemble)` — only
the aggregation step differs.

## Validation

Both new files were run end-to-end (not just syntax-checked) against a
local copy of `common/`, `ensembles/`, and `codes/SEEQST/tools/` with the
project's pinned dependency versions (`qiskit==2.4.1`, `numpy==2.4.4`,
`pandas==3.0.3`, `matplotlib==3.10.9`):

- `--quick` smoke tests for both `--rest-mode` values and both `--estimator`
  values completed without error and produced sane `raw.csv` / `summary.csv`
  / plots.
- The `K`-clamping path was exercised directly (`--mom-num-groups 20` against
  a `--n-samples` list starting at `5`) — printed the expected note and did
  not crash.
- `--estimator mean` vs. `--estimator median_of_means` on the same
  `(n, m, rest_mode)` produce visibly different `summary.csv` numbers, and
  in the small-`N_sample` / large-`K` regime `median_of_means` shows the
  expected degenerate "flat until batches grow past size 1" pattern.
- A realistic-scale run at `n=6, m=5, rest_mode=random` (the locality this
  project has most recently been probing) reproduces the expected ordering
  `seeqst_binomial_tuned < clifford ≈ seeqst_uniform < pauli`, matching
  Theorem 1 — with only the 4 requested ensembles on the legend and no
  reference line.

## Usage

```bash
python exp_9.py --quick
python exp_9.py --n 6 --m 5 --rest-mode random
python exp_9.py --n 6 --m 5 --rest-mode z --estimator median_of_means
python exp_9.py --n 6 --m 5 --estimator median_of_means --mom-num-groups 5


python experiments/exp_9/exp_9.py \
    --n 6 \
    --m 5 \
    --rest-mode random \
    --num-observables 50 \
    --n-samples 10 50 100 200 400 600 800 1000 \
    --num-state-repeats 10 \
    --num-observable-repeats 50 \
    --states haar_random \
    --version n6_m5_exact_random


python experiments/exp_9/exp_9.py \
    --n 6 \
    --m 5 \
    --rest-mode random \
    --num-observables 50 \
    --n-samples 100 500 1000 2000 3000 4000 5000 6000 \
    --num-state-repeats 10 \
    --num-observable-repeats 50 \
    --states haar_random \
```
