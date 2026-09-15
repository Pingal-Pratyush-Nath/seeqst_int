# exp_8 — Ceiling-tuned SEEQST: sparse-*m* X/Y observables (weight ≤ *m*)

**Script:** `experiments/exp_8/exp_8.py` · **Logic:** `tasks/exp_8_sparse_locality.py` ·
**Aggregation:** `experiments/exp_8/plot_m_dependence.py` · **Output:** `results/exp_8/v<N>/`

## Question it answers

`exp_7` tunes SEEQST to a *known lower value* of the X/Y-locality — exactly
`m` (`exact` mode) or at least `m` (`at_least` mode). This experiment tunes
it to a *known upper bound* instead: every observable has X/Y-weight drawn
uniformly from `{1, ..., m}` — never above `m` — and every other qubit is Z
(`pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)` — the *same*
generator `exp_6` already uses for its own untuned family; `exp_8` is
`exp_6`'s observable family with the SEEQST side finally tuned to the cap
instead of left at a fixed `q=1/(n+1)`).

This is the operational, list-of-separately-tracked-observables counterpart
of the **m-sparse operators** studied in
`notes/working/seeqst_sparse_tuning.tex` (`S_m = {s ⊆ [n] : |s| ≤ m}`, the
set of X/Y-support patterns allowed to appear). See below for exactly how
that note's result does — and does not — carry over here.

Six ensembles, exp_7's roster with `exact`/`at_least` replaced by this one
ceiling-tuned family:

| name | class | tuning |
|---|---|---|
| `pauli` | `PauliEnsemble` | — (`β = 3^weight(P)`, always `3^n` here: full weight) |
| `clifford` | `CliffordEnsemble` | — (`β = 2^n+1`, exact) |
| `seeqst_uniform` | `SEEQSTEnsemble` | flat, `q=1/2` |
| `seeqst_binomial_tuned` | `SEEQSTBinomialEnsemble(q=m/n)` | tuned to the **ceiling** `m` |
| `seeqst_binomial_untuned` | `SEEQSTBinomialEnsemble(q=1/(n+1))` | exp_6's baseline `q`, kept for contrast |
| `seeqst_unifsize_tuned` | `SEEQSTUniformSizeEnsemble(l=m)` | cap `= m`, unconditionally (see below) |

## The observable family

Full weight `n` always (every qubit X, Y, or Z, never identity); the
*number* of X/Y factors is drawn uniformly from `{1, ..., m}` and every
other qubit is fixed to Z — exactly `exp_6`'s generator
(`pauli_utils.random_bounded_xy_spec(n, l=m, rng, min_xy=1)`), reused
verbatim. No new generator was written for this experiment; the only new
code is tuning the SEEQST side to the same `m` that already governs the
generator, and choosing which comparison ensembles make sense once that's
done.

## Why `q = m/n` is right here *only for* `m ≤ n/2` — a two-regime result

`notes/working/seeqst_sparse_tuning.tex` proves that for a **single, combined**
`m`-sparse operator `A = Σ_{s∈S_m} c_s P_s` (i.e. estimating one linear
combination whose Pauli terms all happen to have X/Y-support in `S_m`), the
Binomial family's worst-case shadow-norm bound
`‖A‖²_sh ≤ 8 (Σ_{s∈S_m} 1/p(s)) ‖A‖²_∞` is minimized, to leading asymptotic
order, at `q* = m/n` rather than the source material's `q = 1/(n+1)` — via a
genuinely combinatorial argument (`Φ_m(q) = Σ_{w=0}^{m} C(n,w) q^{-w}(1-q)^{-(n-w)}`,
a dominant-term analysis as `n → ∞`).

`exp_8` tests a **different object**: a **list** of `M` *separately-tracked*
observables (the `max_i β(P_i)` sample-complexity criterion of Theorem 1 /
Huang–Kueng–Preskill Prop. S1), each individually having some X/Y-weight
`k ∈ {1, ..., m}`, not a single sum. For this object the right argument is
the **simple, single-string** one `exp_7` already uses — but worked all the
way through, it needs *both* directions of `q`, and that changes the answer
past `m = n/2`:

1. `ln β_q(k) = ln 2 - k ln q - (n-k) ln(1-q)` is **linear** in `k`, with
   slope `ln((1-q)/q)`. So on any fixed range `k ∈ {1,...,m}`, `β_q(·)` is
   monotonic in `k` — increasing if `q<1/2`, decreasing if `q>1/2`, flat
   (constant `= 2^{n+1}`) at exactly `q=1/2`. Its max over `k ∈ {1,...,m}`
   is therefore always at one of the two **endpoints**: `β_q(m)` if
   `q ≤ 1/2`, `β_q(1)` if `q ≥ 1/2` — never in the interior, and the two
   branches agree (both `= 2^{n+1}`) at `q=1/2`.
2. **Branch `q ≤ 1/2`** (minimize `β_q(m)`): this is exactly Theorem 1 of
   `notes/seeqst_sample_complexity.tex` / `main_theorem.tex` — unconstrained
   minimizer `q=m/n`. If `m ≤ n/2` this sits inside the branch's own domain
   — self-consistent — giving `2^{1+nH_2(m/n)}`. If `m > n/2`, the
   unconstrained minimizer falls **outside** `q≤1/2`, so the *constrained*
   minimum on this branch sits at the boundary `q=1/2`, value `2^{n+1}`.
3. **Branch `q ≥ 1/2`** (minimize `β_q(1)`): the same theorem at `k=1` —
   unconstrained minimizer `q=1/n`, which is `<1/2` for any `n>2` and so
   **always** outside this branch's domain. Its constrained minimum is
   therefore always at the boundary `q=1/2` too, value `2^{n+1}`.
4. Taking the better of the two branches:

   ```
   min_q max_{k=1}^{m} β_q(k)  =  2^{1+n H_2(m/n)}  at  q* = m/n     if  m ≤ n/2
                                =  2^{n+1}           at  q* = 1/2    if  m >  n/2
   ```

   i.e. `q* = m/n` is the correct, *unique* minimax tuning only in the
   genuinely "sparse" regime `m ≤ n/2` (`2^{1+nH_2(m/n)} ≤ 2^{n+1}` there,
   since `H_2 ≤ 1`). **Past that point, any `q ≠ 1/2` is actively worse in
   the worst case than not tuning at all** — the best a ceiling-tuned
   Binomial ensemble can do once the ceiling exceeds `n/2` is fall back to
   being exactly `seeqst_uniform`.

This two-branch structure — not just the `m ≤ n/2` half — was checked
numerically **before** finalizing this experiment
(`scipy.optimize.minimize_scalar` over `max_{k=1}^m β_q(k)`, swept over
`m = 1..n-1` at several `n`, compared against the piecewise closed form
above: exact machine-precision agreement in *both* regimes, boundary at
`m=n/2` included). An earlier draft of this reasoning claimed `q*=m/n`
unconditionally, from only the `q<1/2` half of point 1 and without checking
`m>n/2` numerically — that was wrong, and was caught precisely because
`tests/verify_exp8_observables.py`'s check 4 sweeps the *full* range
`1 ≤ m ≤ n-1` rather than a few small-`m` examples.

Because of this, `exp_8`'s `seeqst_binomial_tuned` still uses the literal
`q=m/n` construction **unconditionally** (matching `exp_7`'s naming and the
"naive `m/n` tuning" this experiment is precisely built to stress-test), and
`exp_8.py` prints a warning whenever `--m > n/2` explaining that the
configured tuning is no longer minimax-optimal there — rather than silently
substituting `q=1/2`, which would hide the very breakdown this experiment
exists to demonstrate.

**The two arguments (this one and `seeqst_sparse_tuning.tex`'s) reach the
same `q* = m/n` in the regime where both are meaningful** (`m ≤ n/2`, also
where "`m`-sparse" is a sensible description of the family in the first
place), **but for structurally different reasons and on different objects**
(list-max vs. summed-operator bound). `exp_8` is a direct empirical test of
the argument above — does ceiling-tuned SEEQST-binomial beat untuned and
beat Pauli on a weight-≤*m* observable *list*, and does the advantage
vanish past `m=n/2`? — not of the harder sparse-tuning note's
summed-operator claim. Anyone citing this experiment's results as support
for `seeqst_sparse_tuning.tex`'s theorem (or vice versa) would be
conflating two different objects that merely happen to share an optimal
`q` in one regime.

## `seeqst_unifsize_tuned` needs no special-casing here — unlike exp_7's `at_least`

`exp_7`'s `at_least` mode had to tune `seeqst_unifsize_tuned` to the
generator's own upper cap `l`, *not* the floor `m`, because
`SEEQSTUniformSizeEnsemble.inverse_weight` raises `ValueError` (not merely
degrades) once the true X/Y-weight exceeds its own cap, and `at_least`'s
true weight can exceed its floor `m` by construction.

`exp_8`'s generator can **never** produce a weight above `m` in the first
place (`random_bounded_xy_spec(n, l=m, rng, min_xy=1)` draws
`k ~ Uniform{1,...,m}` by definition), so `seeqst_unifsize_tuned = l=m`
is safe unconditionally — no widening, no `ValueError`, ever.
`tests/verify_exp8_observables.py`'s check 3 confirms this directly (30
draws × every `(n, m)` combination tested, zero exceptions).

## Metrics — only two (no `mean_variance`), same as exp_6/exp_7

```
RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
```

Identical two-layer randomization / running-mean estimator to exp_1,
exp_6, exp_7 (one shared snapshot stream per `(state, ensemble)`, reused
across every inner observable-set draw) — see `exp_1.py`'s module
docstring for the full rationale.

## What you'd expect to see (predictions, to check the run against)

- These predictions are for the `m ≤ n/2` regime, where `q=m/n` is actually
  minimax-optimal (see above). **Past `m=n/2`** (`exp_8.py` prints a
  warning there), expect `seeqst_binomial_tuned` to *stop improving* and
  plausibly get *worse* relative to `seeqst_uniform` as `m` grows further —
  it is running an increasingly mistuned `q=m/n` against a family whose true
  minimax optimum sits at the fixed point `q=1/2` (`seeqst_uniform` itself)
  once the ceiling covers more than half the qubits. A metric-vs-`m` sweep
  through `m=n/2` (`plot_m_dependence.py`) should show `seeqst_binomial_tuned`
  approaching, and then crossing back away from, `seeqst_uniform`'s
  performance right around that point — the empirical signature of the
  theorem's regime change.
- `seeqst_binomial_tuned` and `seeqst_unifsize_tuned` should both beat
  `seeqst_binomial_untuned`, with the gap **widening as `m` grows**, *up to*
  `m≈n/2` — the same `q^{-k}` mechanism `exp_7` already demonstrated for the
  floor-tuned case, now for a ceiling. At `m=1` the two binomial ensembles
  should be close (`q=m/n=1/(n+1)`-ish for small `m`, so
  `q_tuned ≈ q_untuned`); by `m` a sizeable fraction of `n` (but still
  `≤n/2`) the untuned ensemble should be dramatically worse, exactly as it
  was in `exp_7`'s `m=5` row (`>15×` worse).
- `seeqst_binomial_tuned` should beat `pauli` in this observable family for
  every tested `m` — Pauli's own weight is pinned to `n` here (like
  `exp_6`'s family and `exp_7`'s `at_least` family), so the relevant
  comparison is `2^{1+nH_2(m/n)}` (an upper bound on `max_k β_q*(k)` here,
  since the true worst case is `β_{q*}(m)` itself, which by construction
  equals `2^{1+nH_2(m/n)}` exactly) against Pauli's `3^n` — a comparison
  that only gets more favorable to SEEQST as `n` grows, by
  `notes/seeqst_sample_complexity.tex`'s Proposition 7.
- Because the true weight is `≤ m`, not `= m`, `seeqst_binomial_tuned`
  should sit *between* `exp_7`'s `exact`-mode numbers at the same `m` (that
  worst case *is* realized here, at `k=m`) and something better on
  *average* — most draws have `k < m` and get a strictly smaller `β`, so
  `RMSE` (an average-case metric) should come in noticeably below
  `MaxError`'s implied worst case, more so than in `exp_7`'s `exact` mode
  where every draw sits at exactly `k=m`.
- `pauli`, `clifford`, and `seeqst_uniform` should all be **flat across
  `m`** (unlike `exp_7`'s `exact`+`rest_mode∈{random,identity}` families,
  where Pauli's own weight trends with `m` by design) — exactly `exp_6`'s
  flatness pattern, since Pauli weight is pinned to `n` regardless of `m`
  here too. `plot_m_dependence.py`'s flatness check covers this.

(No results table is included here yet — fill this section in with actual
`results/exp_8/v<N>/` numbers once a real sweep has been run, the way
`exp_6.md` / `exp_7.md` do.)

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size |
| `--m` | the swept hyperparameter: ceiling on X/Y-weight (weight ~ `Uniform{1,...,m}`); must satisfy `1<=m<=n-1` (default 2). `q=m/n` is minimax-optimal only for `m<=n/2`; the script warns when `m>n/2` |
| `--q-binomial-untuned` | `seeqst_binomial_untuned`'s `q` (default `1/(n+1)`, exp_6's own default) |
| `--num-observables` | `M`, number of random observables per draw |
| `--n-samples` | list of `N_sample` checkpoints |
| `--num-state-repeats` | outer layer repeats |
| `--num-observable-repeats` | inner layer repeats |
| `--states` | subset of `haar_random ghz random_stabilizer` |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing) |
| `--quick` | small/fast config for smoke-testing (`n=4, m=2`) |
| `--quiet` | suppress per-state-draw progress lines |

## Output (`results/exp_8/v<N>/`)

Same shape as exp_6/exp_7: `hyperparameters.json` (records `q_tuned`,
`q_tuned_is_minimax_optimal` (`= 2*m<=n`), `q_untuned`, `m`, `n`, ...),
`raw.csv`, `summary.csv`, and
`rmse_vs_nsample_<state>.png` / `max_error_vs_nsample_<state>.png` (2
metrics × `len(states)` plots, 6 ensemble lines each).

`experiments/exp_8/plot_m_dependence.py v1 v2 v3 ...` reads several such
`v<N>/` directories (same `n`) and writes
`results/exp_8/m_dependence/{rmse,max_error}_vs_m_<state>.png`, plus the
flatness check for `{pauli, clifford, seeqst_uniform}` described above.

`tests/verify_exp8_observables.py` covers: the observable family's weight
bound and uniformity, `seeqst_unifsize_tuned`'s unconditional safety on
this family, the numerical `q*=m/n` monotonicity argument (independent of
the closed-form formula), and agreement between `SEEQSTBinomialEnsemble`'s
`inverse_weight` and the bare `β_q(k)` formula. No new ensemble classes
were added — both tuned SEEQST ensembles are the same, already-tested
classes from exp_6/exp_7 with different constructor arguments, and the
observable generator is exp_6's own `random_bounded_xy_spec`, reused
verbatim.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_8/exp_8.py --quick
python experiments/exp_8/exp_8.py --n 6 --m 2
python experiments/exp_8/exp_8.py --n 6 --m 3   # m=n/2 boundary
python experiments/exp_8/exp_8.py --n 6 --m 5   # m>n/2: prints the minimax-optimality warning

# across-m view, after running a few --m values at the same n:
python experiments/exp_8/plot_m_dependence.py v0 v1 v2 v3 v4

# unit tests for the observable family / tuning argument:
python tests/verify_exp8_observables.py
```
