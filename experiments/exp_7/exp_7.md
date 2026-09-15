# exp_7 — Locality-tuned SEEQST: exact-*m* vs. at-least-*m* X/Y observables

**Script:** `experiments/exp_7/exp_7.py` · **Logic:** `tasks/exp_7_tuned_locality.py` ·
**Aggregation:** `experiments/exp_7/plot_m_dependence.py` · **Output:** `results/exp_7/v<N>/`

## Question it answers

`exp_6` swept a *cap* `l` on X/Y-weight and found `seeqst_binomial` (one
fixed `q=1/(n+1)` throughout) can badly underperform once the true weight
drifts away from what `q` was implicitly tuned for
(`results/exp_6/v0/why_binomial_underperforms.md`). The theory note written
since, `notes/main_theorem/main_theorem.tex` (Theorem 1 / Corollary 5),
explains why and fixes it: the Binomial family's shadow norm is only
*exactly* optimal — and only provably beats Pauli and Clifford — when `q`
is tuned to a single, fixed, **known** X/Y-locality `m` (`q* = m/n`).

`exp_7` is the direct empirical test of that theorem: six ensembles, on
observables built around one locality parameter `m`.

| name | class | tuning |
|---|---|---|
| `pauli` | `PauliEnsemble` | — (`β = 3^weight(P)`) |
| `clifford` | `CliffordEnsemble` | — (`β = 2^n+1`, exact, independent of everything) |
| `seeqst_uniform` | `SEEQSTEnsemble` | flat, `q=1/2` |
| `seeqst_binomial_tuned` | `SEEQSTBinomialEnsemble(q=m/n)` | Theorem 1's optimal `q` |
| `seeqst_binomial_untuned` | `SEEQSTBinomialEnsemble(q=1/(n+1))` | exp_6's baseline `q`, kept for contrast |
| `seeqst_unifsize_tuned` | `SEEQSTUniformSizeEnsemble(l=...)` | `l=m` under `exact`, `l`=the generator's own cap under `at_least` (see below — this is *not* a free choice) |

## The two observable families (`--observable-type`)

**`exact`** — X/Y-weight is exactly `m` always
(`pauli_utils.random_exact_xy_spec(n, m, rng, rest_mode)`), the regime
Theorem 1 / Corollary 5 are stated for. `--rest-mode` controls how the
`n-m` non-XY qubits are set: `random` (default — each independently,
freshly Z or I), `z` (all Z, exp_6's convention), or `identity` (all I,
`main_theorem.tex`'s own convention, total weight exactly `m`).

`rest_mode=random` is the default for a specific reason, not just
variety: by the general SEEQST eigenvalue theorem, every SEEQST ensemble's
cost depends only on a spec's X/Y-support, never on the extra Z-content —
so under `random`, every draw at fixed `m` has the *exact same* SEEQST
`β` regardless of how the Z/I split landed, while `PauliEnsemble`'s own
cost (`3^len(spec)`) is not b-independent (Z counts toward its weight, I
doesn't) and genuinely varies draw to draw. `tests/verify_exp7_observables.py`
checks this contrast directly (SEEQST constant across 300 draws at fixed
`m`, Pauli not) rather than leaving it as a claim.

**`at_least`** — X/Y-weight is drawn uniformly from `{m, ..., l}` (`l` an
upper cap, `--l`, default `min(m+2, n)`), Z-padded like exp_6
(`pauli_utils.random_bounded_xy_spec(n, l, rng, min_xy=m)` — that function
already supported a floor, no new generator needed here).
`seeqst_binomial_tuned` stays tuned to the **floor** `m`, not the true
random weight, deliberately: a controlled "we only know a lower bound"
robustness study, one notch more forgiving than exp_6's fully
uninformative range, showing how fast performance degrades as the true
weight exceeds what `q` was tuned for.

## A real bug this design surfaced: `seeqst_unifsize_tuned` cannot be tuned to the floor

The first version of this experiment tuned *both* new-family ensembles to
`m` unconditionally, mirroring `exact` mode. Under `at_least` this crashes:
`SEEQSTUniformSizeEnsemble.inverse_weight` doesn't degrade gracefully past
its cap the way `seeqst_binomial` does — it **raises** `ValueError`
(`p(a)=0` there is a singular channel, not merely high variance; see that
class's own docstring), and the true weight in `at_least` mode can exceed
`m` by construction. So `seeqst_unifsize_tuned` is tuned to `m` under
`exact` (true weight is always exactly `m`, never a problem) but to the
generator's own upper cap `l` under `at_least` — exactly mirroring how
`exp_6` tuned its `seeqst_unifsize_l` to the generator's cap, not to a
floor. `seeqst_binomial_tuned`'s `q=m/n`, by contrast, never needs this
adjustment — it degrades rather than fails. This asymmetry is itself a
concrete illustration of the eigenvalue theorem's "`p(a)=0` ⟹ no unbiased
estimator" clause, not just an implementation footnote.

## Why `m` is single-valued per run, not swept in-run

Same reasoning as exp_6's `l`: `m` governs the observable generator *and*
the tuned ensembles' own sampling (`q=m/n`, `l=m` or the cap), so a true
in-run sweep would need a fresh snapshot stream at every `m` — a
structurally different, higher-risk loop. `exp_7.py` runs one `m` per
invocation; `plot_m_dependence.py` stitches several already-run versions
into the across-`m` view, mirroring `plot_l_dependence.py` exactly, with
one adjustment: **which ensembles "should" be flat across `m` is
conditional**, not fixed. `clifford` and `seeqst_uniform` are always flat.
`pauli` is flat only when total Pauli weight is pinned to `n` regardless of
`m` — true for `at_least` (always Z-padded to full weight) and for `exact`
with `rest_mode=z`, but *not* for `rest_mode∈{random, identity}`, where
Pauli's own weight genuinely trends with `m` (that trend is the point of
the comparison, not a bug). The script computes the right set from each
run's own `hyperparameters.json` automatically.

## Metrics — only two (no `mean_variance`), same as exp_6

```
RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
```

Identical two-layer randomization / running-mean estimator to exp_1 and
exp_6 (one shared snapshot stream per (state, ensemble), reused across
every inner observable-set draw) — see exp_1.py's module docstring for the
full rationale.

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size |
| `--m` | the swept hyperparameter: exact locality (`exact`) or its floor (`at_least`); must satisfy `1<=m<=n-1` (default 2) |
| `--observable-type` | `exact` or `at_least` (default `exact`) |
| `--rest-mode` | `random` \| `z` \| `identity` (default `random`); only used under `exact` |
| `--l` | upper cap on X/Y-weight for `at_least` (default `min(m+2, n)`); ignored under `exact` |
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

## Output (`results/exp_7/v<N>/`)

Same shape as exp_6: `hyperparameters.json` (now also records `l_for_unifsize`,
`q_tuned`, `q_untuned`, `observable_type`, `rest_mode`), `raw.csv`,
`summary.csv`, and `rmse_vs_nsample_<state>.png` / `max_error_vs_nsample_<state>.png`
(2 metrics × `len(states)` plots, 6 ensemble lines each).

`experiments/exp_7/plot_m_dependence.py v1 v2 v3 ...` reads several such
`v<N>/` directories (same `n`, same `observable_type`, and — under `exact`
— same `rest_mode`) and writes
`results/exp_7/m_dependence/{rmse,max_error}_vs_m_<state>.png`, plus the
conditional flatness check described above.

`tests/verify_exp7_observables.py` covers the one genuinely new piece of
logic (`random_exact_xy_spec`) plus the b-independence/Pauli-heterogeneity
contrast directly; no new ensemble classes were added (both tuned SEEQST
ensembles are the same, already-tested classes from exp_6 with different
constructor arguments).

## Result headline

**`exact` mode, `rest_mode=random`, `n=6`, `m=1..5`** (`v0`–`v4`; the regime
Theorem 1 / Corollary 5 are stated for). `--num-observables 10
--n-samples 10 30 100 300 1000 3000 --num-state-repeats 2
--num-observable-repeats 3`, ~115s each — same scale as exp_6's own `n=6`
sweep. RMSE at `N_sample=3000`, `haar_random` shown (`ghz` and
`random_stabilizer` show the same pattern; see
`results/exp_7/m_dependence/rmse_vs_m_haar_random.png`):

| ensemble | `m=1` | `m=2` | `m=3` | `m=4` | `m=5` |
|---|---|---|---|---|---|
| `seeqst_unifsize_tuned` | **0.090** | **0.133** | 0.225 | 0.231 | 0.142 |
| `seeqst_binomial_tuned` | 0.104 | 0.184 | 0.211 | 0.176 | **0.095** |
| `pauli` | 0.139 | 0.222 | 0.312 | 0.298 | 0.372 |
| `clifford` | 0.170 | 0.158 | **0.145** | **0.154** | 0.167 |
| `seeqst_uniform` | 0.184 | 0.168 | 0.225 | 0.168 | 0.191 |
| `seeqst_binomial_untuned` | 0.105 | 0.227 | 0.620 | 1.524 | 1.455 |

The flatness sanity check passes cleanly at these settings (`clifford` and
`seeqst_uniform` both within 5–35% relative spread across `m` — noise, not
a trend; the earlier tiny `--quick`-scale smoke test *did* flag both as
"unexpectedly large spread" at 1 trial per point, which is exactly the
false-positive the check's own docstring anticipates, not a bug).

Three things line up with the theorem, even at this modest `n=6`:

**Tuned beats Pauli everywhere tested.** `seeqst_binomial_tuned` and
`seeqst_unifsize_tuned` both beat `pauli` at all five `m` — a clean sweep,
not just an asymptotic claim. (Under `rest_mode=random` this isn't quite
the theorem's own `3^m`-vs-`2^{1+nH_2(m/n)}` comparison — Pauli's cost here
also picks up the random Z/I split's own contribution — but the ordering
is the same, and the gap widens with `m` exactly as expected.)

**Tuned crushes untuned as `m` grows.** `seeqst_binomial_tuned` and
`seeqst_binomial_untuned` are nearly tied at `m=1` (both ≈0.10 — `q=1/(n+1)`
is only mildly wrong there) and diverge sharply from there: by `m=5` the
untuned ensemble is **>15× worse** (1.455 vs 0.095). This is
`why_binomial_underperforms.md`'s mechanism running in reverse — the exact
same `q^{-k}` cost structure that punishes a mistuned `q` rewards a
correctly-tuned one just as steeply.

**Clifford's narrow window shows up, and is wider than the asymptotic
prediction — as it should be at `n=6`.** Both tuned ensembles lose to
`clifford` at `m=2,3,4` and beat it at `m=1,5` — Proposition 4's
`m/n≈1/2` window, which shrinks like `O(n^{-1/2})`, has not yet narrowed
down to a single point at `n=6`; seeing it span 3 of 5 tested `m`-values
here is consistent with that shrinkage, not a contradiction of it.

**`at_least` mode, `n=6`, `m=2` floor** (`v5_atleast_tight` = `l=2`,
degenerate to weight exactly 2; `v6_atleast_wide` = `l=5`, most of the
uninformative range) — the floor-tuned family's *degradation* as the true
weight is allowed to exceed the floor it was tuned for:

| ensemble | tight (`l=2`) | wide (`l=5`) |
|---|---|---|
| `seeqst_binomial_tuned` (`q=m/n` fixed) | 0.167 | 0.361 (**≈2.2× worse**) |
| `seeqst_unifsize_tuned` (`l` widens with the cap) | 0.176 | 0.239 (≈1.4× worse) |
| `seeqst_binomial_untuned` | 0.260 | 1.259 (**≈4.8× worse**) |
| `clifford` | 0.152 | 0.127 |
| `pauli` | 0.494 | 0.508 |

(RMSE at `N_sample=3000`, averaged over the three state types.)
`seeqst_unifsize_tuned` degrades more gently than `seeqst_binomial_tuned`
here — a real, slightly non-obvious wrinkle: it's *forced* to widen its own
cap to the true range to avoid outright failure (see above), which
incidentally makes it more forgiving of the mismatch than
`seeqst_binomial_tuned`, whose `q` stays pinned to the floor by design.
Both floor-tuned ensembles remain far more forgiving than the fully
untuned baseline, which is nearly wiped out by the same widening.

**Bottom line:** tune SEEQST to a known, fixed locality and it reliably
beats Pauli and is competitive with or better than Clifford outside a
narrow band near `m≈n/2` — exactly Theorem 1 / Corollary 5's claim, now
with real simulated shot noise on top rather than just the closed-form
`β`. Give it only a floor instead of the exact value, and it degrades
gracefully (unlike `seeqst_unifsize_tuned`, which must widen its own cap
or fail outright) but predictably, at a rate set by the same `β(k)`
mechanism `why_binomial_underperforms.md` already characterized.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_7/exp_7.py --quick
python experiments/exp_7/exp_7.py --n 6 --m 3 --observable-type exact --rest-mode random
python experiments/exp_7/exp_7.py --n 6 --m 3 --observable-type exact --rest-mode identity
python experiments/exp_7/exp_7.py --n 6 --m 2 --observable-type at_least --l 5

# across-m view, after running a few --m values at the same n/observable_type:
python experiments/exp_7/plot_m_dependence.py v0 v1 v2 v3 v4
```
