# exp_6 — Bounded-X/Y-weight observables: Pauli vs. Clifford vs. 3 SEEQST variants

**Script:** `experiments/exp_6/exp_6.py` · **Logic:** `tasks/exp_6_bounded_xy_weight.py` ·
**Aggregation:** `experiments/exp_6/plot_l_dependence.py` · **Output:** `results/exp_6/v<N>/`

## Question it answers

The theory note `QIP_notes/SEEQST_shadows_threshold.tex` derives a phase
diagram for when SEEQST beats Pauli and Clifford, driven by the weight
fraction `f = k/n` of an observable's X/Y content, and built on two tunable
SEEQST sub-families: Binomial(n, q) and uniform-subset-size. This experiment
is the direct empirical counterpart: fix an observable family whose only
free parameter is exactly that X/Y-content knob, and watch five ensembles'
errors move as the knob turns.

Five ensembles are compared:

| name | class | `β(P)` under this experiment's observables |
|---|---|---|
| `pauli` | `PauliEnsemble` | `3^n` (weight is always `n`, see below) |
| `clifford` | `CliffordEnsemble` | `2^n+1` (exact; independent of weight) |
| `seeqst_uniform` | `SEEQSTEnsemble` | `2^(n+1)` (flat/uniform SEEQST; independent of weight) |
| `seeqst_binomial` | `SEEQSTBinomialEnsemble(q=1/(n+1))` | `2·q^{-k}(1-q)^{-(n-k)}`, `k`=X/Y-weight |
| `seeqst_unifsize_l` | `SEEQSTUniformSizeEnsemble(l)` | `2(l+1)\binom{n}{k}`, `k≤l` |

**Observables:** Hermitian Pauli strings with **full weight `n`** — every
qubit is X, Y, or Z, never identity — where the *number* of X/Y factors is
drawn uniformly from `{1, ..., l}` and every other qubit is fixed to Z
(`pauli_utils.random_bounded_xy_spec`; `l` is a hyperparameter, `--l`).
Fixing full weight (rather than allowing arbitrary identity-padding) keeps
every observable inside the support of every ensemble under test — in
particular `seeqst_unifsize_l`, whose closed form only covers X/Y-weight
`k ≤ l` and has no clean closed form for the pure-Z sector (see
`ensembles/seeqst_unifsize_ensemble.py`'s docstring) — and it is also what
makes `pauli`, `clifford`, and `seeqst_uniform`'s `β` all come out
independent of `l`: their weight-dependence (where they have any) is on the
*total* Pauli weight, which never changes here.

## Why `l` is single-valued per run, not swept in-run

Unlike exp_1's optional `--k` (which only restricts the observable
generator), `l` here governs **two** things at once: the observable
generator *and* `seeqst_unifsize_l`'s own sampling distribution. A true
in-run sweep would need to redraw a fresh snapshot stream (not just a fresh
observable set) at every `l`, which is a structurally different,
higher-risk loop than exp_1's — so `exp_6.py` mirrors exp_1's `--k`
structure exactly (one value of `l` per invocation) and a separate,
read-only aggregation script (`plot_l_dependence.py`) stitches several runs
together into the across-`l` view instead. See exp_6.py's module docstring
for the full reasoning.

## Metrics — only two (no mean_variance)

Per the user's request, only:

```
RMSE(N_sample)     = sqrt( mean_i (o_hat_i(N_sample) - o_i)^2 )
MaxError(N_sample) = max_i |o_hat_i(N_sample) - o_i|
```

exp_1's `MeanVariance` metric is dropped — not computed at all in
`run_exp6`, not just omitted from the output — since it isn't needed here.
Otherwise the estimator is identical to exp_1's: one shared stream of
`max(n_samples)` snapshots per (state, ensemble), reused via the
running-mean checkpoint construction across every inner observable-set
draw. See exp_1.py's module docstring for the full two-layer-randomization
rationale, reused unchanged.

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size (number of qubits) |
| `--l` | max number of X/Y factors per observable, and `seeqst_unifsize_l`'s own parameter; must satisfy `1 <= l <= n` (default 2) |
| `--q-binomial` | per-qubit Bernoulli inclusion probability for `seeqst_binomial` (default `1/(n+1)`) |
| `--num-observables` | `M`, number of random observables per draw |
| `--n-samples` | list of `N_sample` checkpoints to evaluate at |
| `--num-state-repeats` | outer layer: how many random states to draw |
| `--num-observable-repeats` | inner layer: how many random observable sets per state |
| `--states` | subset of `haar_random ghz random_stabilizer` to run |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing `v0, v1, ...`) |
| `--quick` | small/fast config for smoke-testing (`n=4, l=2`) |
| `--quiet` | suppress per-state-draw progress lines |

## Output (`results/exp_6/v<N>/`)

- `hyperparameters.json` — every setting used, including the actual
  `q_binomial` value (resolved from `1/(n+1)` if not passed explicitly).
- `raw.csv` — one row per `(state_repeat, obs_repeat, state_type, ensemble, n_sample)`.
- `summary.csv` — the above averaged over both randomization layers.
- `rmse_vs_nsample_<state>.png`, `max_error_vs_nsample_<state>.png` — one
  pair per state type, log-log, 5 ensemble lines, `1/sqrt(N_sample)`
  reference line. 2 metrics × `len(states)` plots (6 for the default three
  state types).

`experiments/exp_6/plot_l_dependence.py v1 v2 v3 ...` reads several such
`v<N>/` directories (same `n`, varying `l`) and writes
`results/exp_6/l_dependence/{rmse,max_error}_vs_l_<state>.png` — one line
per ensemble, evaluated at the largest `N_sample` checkpoint shared by all
the given runs — plus a printed flatness sanity check (see that script's
docstring): `pauli`, `clifford`, and `seeqst_uniform` should come out flat
across `l` under this observable family, so a large systematic trend there
would flag a bug rather than real physics.

`v0` is a `--quick` smoke test (`n=4, l=2`). `v1`–`v4` are real example runs
at `n=6`, `l ∈ {1, 2, 4, 6}`
(`--num-observables 10 --n-samples 10 30 100 300 1000 3000
--num-state-repeats 2 --num-observable-repeats 3`, ~100s each).

## Result headline

At `n=6`, `N_sample=3000`, RMSE averaged over both randomization layers
(all three state types show the same pattern; `haar_random` shown, see
`results/exp_6/l_dependence/rmse_vs_l_haar_random.png` for the plot):

| ensemble | `l=1` | `l=2` | `l=4` | `l=6` |
|---|---|---|---|---|
| `seeqst_unifsize_l` | **0.077** | **0.126** | 0.200 | 0.221 |
| `seeqst_binomial` | 0.112 | 0.199 | 0.526 | 1.164 |
| `clifford` | 0.174 | 0.147 | 0.153 | **0.140** |
| `seeqst_uniform` | 0.166 | 0.191 | 0.189 | 0.195 |
| `pauli` | 0.527 | 0.456 | 0.430 | 0.467 |

This is the theory note's phase diagram showing up directly in measured
error, not just in the closed-form `β`: `pauli`, `clifford`, and
`seeqst_uniform` are flat across `l` (as they must be — none of their `β`
depends on `l` under this full-weight observable family; the small residual
wiggle is Monte-Carlo noise from the modest repeat counts above, confirmed
by `plot_l_dependence.py`'s flatness check, which passes cleanly except one
borderline case — `pauli`'s `max_error` on `ghz` — that is itself
non-monotonic in `l` with a per-point std comparable to the spread, i.e.
noise, not a trend), while the two SEEQST-tuned ensembles are *not*: both
start out clearly best at `l=1` (`seeqst_unifsize_l` ≈2.3× better than
Clifford, `seeqst_binomial` ≈1.6× better), then degrade monotonically as
`l` grows. `seeqst_binomial`'s degradation is the steeper of the two —
consistent with its `q^{-k}` closed form, geometric in `k` — and by `l=6`
it has become the *worst* of all five ensembles (worse even than `pauli`),
while `seeqst_unifsize_l` degrades more gently and remains competitive with
Clifford out to `l≈4`. The `seeqst_unifsize_l`/`clifford` crossover falls
between `l=2` and `l=4` in every state type tested — i.e. a weight fraction
`f=l/n` between `1/3` and `2/3` — bracketing the theory note's asymptotic
prediction `f*≈0.609` about as closely as an `n=6` system with 4 sampled
`l`-points can be expected to.

**Bottom line:** there is no single best ensemble here — which one wins is
governed entirely by where the observable's X/Y-content sits relative to
each SEEQST variant's tuning, exactly the tunable-threshold story the
theory note makes.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_6/exp_6.py --quick
python experiments/exp_6/exp_6.py --n 6 --l 2
python experiments/exp_6/exp_6.py --n 6 --l 4 --q-binomial 0.3 --states haar_random ghz
python experiments/exp_6/exp_6.py --n 8 --l 3 --num-observables 20 \
    --n-samples 10 30 100 300 1000 3000 10000   # bigger, several minutes

# across-l view, after running a few --l values at the same n:
python experiments/exp_6/plot_l_dependence.py v1 v2 v3 v4
```
