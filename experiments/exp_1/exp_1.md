# exp_1 — Error scaling of classical-shadow estimators vs. number of shots

**Script:** `experiments/exp_1/exp_1.py` · **Logic:** `tasks/exp_1_error_scaling.py` · **Output:** `results/exp_1/v<N>/`

## Question it answers

For a fixed system size `n`, how does the error of the classical-shadow mean
estimator shrink as you take more shadow snapshots `N_sample`, and how do the
three measurement ensembles (Pauli, Clifford, SEEQST) compare on that curve?
Each ensemble draws its measurement circuit **uniformly at random** from its
own full set, oblivious to which observables will be asked about later — that
obliviousness is the defining feature of classical shadows (Huang, Kueng &
Preskill, arXiv:2002.08953): you measure first, decide what to estimate
afterward. exp_2/exp_3 ask the opposite question (circuits chosen with the
observables known in advance); this experiment is the randomized baseline
both are compared against.

## Design: two layers of randomization

- **Outer layer — random state.** For each requested state family (Haar-random
  pure states, GHZ states, random stabilizer states — `common/states.py`), draw
  a fresh random state of size `n`. Repeated `--num-state-repeats` times per
  family.
- **Inner layer — random observable set, same state.** For that state, draw a
  fresh set of `M = --num-observables` random Pauli observables (each qubit
  independently `I`/`X`/`Y`/`Z`, conditioned non-identity; `--k` restricts to
  exactly weight `k` instead of fully random weight). Repeated
  `--num-observable-repeats` times per state.

All metrics are finally averaged over both layers pooled together (every
outer state × every inner observable set).

## Measure once, mine many times

Right after each outer-layer state is drawn — **before** looking at any
observable — one stream of `max(n_samples)` classical-shadow snapshots per
ensemble is drawn (`ShadowEnsemble.sample_snapshots`). That same stream is
reused for every inner observable-set draw on that state
(`ShadowEnsemble.evaluate_snapshots`); the state is never re-measured just
because the observable set changed. This mirrors the actual physical protocol
and is the whole point of classical shadows: one round of measurement, mined
for arbitrarily many downstream observables. (An earlier version of this
script redrew snapshots per observable set, which conflated new measurement
noise with new observable choice — fixed.)

Within a single (state, ensemble) pair, the metrics at every `N_sample`
checkpoint are computed from a **running (cumulative) mean** over a prefix of
that one `max(n_samples)`-long stream, rather than by drawing a fresh
`N_sample`-shot batch per checkpoint. A prefix of an i.i.d. sequence is itself
a valid i.i.d. sample of that size, so this is statistically equivalent to the
"redraw every time" approach but needs only one pass of `max(n_samples)` shots
instead of one pass per checkpoint.

## Metrics

For each (state, ensemble), and each Pauli observable `O_i` in the drawn set,
let `o_i = tr(O_i ρ)` (exact, from the statevector) and let
`ô_i(N_sample)` be the running mean of the first `N_sample` single-shot
shadow estimates. Reported at every checkpoint, averaged over the `M`
observables:

- **RMSE**`(N_sample) = sqrt( mean_i (ô_i(N_sample) − o_i)² )`
- **MeanVariance**`(N_sample) = mean_i Var[ô_i^(1)]`, estimated as the sample
  variance of the first `N_sample` *raw, pre-averaging* single-shot values.
  This directly estimates the quantity bounded by Lemma S1 of the Huang SI —
  the single-shot shadow-norm-squared `‖O‖²_shadow ≥ Var[ô]`. It's a fixed
  number depending on the observable/ensemble, not on `N_sample`, so it should
  **flatten out** as `N_sample` grows rather than keep shrinking — that
  qualitative signature is what separates "genuinely lower variance" from
  "just a noisier estimate of variance." It connects to RMSE via
  `Var[ô(N_sample)] = Var[ô(1)]/N_sample`, i.e.
  `RMSE(N_sample) ≈ sqrt(MeanVariance / N_sample)`.
- **MaxError**`(N_sample) = max_i |ô_i(N_sample) − o_i|`

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size (qubits) |
| `--num-observables` | `M`, random Pauli observables per draw |
| `--n-samples` | list of `N_sample` checkpoints to evaluate at |
| `--num-state-repeats` | outer-layer repeat count |
| `--num-observable-repeats` | inner-layer repeat count |
| `--k` | restrict observables to exactly weight `k` (default: fully random weight) |
| `--states` | subset of `haar_random ghz random_stabilizer` to run |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing `v0, v1, ...`) |
| `--quick` | small/fast config for smoke-testing |
| `--quiet` | suppress per-state-draw progress lines |

## Output (`results/exp_1/v<N>/`)

- `hyperparameters.json` — every setting used for the run.
- `raw.csv` — one row per (state_repeat, obs_repeat, state_type, ensemble, n_sample).
- `summary.csv` — the above averaged over both randomization layers.
- 9 plots: `{rmse, mean_variance, max_error}_vs_nsample_{haar_random, ghz, random_stabilizer}.png`
  — log-log, one line per ensemble. RMSE/MaxError get a `1/√N_sample`
  reference line; MeanVariance gets a flat reference line at its
  most-converged (largest-`N_sample`) value.

`v0` in this repo is the first manually-saved run (single-layer repeats,
predates MeanVariance); `v1` is a `--quick` smoke test; `v2` is a real example
run of the current two-layer design.

## Result headline

At fixed locality `k=2`, Pauli-shadow variance is flat in `n` (~9, i.e.
`3^2`), while Clifford and SEEQST variance grow exponentially with `n`.
SEEQST tracks Clifford closely to somewhat worse for generic random Pauli
observables of fixed small weight, since its `2^(n+1)` inverse-map weight
applies to any Pauli with an X or Y factor — a `(1 − (1/3)^k)` fraction of
random weight-`k` draws — so at `k=2` that regime dominates almost as often as
it would for a fully global Clifford observable.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_1/exp_1.py --quick
python experiments/exp_1/exp_1.py --n 5 --num-observables 10 \
    --num-state-repeats 3 --num-observable-repeats 3 \
    --n-samples 10 30 100 300 1000 3000 10000
python experiments/exp_1/exp_1.py --n 6 --k 2 --num-observables 20 --states haar_random ghz
python experiments/exp_1/exp_1.py --n 5 --version my_run_name
```
