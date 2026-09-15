# exp_2 — Derandomized SEEQST: error scaling under an observable-adapted measurement schedule

**Script:** `experiments/exp_2/exp_2.py` · **Logic:** `tasks/exp_2_derandomized_scaling.py` · **Derandomization rule:** `experiments/Archived/exp_3/exp_3.py` (lives under `Archived/` for folder organization only — it's a live dependency, not deprecated) · **Output:** `results/exp_2/v<N>/`

## Question it answers

exp_1 measures each state with a uniformly-random draw from the full SEEQST
ensemble, oblivious to which observables you'll eventually ask about. exp_2
asks the opposite question: **if you know your `M` observables in advance**,
how much better can you do by choosing your measurement circuits specifically
for them? This is the SEEQST analogue of Huang-Kueng-Preskill's derandomized
Pauli shadows.

## Protocol

For a chosen system size `n` and a drawn set of `M` random Pauli observables
`O_1, ..., O_M` (same generating distribution as exp_1's inner layer):

1. **Derandomize** each observable: `exp_3.circuit_for_pauli` (via
   `select_subset_and_branch` + `circuit_for_subset_branch`) gives the exact
   SEEQST circuit that measures that specific observable with certainty — see
   `experiments/Archived/exp_3/exp_3.md` for the derivation. Distinct observables can map to the *same*
   circuit (e.g. `Z_1` and `Z_1 Z_2` are both measured by "just measure
   everything in the computational basis"), and a single shot with that
   circuit gives a deterministic ±1 reading for *every* observable that maps
   to it, simultaneously. This grouping is where the entire efficiency gain
   over exp_1 comes from.
2. **Allocate** the shot budget across the resulting distinct circuits,
   proportional to observable demand: if circuit `S_j` is the exact match for
   `count_j` of the `M` observables, each individual shot's circuit is drawn
   independently from the categorical distribution `P(S_j) = count_j / M`.
   (E.g. if `S_1` is needed by 2 of 6 observables and `S_3` by 4 of 6, each
   shot uses `S_1` with probability 2/6 and `S_3` with probability 4/6.) This
   is a simple, greedy, demand-proportional allocation — not a
   variance-optimized one. See
   `experiments/Archived/exp_3/exp_3_shot_scheduling_suggestions.md` for
   concrete ways to improve on it.
3. **Estimate**: every shot updates the running history of every observable
   that shares its circuit. At checkpoint `N_sample` (total shots drawn from
   the shared budget), observable `O_i`'s estimate is the mean of however many
   of its *own* matching shots have occurred so far — a random, and at small
   `N_sample` often zero, count (see Coverage below).

Implementation detail: `run_exp2` groups the `M` observables by their
`(subset, branch)` derandomization key (`bisect_right` over each observable's
sorted list of matching-shot indices gives its match count at any checkpoint
in O(log) time), draws `max(n_samples)` shots once from the categorical
distribution over distinct circuits, and evaluates every checkpoint from that
single stream — analogous to exp_1's "one pass, many checkpoints" trick.

## Two differences from exp_1's protocol

1. **No shared snapshot stream across observable-set draws.** exp_1 measures a
   state once per outer-layer draw and reuses that stream for every inner
   observable-set draw, because the (uniform) SEEQST sampling distribution
   doesn't depend on the observables. Here it does — the whole point is to
   adapt the circuit-sampling policy to the observable set — so a fresh batch
   of shots is measured from scratch for *every* inner observable-set draw.
2. **No beta rescaling.** exp_1's single-shot estimator is
   `β(P) · ⟨ψ_pre|P|ψ_pre⟩`, unbiased only because `P`'s measurement circuit is
   drawn from the same uniform, full-ensemble distribution that
   `β(P) = 1/α_P` was computed under — β corrects for the fact that, on
   average, only a `1/β` fraction of draws "hit" (measure `P` exactly). Here,
   every shot used for `O_i` comes from `O_i`'s own exactly-diagonalizing
   circuit *by construction* — there is no "miss" left to correct for, so
   `⟨ψ_pre|P_i|ψ_pre⟩` (no β factor) is already an exactly unbiased
   single-shot estimate of `Tr(P_i ρ)`: ordinary basis-rotated projective
   measurement, not the shadow inverse-map trick. Multiplying by `β(P_i)` here
   would silently inflate every estimate by a factor of 2 or `2^(n+1)` — the
   task module's docstring notes this was checked numerically (the no-beta
   estimator converges to the true expectation value; the β-scaled version
   does not).

## Built-in comparison against random SEEQST

By default (`--no-random-seeqst` to disable), this script *also* runs exp_1's
exact random-SEEQST protocol — uniformly-random `(subset, branch)` per shot,
one shared snapshot stream per state reused across every observable-set draw,
β(P)-rescaled hit-or-miss estimator — on the *same* drawn states and
observable sets as the derandomized protocol, in the same run (rather than
reading a separately-run exp_1's output file). This guarantees a true
apples-to-apples comparison — identical states and observables, not just the
same `n` and seed — and needs no prior exp_1 run to exist. Every row in
`raw.csv`/`summary.csv` carries a `method` column
(`derandomized_seeqst` or `random_seeqst`), and both curves are drawn on every
plot.

## Metrics

For every `n_sample` checkpoint, across the `M` observables, for each method:

- **RMSE**`(N_sample) = sqrt( mean_i (ô_i(N_sample) − o_i)² )`
- **MeanVariance**`(N_sample) = mean_i Var[ô_i^(1)]`, sample variance of
  `O_i`'s own matching-shot history so far.
- **MaxError**`(N_sample) = max_i |ô_i(N_sample) − o_i|`
- **Coverage**`(N_sample)` = fraction of the `M` observables with ≥ 1 matching
  shot in the first `N_sample` total shots.
- **MeanMatchesPerObservable**`(N_sample)` = mean number of matching shots
  per observable, within the first `N_sample` total shots.

For `derandomized_seeqst`, observables with **zero** matching shots so far are
**excluded** from RMSE/MeanVariance/MaxError at that checkpoint (there's
nothing to estimate with yet) — `Coverage` tracks how many that excludes, so
you can tell when the other three metrics have "warmed up" (coverage → 1)
versus when they're still based on a small, possibly unrepresentative subset.
Expect coverage to be worst at small `N_sample` and for observables whose
circuit is rare (low `count_j / M`). For `random_seeqst`, every shot
contributes (via β rescaling, not exclusion) to every observable's running
mean, so coverage is fixed at 1.0 and `MeanMatchesPerObservable = N_sample`
always.

**This exclusion behavior is intentional and load-bearing**: an observable
with zero realized samples has no self-normalized estimate to report, so it
is dropped from that checkpoint's error metrics rather than assigned a
default value of 0 (which would bias RMSE/MaxError downward for exactly the
hardest-to-cover observables).

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size (qubits) |
| `--num-observables` | `M`, random Pauli observables per draw |
| `--n-samples` | list of `N_sample` (total shot budget) checkpoints |
| `--num-state-repeats` | outer-layer repeat count |
| `--num-observable-repeats` | inner-layer repeat count (each gets fresh shots) |
| `--k` | restrict observables to exactly weight `k` (default: fully random weight) |
| `--states` | subset of `haar_random ghz random_stabilizer` to run |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing `v0, v1, ...`) |
| `--no-random-seeqst` | skip the random-SEEQST comparison method (faster) |
| `--quick` | small/fast config for smoke-testing |
| `--quiet` | suppress per-state / per-observable-set progress lines |

## Output (`results/exp_2/v<N>/`)

- `hyperparameters.json` — every setting used for the run.
- `raw.csv` — one row per (method, state_repeat, obs_repeat, state_type, n_sample).
- `summary.csv` — the above averaged over both randomization layers.
- 12 plots: `{rmse, mean_variance, max_error, coverage}_vs_nsample_{haar_random, ghz, random_stabilizer}.png`
  — one line per method. RMSE/MaxError get a `1/√N_sample` reference line
  anchored to the derandomized curve; MeanVariance gets a flat reference at
  the derandomized curve's most-converged value; Coverage gets a flat
  reference at 1.0.

`v0` in this repo is a real full run (`n=10`, 100 observables, 10×10
state/observable repeats — ~7.9 hours), not a smoke test.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_2/exp_2.py --quick
python experiments/exp_2/exp_2.py --n 5 --num-observables 10 \
    --num-state-repeats 3 --num-observable-repeats 3 \
    --n-samples 10 30 100 300 1000 3000 10000
python experiments/exp_2/exp_2.py --n 6 --k 2 --num-observables 20 --states haar_random ghz
python experiments/exp_2/exp_2.py --n 5 --version my_run_name
python experiments/exp_2/exp_2.py --n 5 --no-random-seeqst
```
