# exp_4 — Derandomized SEEQST vs. Huang-Kueng-Preskill's derandomized Pauli shadows

**Script:** `experiments/exp_4/exp_4.py` · **Logic:** `tasks/exp_4_derandomized_comparison.py` · **External dependency:** `codes/predicting-quantum-properties/data_acquisition_shadow.derandomized_classical_shadow` (via `common/external_paths.import_huang_data_acquisition_shadow`) · **Output:** `results/exp_4/v<N>/`

## Question it answers

exp_2/exp_3 derandomize *SEEQST*: given `M` known observables, pick the exact
entangling GHZ-block circuit that measures each one deterministically. But
SEEQST isn't the only derandomizable protocol — Huang, Kueng & Preskill's own
follow-up paper (arXiv:2103.07510, `papers/derandomized_shadows.pdf`)
derandomizes their *original*, fully local (product-measurement) protocol
instead, and ship a reference implementation in the sibling repo
`codes/predicting-quantum-properties`. exp_4 asks: **how much does SEEQST's
larger, entangling measurement ensemble actually buy you**, once both
protocols get to use the same derandomization advantage? It runs both
strategies on identical drawn states and observable sets and compares them
metric-for-metric.

## The two methods

**`derandomized_seeqst`** — exp_2/exp_3's protocol, reused as-is (the
orchestration logic — grouping, shot allocation, hit-conditional checkpoint
evaluation — is duplicated from `tasks/exp_2_derandomized_scaling.py`'s
`run_exp2` into this file's `_run_seeqst_arm`, the same way exp_2 duplicates
`_draw_observables` from exp_1: on purpose, so exp_2 stays completely
untouched and every `exp_N` task file stays independently readable. What's
genuinely *shared*, not duplicated, is `experiments.Archived.exp_3.exp_3`'s
`select_subset_and_branch`/`circuit_for_subset_branch` — that module is
built as a reusable utility, and lives under `Archived/` for folder
organization only (it's a live dependency, not deprecated). See
`experiments/exp_2/exp_2.md` / `experiments/Archived/exp_3/exp_3.md` for the
full derivation:

1. Derandomize each observable into its exact-match SEEQST `(subset,
   branch)` circuit.
2. Group observables sharing a circuit; draw shots i.i.d. from a categorical
   distribution over the resulting distinct circuits, proportional to how
   many observables each one serves.
3. Every shot updates every observable sharing its circuit.

**`derandomized_pauli`** — Huang-Kueng-Preskill's derandomized classical
shadows, called unmodified via
`data_acquisition_shadow.derandomized_classical_shadow(all_observables,
num_of_measurements_per_observable, system_size)`:

1. A single call computes the **entire** measurement schedule up front — a
   deterministic greedy procedure (their Algorithm 1) that, round by round,
   picks the single-qubit Pauli basis for *every* qubit that best serves the
   observables not yet sufficiently measured, using a pessimistic-estimator
   cost function derived from a Chernoff-style tail bound (`eta = 0.9`,
   hardcoded in their code as "a hyperparameter subject to change"). No
   entangling gates at all — every round is a plain product measurement.
2. An observable "matches" a round iff *every* qubit it acts on was measured
   in exactly the basis it needs. Unlike SEEQST's grouping (where a whole
   group of observables always hits together, since group membership *is*
   circuit membership), here a different, changing subset of observables can
   match each round — there's no small fixed set of "circuits" to group
   into.
3. **The schedule itself has no per-shot randomness at all**: given the same
   observables/target/`n`, `derandomized_classical_shadow` always returns
   the identical schedule (verified directly — no `random` module calls
   anywhere in that function; only its sibling `randomized_classical_shadow`
   uses one). The only randomness left is which state/observables get
   drawn, and the physical Born-rule measurement outcomes when that fixed
   schedule is actually executed against the state.

Point 3 is worth pausing on: Huang's algorithm already realizes, natively,
the *mechanism* side of Suggestion A in
`experiments/Archived/exp_3/exp_3_shot_scheduling_suggestions.md` (deterministic
scheduling instead of i.i.d. sampling, to avoid coupon-collector-style
coverage delay) — it's effectively the "gold standard" that suggestion was
proposing we approximate for our own SEEQST protocol. `derandomized_seeqst`
here still draws its shots i.i.d. (categorical over distinct circuits), so
this comparison also indirectly shows what SEEQST's derandomization is
*missing* relative to Huang's, independent of the entangling-vs-product
question.

Both methods use the same **hit-conditional (self-normalized) estimator**:
no rescaling factor needed, since a "hit" already gives an exact, unbiased
single-shot value (see `exp_2.md`'s "no beta rescaling" section — identical
reasoning applies to `derandomized_pauli`, and is exactly what Huang's own
`estimate_exp` reference function does). Observables with zero matches so
far are excluded from RMSE/MeanVariance/MaxError at that checkpoint;
`coverage` tracks how many that excludes — same convention as exp_2.

## One long schedule, many checkpoints — and its caveat

Both methods run **once** per (state, observable-set) trial with a shot
budget of `max(n_samples)`, then get evaluated at every smaller checkpoint
from a prefix of that one run — the same trick exp_1/exp_2 use. For
`derandomized_seeqst` this is exact (shots are i.i.d., so a prefix of a long
run is statistically identical to a fresh short run). For
`derandomized_pauli` it is only an **approximation**: Huang's greedy
algorithm's round-by-round choices depend on the target
`num_of_measurements_per_observable` it's told to aim for (its internal cost
function explicitly weighs `measurements_so_far` against that target, so
later rounds get more "urgent" as the target approaches). A schedule
calibrated for `max(n_samples)` is therefore not bit-for-bit identical to
one calibrated exactly for a smaller checkpoint — recomputing a fresh
schedule at every checkpoint would be more faithful but requires re-running
the derandomization call once per checkpoint instead of once per trial, and
was not done here (see Performance below for why).

**Guaranteed schedule length:** `run_exp4` calls
`derandomized_classical_shadow(all_observables, max_n_sample, n)` — passing
`max_n_sample` as `num_of_measurements_per_observable`. Since an
observable's match count can advance by at most 1 per round, and the
algorithm doesn't stop until *every* observable has reached `max_n_sample`
matches, the returned schedule is mathematically guaranteed to have **at
least** `max_n_sample` rounds (checked with an explicit `assert` in
`tasks/exp_4_derandomized_comparison.py`). In practice, observed schedule
lengths ranged from `max_n_sample` (perfect grouping — every round matches
every observable) up to `max_n_sample x M` (the theoretical worst case, seen
in smoke tests for small `n` with several mutually-incompatible
observables) — `num_schedule_rounds` in `raw.csv` records the actual length
for every trial as a diagnostic.

## Metrics

Identical schema to exp_2 (see `exp_2.md`'s "Metrics" section for full
definitions): **RMSE**, **MeanVariance**, **MaxError**, **Coverage**, and
**MeanMatchesPerObservable**, all as a function of `n_sample` = total
measurement rounds. `raw.csv` additionally carries `num_distinct_circuits`
(SEEQST rows only — not a meaningful concept for `derandomized_pauli`, left
`NaN` there) and `num_schedule_rounds` (both methods: `max_n_sample` for
SEEQST by construction, the actual — often larger — schedule length for
Huang's method).

## Performance

`derandomized_classical_shadow` is unmodified, pure-Python, uncompiled, and
its per-round cost scales as `O(n x M)` (a `cost_function` evaluation per
qubit per candidate letter, each `O(M)`), so total cost per trial is roughly
`O(rounds x n x M)` where `rounds >= max(n_samples)` and grows with how
incompatible the `M` observables are with each other. This tends to be the
slower of the two methods per trial. Observed timings during development
(this sandbox's CPU, not necessarily representative of yours):

- `--quick` (`n=4, M=5, max_n_sample=800`, 6 state draws x 2 obs draws = 12
  trials, both methods): **~10s total**; Huang schedules ranged 1602–4000
  rounds against a target of 800 (up to 5x, since 4 of `n=4`'s worst-case
  grouping was often hit).
- `n=6, M=10, max_n_sample=1000`, 12 trials, both methods: **~26s total**.
- A single trial at `n=5, M=10, max_n_sample=10000` (exp_4's *default* top
  checkpoint): **~16s**; Huang's schedule reached exactly `100000` rounds
  (the theoretical worst case, `target x M`) for this particular
  observable set. Extrapolating (27 trials for the full 3x3x3 default grid)
  suggests a full default run takes on the order of several minutes to
  ~10 minutes — nowhere near exp_2's real 7.9-hour `v0` run, but budget
  accordingly and start with `--quick` or `--no-derandomized-pauli` if
  you're unsure.

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size (qubits) |
| `--num-observables` | `M`, random Pauli observables per draw |
| `--n-samples` | list of `N_sample` (total round budget) checkpoints |
| `--num-state-repeats` | outer-layer repeat count |
| `--num-observable-repeats` | inner-layer repeat count (each gets fresh measurements, both methods) |
| `--k` | restrict observables to exactly weight `k` (default: fully random weight) |
| `--states` | subset of `haar_random ghz random_stabilizer` to run |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing `v0, v1, ...`) |
| `--no-seeqst` | skip the `derandomized_seeqst` arm |
| `--no-derandomized-pauli` | skip the `derandomized_pauli` arm (usually the slower one) |
| `--quick` | small/fast config for smoke-testing |
| `--quiet` | suppress per-state / per-observable-set progress lines |

## Output (`results/exp_4/v<N>/`)

- `hyperparameters.json` — every setting used for the run, including which
  method arms were enabled.
- `raw.csv` — one row per (method, state_repeat, obs_repeat, state_type,
  n_sample), plus `num_distinct_circuits`/`num_schedule_rounds`.
- `summary.csv` — the above averaged over both randomization layers.
- 12 plots: `{rmse, mean_variance, max_error, coverage}_vs_nsample_{haar_random, ghz, random_stabilizer}.png`
  (fewer if a method is skipped) — one line per method. RMSE/MaxError get a
  `1/√N_sample` reference line anchored to `derandomized_seeqst`;
  MeanVariance gets a flat reference at `derandomized_seeqst`'s
  most-converged value; Coverage gets a flat reference at 1.0.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_4/exp_4.py --quick
python experiments/exp_4/exp_4.py --n 5 --num-observables 8 \
    --num-state-repeats 3 --num-observable-repeats 3 \
    --n-samples 10 30 100 300 1000
python experiments/exp_4/exp_4.py --n 6 --k 2 --num-observables 15 --states haar_random ghz
python experiments/exp_4/exp_4.py --n 5 --version my_run_name
python experiments/exp_4/exp_4.py --n 5 --no-derandomized-pauli   # SEEQST arm only, fast
```

## Caveats / what this comparison does and doesn't show

- This is a **fixed-`n`, fixed-M** comparison of two heuristics, not a proof
  that either dominates the other — the derandomized_pauli arm's schedule
  length can genuinely vary a lot (`max_n_sample` to `max_n_sample x M`)
  depending on how compatible the drawn observables happen to be, so results
  are somewhat observable-set-dependent; average over enough
  `--num-observable-repeats` before drawing conclusions.
- The prefix-checkpoint approximation for `derandomized_pauli` (see above)
  means its curve at small `n_sample` is not exactly what you'd get from
  re-deriving a schedule targeted at that smaller budget — treat early
  checkpoints as indicative, not exact.
- Both arms measure the **same physical resource** (one full-system
  measurement round = one shot) as `n_sample`, so the comparison is
  apples-to-apples on that axis, but doesn't account for other possible cost
  asymmetries (e.g. entangling gates being more error-prone or slower on
  real hardware than product measurements) — this is a shot-count
  comparison only.
