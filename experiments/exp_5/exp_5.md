# exp_5 — Estimating a physical Hamiltonian's energy from classical shadows

**Script:** `experiments/exp_5/exp_5.py` · **Logic:** `tasks/exp_5_energy_estimation.py` ·
**Hamiltonian:** `common/hamiltonians.py` · **Output:** `results/exp_5/v<N>/`

## Question it answers

exp_0 and exp_1 both study **random** Pauli observables on **random** states —
a clean way to isolate each ensemble's raw statistical behaviour, but not a
question any real experiment asks on its own. This experiment asks the
question classical-shadow papers actually lead with: given a **real physical
Hamiltonian**, how well — and via which ensemble — can you estimate its
energy from classical shadows? This is Huang, Kueng & Preskill's own
main-text Fig. 5 (arXiv:2002.08953), and the measurement task behind Kokail
et al.'s "self-verifying" variational quantum simulation
(arXiv:1810.03421): checking how close a prepared state's energy is to the
target, from measurement data that's recorded once and reused.

The Hamiltonian is the **lattice Schwinger model** — a spin chain with a
staggered mass term, an XY-hopping term, and a long-range ZZ term from
integrating out the U(1) gauge field via Gauss's law (Kokail et al. Eq. 1-2 /
HKP SI Eq. S32). See `common/hamiltonians.py`'s module docstring for the
construction, and `tests/verify_schwinger_hamiltonian.py` for its
correctness verification (7 independent checks, including a from-scratch
plain-numpy cross-check and a hand-solved n=2 special case). Default
couplings `w=1, m=0.9, g=1` are Kokail et al.'s own values (their Fig. 2a
caption, the 20-ion simulation).

## Why this needs a different design than exp_1, not just a different observable list

exp_1 needs **two** randomization layers (state, then observables) because
both are, by design, arbitrary. Here **neither** is arbitrary:

- The **observables** are the Hamiltonian's own Pauli terms, fixed by
  `(n, w, m, g)` and never redrawn. They come in exactly three structural
  types (`common/hamiltonians.classify_term`, verified never to produce
  anything else): `z_single` (the staggered mass), `zz_long_range` (the
  Gauss-law term — "long range" because `L_j^2` correlates every pair of
  links up to `j`, unlike the strictly-nearest-neighbour hopping), and
  `hopping` (nearest-neighbour XX/YY). All metrics are reported broken down
  by these three groups, because — unlike a random Pauli set — the
  ensembles do **not** perform uniformly across them (see Result headline).
- The **state** is one of a few specific, physically meaningful choices, not
  a random draw:
  - `ground_state` (default) — the exact ground state (dense
    diagonalization). The state a VQE/VQS run is actually trying to reach.
  - `neel` — the staggered classical/product reference state (the "bare
    vacuum" — zero entanglement, and by construction exactly zero
    expectation value on every hopping term). The standard VQE starting
    point for this model, and the opposite extreme from `ground_state` on
    the entanglement axis.
  - `haar_random` (available via `--state-types`, not run by default) — a
    generic structureless baseline, reusing `common/states.py`.

So the only thing left to randomize is measurement (shot) noise —
`--num-repeats` independent snapshot streams per state, purely for error
bars. One randomization layer, not two. Snapshots are still drawn **once**
per (repeat, state, ensemble) and reused (`evaluate_snapshots`) across every
term and every `N_sample` checkpoint — the same "measure once, mine many
times" discipline as exp_1.

## Metrics

**Per term group** (`z_single` / `zz_long_range` / `hopping`): the same
RMSE / MeanVariance / MaxError as exp_1 (see its docstring for the exact
Lemma S1 connection), computed within each group.

**Energy level** (the actual quantity of interest): the running-mean total
energy estimate `E_hat(N) = identity_offset + Σ_i c_i · ô_i(N)`, built from
the **same shared per-shot data** as the per-term metrics — so its variance,
`energy_variance`, automatically includes every cross-term covariance
`Cov[ô_i, ô_j]` induced by sharing one measurement stream across all terms,
with no separate derivation needed. `energy_variance_naive` is the variance
you'd get by (incorrectly) assuming the per-term estimators are independent,
`Σ_i c_i² Var[ô_i]` — plotted alongside the real value so the effect of
shared-snapshot correlations is visible directly (see Result headline: the
two ensembles disagree on the *sign* of this effect).

## CLI options

| Flag | Meaning |
|---|---|
| `--n` | system size (qubits/sites; must be even) |
| `--w`, `--m`, `--g` | Hamiltonian couplings (default: Kokail et al.'s own `1, 0.9, 1`) |
| `--n-samples` | list of `N_sample` checkpoints to evaluate at |
| `--num-repeats` | independent measurement-noise repeats (error bars) |
| `--state-types` | subset of `ground_state neel haar_random` to run |
| `--seed` | RNG seed |
| `--version` | explicit output folder name (default: auto-incrementing `v0, v1, ...`) |
| `--quick` | small/fast config for smoke-testing |
| `--quiet` | suppress per-draw progress lines |

## Output (`results/exp_5/v<N>/`)

- `hyperparameters.json` — every setting, plus the Hamiltonian's term-group
  counts and exact ground energy (provenance).
- `raw_terms.csv` — one row per (repeat, state_type, ensemble, n_sample, term_group).
- `raw_energy.csv` — one row per (repeat, state_type, ensemble, n_sample).
- `summary_terms.csv`, `summary_energy.csv` — the above averaged over repeats.
- Plots, per state type: 9 term-group plots
  (`{rmse,mean_variance,max_error}_vs_nsample_{z_single,zz_long_range,hopping}_<state>.png`,
  log-log, same reference-line convention as exp_1) + 2 energy plots
  (`energy_error_vs_nsample_<state>.png`,
  `energy_variance_vs_nsample_<state>.png` — the latter shows actual vs.
  naive/uncorrelated variance side by side). 11 plots × `len(state_types)`
  (22 for the default two state types).

`v0` is a `--quick` smoke test (n=4); `v1` is a real example run
(`--n 6 --num-repeats 4 --n-samples 10 30 100 300 1000 3000`, ~5.5 minutes —
Clifford's `random_clifford` sampling dominates the runtime, same as
elsewhere in this codebase).

## Result headline

At `n=6` (26 Hamiltonian terms: 6 `z_single`, 10 `zz_long_range`, 10
`hopping`), the per-term-group picture matches the ensembles' inverse-map
weights `β(P)` exactly:

- **`zz_long_range` (Gauss law):** SEEQST wins by far (`Var≈1.3`, its exact
  `β=2` for any pure-Z-type string, *independent of weight*) vs. Pauli
  (`Var≈8`, `β=3²=9`, growing with weight) vs. Clifford (`Var≈64`,
  `n`-dependent regardless of locality). SEEQST's advantage over Pauli here
  is *larger* than for `z_single` below, precisely because these are
  weight-2 strings — Pauli's `3^k` cost grows with weight while SEEQST's
  pure-Z cost doesn't.
- **`z_single` (mass):** SEEQST still wins (`Var≈1.2` vs. Pauli's `Var≈2.2`),
  by a smaller margin, since `β=2` vs. `β=3` is a weight-1 comparison.
- **`hopping` (XY):** Reverses completely — Pauli wins by over an order of
  magnitude (`Var≈9`, matching `3²` almost exactly) vs. Clifford (`Var≈64`)
  vs. **SEEQST worst of all three** (`Var≈128`, matching its own
  `β=2^{n+1}` for any X/Y-containing string almost exactly — the
  `shadow_norm_seeqst.ipynb` finding that SEEQST's shadow norm exactly
  equals its inverse-map weight shows up numerically here too).

**But the aggregate total-energy metric belongs to Pauli**, not SEEQST,
even though SEEQST wins on 16 of the 26 terms (all `z_single` +
`zz_long_range`): `energy_variance` at `N_sample=3000` is ≈105 for Pauli vs.
≈620 for SEEQST vs. ≈1490 for Clifford (ground state; Néel is similar). The
10 `hopping` terms carry large enough coefficients that SEEQST's huge
hopping-term variance (`≈128`, ~14× Pauli's `≈9`) dominates the weighted sum
and outweighs its wins elsewhere — "winning on more individual terms"
doesn't imply "winning on the physical quantity built from all of them
together." **Whether shadows help a given Hamiltonian's energy estimate
depends on the *mix* of term types it contains, not just on which ensemble
is best "on average."**

`energy_variance_naive` vs. `energy_variance` also shows the two ensembles
disagree on the *sign* of the shared-snapshot correlation effect: for
Pauli, the true (correlated) variance is *lower* than the naive
independent-sum estimate (≈105 vs. ≈147 — net anti-correlation across
terms); for SEEQST and Clifford it's *higher* (≈620 vs. ≈353, and ≈1490 vs.
≈1436 — net positive correlation). Assuming independence would have
under-estimated SEEQST's and Clifford's actual sample complexity here.

## Usage

```bash
cd shadow_benchmark
python experiments/exp_5/exp_5.py --quick
python experiments/exp_5/exp_5.py --n 6
python experiments/exp_5/exp_5.py --n 8 --num-repeats 8 \
    --n-samples 10 30 100 300 1000 3000 10000   # bigger, several minutes (Clifford sampling dominates)
python experiments/exp_5/exp_5.py --n 6 --state-types ground_state neel haar_random
python experiments/exp_5/exp_5.py --n 6 --m 0.0    # switch off the staggered mass term
python experiments/exp_5/exp_5.py --n 6 --version my_run_name
```
