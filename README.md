# Shadow benchmark: Pauli vs. Clifford vs. SEEQST

Standalone benchmark code that **imports from, but never modifies**, the two
sibling repos in this project (`codes/predicting-quantum-properties` and
`codes/SEEQST`). Nothing in either repo is touched; this folder only reads
specific functions from them (see `common/external_paths.py` for exactly
what and why).

It compares three classical-shadow measurement ensembles:

1. **Pauli** (`ensembles/pauli_ensemble.py`) — the local random-Pauli /
   single-qubit-Clifford protocol from Huang, Kueng & Preskill, *Predicting
   Many Properties of a Quantum System from Very Few Measurements*
   (arXiv:2002.08953), the protocol implemented operationally in
   `codes/predicting-quantum-properties`.
2. **Clifford** (`ensembles/clifford_ensemble.py`) — the global random-Clifford
   protocol from the same paper (their "Example 1").
3. **SEEQST** (`ensembles/seeqst_ensemble.py`) — the GHZ-block ensemble from
   your draft, `papers/SEEQST_shadows.pdf` (Definition 3 / Theorem 1),
   built using the *unmodified* `build_parallel_entangler_blocks` function
   from `codes/SEEQST/tools/setup.py`.

exp_6 additionally introduces two **tunable SEEQST sub-variants** built on
the same underlying circuits (`ensembles/seeqst_binomial_ensemble.py`,
`ensembles/seeqst_unifsize_ensemble.py`), matching the two SEEQST
sub-families analyzed in the theory note `QIP_notes/SEEQST_shadows_threshold.tex`
-- see the exp_6 section below.

## shadow_norm_seeqst.ipynb: numerics for the shadow norm (SI Eq. S7)

A notebook (not a `run_*.py`/`exp_*.py` script, since it's meant for
interactive exploration) that numerically computes the shadow norm
`||O||^2_shadow` from SI Eq. S7 for SEEQST, which has no known closed form
yet. Key idea: for a fixed sampled unitary U, the sup-over-states in Eq. S7
turns into a linear functional of the state, so the whole thing reduces to
finding the top eigenvalue of a single (small, 2^n x 2^n) operator — no need
to search over candidate states. Since the Pauli and SEEQST ensembles are
finite (3^n and 2^{n+1} elements respectively), this is computed **exactly**
(no Monte Carlo noise) via new `enumerate_unitaries(n)` methods added to
`PauliEnsemble`/`SEEQSTEnsemble`; the Clifford ensemble isn't enumerable, so
it uses Monte Carlo sampling via the new `CliffordEnsemble.sample_unitary_circuit`.

It validates against the two known closed forms first (Pauli: exact match to
`3^k`; Clifford: numerically respects the SI's `<= 3*2^n` bound — note that
one really is only an upper bound in the SI, not a claimed equality, so
landing below it is expected, not a bug), then computes SEEQST's shadow norm
across several system sizes and Pauli types. **Result:** `||P||^2_shadow`
comes out numerically *identical* (to floating-point precision) to the
already-known inverse-map weight `beta(P) = 1/alpha_P` for every case tried —
i.e. `2` for non-trivial pure-Z-type Paulis and `2^(n+1)` for anything with an
X/Y factor. The notebook's markdown includes the short argument for why this
is exact (not a coincidence): every SEEQST unitary is a Clifford circuit, so
it maps Paulis to Paulis, and that structure alone forces
`||P||^2_shadow = beta(P)` for any such "Clifford-type" ensemble — which is
also exactly why the Pauli-ensemble check above reproduces `3^k` on the nose.
This is effectively the analytic form you were looking for; the notebook's
last section suggests how to turn this into a larger regression-test sweep
in a follow-up `.py` script.

## The task: predicting random k-local Pauli observables

Framed exactly as in the paper's Supplementary Information (Sec. 3,
"Example 1" and "Example 2", Theorem S1): for `M` random weight-`k` Pauli
observables, how many single-shot classical shadows are needed to predict
all of their expectation values to additive accuracy `eps`? This is
controlled by the variance (shadow-norm-squared) of the single-shot
estimator, for which the SI derives two closed forms used here as
reference lines:

- Pauli ensemble: `Var <= 3^k`, independent of system size `n` (exact for
  single Pauli-string observables — SI Eq. S17).
- Clifford ensemble: `Var <= 3 * tr(P^2) = 3 * 2^n`, independent of locality
  `k` but exponential in the *full* system size `n` (SI Eq. S16).

No closed form is derived in the SEEQST draft, so its variance is reported
empirically only (derived from `M(P) = alpha_P P` in Theorem 1 / Prop. 9 —
see the module docstring in `ensembles/seeqst_ensemble.py`).

See `tasks/exp_0_pauli_observable_prediction.py` for the full derivation of how a
single sampled snapshot gives an unbiased estimate of `tr(P rho)` for *any*
Pauli `P` simultaneously (all three channels are diagonal in the Pauli
basis), which is what makes this benchmark tractable without ever forming
the exponentially large shadow matrix explicitly.

This "random observables" framing is what exp_0 and exp_1 both study. exp_5
asks a different, physical question on top of the same `common`/`ensembles`
machinery — estimating the energy of a real Hamiltonian rather than random
Pauli strings — see [exp_5](#exp_5-estimating-a-physical-hamiltonians-energy-lattice-schwinger-model)
below.

## Structure

```
shadow_benchmark/
  common/
    external_paths.py     # read-only wiring into the two repos
    pauli_utils.py         # random Pauli strings (fixed-weight, fully-random, bounded-X/Y-weight,
                          # exact-X/Y-weight), label conversion
    states.py               # Haar-random / GHZ / random-stabilizer states
    hamiltonians.py          # lattice Schwinger model (Kokail et al. / HKP Fig. 5) construction
    versioning.py           # shared results/<task>/v<N> auto-incrementing helper
  ensembles/
    base.py                 # shared "diagonal-in-Pauli-basis" estimator machinery
    pauli_ensemble.py
    clifford_ensemble.py
    seeqst_ensemble.py           # flat/uniform SEEQST -- thin wrapper around _seeqst_circuits
    _seeqst_circuits.py          # shared circuit construction/cache for ALL seeqst_* ensembles
                                  # below (parsing, caching, and the qubit<->circuit-block bit
                                  # convention -- one shared implementation so the three SEEQST
                                  # variants can't silently disagree on it)
    seeqst_binomial_ensemble.py  # SEEQSTBinomialEnsemble(q): per-qubit Bernoulli(q) SEEQST
    seeqst_unifsize_ensemble.py  # SEEQSTUniformSizeEnsemble(l): capped-uniform-subset-size SEEQST
  tasks/
    exp_0_pauli_observable_prediction.py  # experiment logic for experiments/exp_0/exp_0.py
    exp_1_error_scaling.py            # experiment logic for experiments/exp_1/exp_1.py
    exp_2_derandomized_scaling.py     # experiment logic for experiments/exp_2/exp_2.py
    exp_4_derandomized_comparison.py  # experiment logic for experiments/exp_4/exp_4.py
    exp_5_energy_estimation.py        # experiment logic for experiments/exp_5/exp_5.py
    exp_6_bounded_xy_weight.py        # experiment logic for experiments/exp_6/exp_6.py
    exp_7_tuned_locality.py           # experiment logic for experiments/exp_7/exp_7.py
  tests/
    cross_check_huang_reference.py   # validates our Pauli estimator against
                                       # Huang's own estimate_exp() on identical data
    verify_derandomized_seeqst.py    # validates experiments/Archived/exp_3/exp_3.py's circuits
    verify_schwinger_hamiltonian.py  # validates common/hamiltonians.py's Hamiltonian construction
    verify_seeqst_binomial_and_unifsize.py  # validates the two new seeqst_* ensembles above:
                                              # closed-form-vs-brute-force, Monte-Carlo bit-convention
                                              # check, and the seeqst_ensemble.py refactor regression
    verify_exp7_observables.py       # validates common/pauli_utils.py::random_exact_xy_spec
                                       # (exp_7's one new generator), incl. the SEEQST-vs-Pauli
                                       # b-independence contrast directly
  experiments/
    exp_0/exp_0.py           # run file #1 -- closed-form theory-bound / shadow-norm view (see exp_0.md)
    exp_1/exp_1.py           # run file #2 -- empirical error-vs-N_sample view (see exp_1.md)
    exp_2/exp_2.py           # run file #3 -- derandomized-SEEQST protocol (see exp_2.md)
    exp_4/exp_4.py           # run file #4 -- derandomized SEEQST vs. Huang's derandomized
                              # Pauli shadows, head to head (see exp_4.md)
    exp_5/exp_5.py           # run file #5 -- physical Hamiltonian (lattice Schwinger model)
                              # energy estimation (see exp_5.md)
    exp_6/exp_6.py            # run file #6 -- bounded-X/Y-weight observables, 5-way ensemble
                              # comparison (see exp_6.md); exp_6/plot_l_dependence.py aggregates
                              # several exp_6 runs into a metric-vs-l view
    exp_7/exp_7.py            # run file #7 -- locality-tuned SEEQST, exact-m vs. at-least-m
                              # X/Y observables, 6-way ensemble comparison (see exp_7.md);
                              # exp_7/plot_m_dependence.py aggregates several exp_7 runs into a
                              # metric-vs-m view
    Archived/                # not deprecated code -- things that aren't standalone run
                              # files with their own results/, so don't fit the exp_N
                              # numbering pattern above:
      exp_3/exp_3.py            # the derandomization module exp_2/exp_4 import (given P,
                                 # returns the SEEQST circuit that measures it exactly) --
                                 # LIVE dependency, not deprecated; see exp_3.md and
                                 # exp_3_shot_scheduling_suggestions.md
      shadow_norm_seeqst.ipynb  # exploratory notebook, genuinely unmaintained
  results/
    exp_0/                            # output of experiments/exp_0/exp_0.py
    exp_1/                            # output of experiments/exp_1/exp_1.py
    exp_2/                            # output of experiments/exp_2/exp_2.py
    exp_4/                            # output of experiments/exp_4/exp_4.py
    exp_5/                            # output of experiments/exp_5/exp_5.py
    exp_6/                            # output of experiments/exp_6/exp_6.py, incl. l_dependence/
    exp_7/                            # output of experiments/exp_7/exp_7.py, incl. m_dependence/
```

Each experiment lives in its own file under `experiments/exp_<N>/`, all
sharing the `common/` and `ensembles/` building blocks, with its logic
factored out into `tasks/`. Every run file writes its data + plots into
its own subfolder of `results/`, at the shadow_benchmark root (not under
`experiments/`) regardless of which subfolder the script itself lives in.
Run file, task file, and results folder are named consistently by
experiment number (`exp_0` through `exp_7`) throughout -- except `exp_3`,
which was moved under `experiments/Archived/` since it has no separate task
module and no `results/` folder of its own (it's a small, self-contained
derandomization utility imported directly by `exp_2`/`exp_4`, not a run
file -- see the note at the top of `exp_3.py` and `exp_3.md` if you're
tempted to delete or further relocate it: exp_2 and exp_4 both import it
directly, so it is NOT deprecated despite the folder name).

**Note:** `exp_0.py` through `exp_4.py` (and `exp_3.py`, before its move
into `Archived/`) were all originally written directly at the
`shadow_benchmark/` root (`exp_0.py` also briefly lived under
`experiments/Archived/` itself, under its old name,
`run_pauli_observable_benchmark.py`, before being promoted into
`experiments/exp_0/`) and later moved into subfolders; each now bootstraps
its own `sys.path` back to the root at the top of the file, so
`python experiments/exp_1/exp_1.py` (etc.) works the same as running it
from the root would have.

## Running it

```bash
cd shadow_benchmark
pip install numpy scipy qiskit pandas matplotlib   # if not already installed

# fast smoke tests (~15s each, exp_4's takes ~10s, exp_5's takes ~20s, exp_6's/exp_7's take ~15s)
python experiments/exp_0/exp_0.py --quick
python experiments/exp_1/exp_1.py --quick
python experiments/exp_2/exp_2.py --quick
python experiments/exp_4/exp_4.py --quick
python experiments/exp_5/exp_5.py --quick
python experiments/exp_6/exp_6.py --quick
python experiments/exp_7/exp_7.py --quick
```

Cross-checks / verification scripts:
```bash
python tests/cross_check_huang_reference.py
python tests/verify_derandomized_seeqst.py
python tests/verify_schwinger_hamiltonian.py
python tests/verify_seeqst_binomial_and_unifsize.py
python tests/verify_exp7_observables.py
```

## exp_0: Pauli vs. Clifford vs. SEEQST against closed-form theory bounds

`experiments/exp_0/exp_0.py` (protocol logic in
`tasks/exp_0_pauli_observable_prediction.py`) is the original benchmark
script for this project — see that module's docstring, or
`experiments/exp_0/exp_0.py`'s own docstring, for the full framing. Unlike
exp_1 onward, its output isn't versioned (`--version` doesn't exist here):
every run overwrites the same flat files directly under `results/exp_0/`.
Results below are from the reference sweep (n = 2..10, M = 12 random
observables/point, 1000 shots/point):

`results/exp_0/pauli_observable_raw.csv` — one row per (state type, n, k,
ensemble, observable). `results/exp_0/pauli_observable_summary.csv` —
aggregated over the `M` random observables per point. Plots:

- `variance_vs_k_<state>.png` — variance vs. Pauli locality `k`, one panel
  per system size `n`, empirical points + theory dashed lines where known.
- `variance_and_complexity_vs_n_<state>.png` — variance and derived sample
  complexity `N_eps = Var/eps^2` vs. system size `n` at fixed `k=2`.

**Headline result** (matches the qualitative story in the paper): at fixed
locality `k=2`, Pauli-shadow variance is flat in `n` (~9, i.e. `3^2`,
regardless of system size), while both Clifford and SEEQST variance grow
exponentially with `n`. SEEQST tracks Clifford closely to somewhat worse
for generic random Pauli observables of fixed small weight — its
`2^{n+1}` inverse-map weight applies to any Pauli with an X or Y factor
(a `(1 - (1/3)^k)` fraction of random weight-`k` draws), so for k=2 the
`2^{n+1}` regime dominates almost as often as it would for a fully global
Clifford observable.

## Known caveat: full-weight (`k = n`) points are noisy

At `k = n`, the Pauli ensemble's match probability is `(1/3)^n`, so
`num_shots=1000` sees essentially zero matching shots once `3^n` exceeds a
few thousand (`n >~ 8`). The *empirical* sample variance at those points is
therefore itself poorly estimated (you can see it plunge unrealistically
in `variance_vs_k_*.png` for the rightmost panels) even though the
theoretical bound `3^n` (dashed line) is exact. This is a Monte-Carlo
sample-size artifact of verifying the estimator, not a real effect —
reliably resolving it needs `num_shots` on the order of the variance
itself (`3^n`), which grows very quickly. Interpret the `k=n` empirical
points with that in mind, or increase `--num-shots` substantially if you
need them to converge.

## exp_1: error scaling vs. number of shadow snapshots

`experiments/exp_1/exp_1.py` is a second, fully-configurable take on the same
underlying question, viewed empirically rather than through the paper's
closed-form bounds. Full explanation (theory, exact definitions, output
format) is in the module docstring at the top of `exp_1.py` itself, and in
`experiments/exp_1/exp_1.md` — read those first.

It uses **two layers of randomization**: an OUTER layer draws a fresh random
state (`--num-state-repeats` times, for each requested state family), and
for each such state an INNER layer draws a fresh random set of `M`
observables (`--num-observable-repeats` times) and evaluates all metrics for
that (state, observable-set) pair. Everything is finally averaged over both
layers pooled together. This is a strict superset of "one random state, one
random observable set, repeated" — set either repeat count to 1 to recover
that simpler design if you want.

For every (state, observable-set, ensemble), it reports, at each requested
`N_sample`: RMSE, mean variance, and max-error of the standard shadow
mean-estimator across the `M` observables.

```bash
cd shadow_benchmark
python experiments/exp_1/exp_1.py --quick                   # ~15s smoke test
python experiments/exp_1/exp_1.py --n 5 --num-observables 6 --num-state-repeats 2 \
    --num-observable-repeats 2 --n-samples 10 30 100 300 1000  # ~30s example (already run; see results/exp_1/v2/)
python experiments/exp_1/exp_1.py --n 6 --num-observables 15 --num-state-repeats 5 \
    --num-observable-repeats 5 \
    --n-samples 10 30 100 300 1000 3000 10000                  # bigger, several minutes
```

**Output is versioned**, so different runs don't overwrite each other:
every run writes to its own `results/exp_1/v<N>/` folder, where `v<N>`
auto-increments to the next unused number by default (or pass
`--version <name>` for an explicit folder name). Each version folder
contains:
- `hyperparameters.json` — every setting used for that run (n, M,
  n_samples, repeat counts, k, states, seed, ...), so old runs stay
  self-describing.
- `raw.csv` — one row per (state_repeat, obs_repeat, state_type, ensemble,
  n_sample).
- `summary.csv` — the above averaged over both randomization layers.
- 9 plots: `{rmse,mean_variance,max_error}_vs_nsample_{haar_random,ghz,random_stabilizer}.png`,
  each a log-log plot of the three ensembles. RMSE and max-error get a
  `1/sqrt(N_sample)` reference line; mean variance gets a flat reference
  line at its most-converged value instead (see below for why).
- `v0/` in this repo is the first run you saved manually (single-layer
  repeats, includes MAE instead of mean variance — an earlier version of
  this script); `v1/` is a `--quick` smoke test of the current version;
  `v2/` is a real example run of the current two-layer design.

**Mean variance, briefly:** Lemma S1 of the Huang SI bounds the variance of
the *single-shot* shadow estimator, `Var[o_hat] <= ||O||^2_shadow` — a fixed
number depending only on the observable/ensemble, not on how many shots you
take. `mean_variance` estimates exactly this quantity: for each observable
we take its first `N_sample` *raw, pre-averaging* single-shot values and
compute their ordinary sample variance, then average over the `M`
observables. Because it's estimating a fixed number, it should flatten out
(converge) as `N_sample` grows, rather than keep shrinking — that's the
qualitative signature to look for, and it's what tells apart "genuinely
lower variance" from "just noisier estimate of variance." It connects to
RMSE via `Var[o_hat(N_sample)] = Var[o_hat(1)]/N_sample`, i.e.
`RMSE(N_sample) ≈ sqrt(mean_variance / N_sample)` — so mean_variance is the
flat quantity, RMSE is what you get after dividing by `N_sample` and taking
a square root. Full derivation is in the module docstring at the top of
`exp_1.py`. Runtime scales roughly linearly in `num_observables`,
`num_state_repeats x num_observable_repeats`, and the top of `n_samples` —
bump those up for smoother, lower-variance curves.

## exp_2 and exp_3: derandomized SEEQST

`experiments/exp_2/exp_2.py` (protocol logic in
`tasks/exp_2_derandomized_scaling.py`) and `experiments/Archived/exp_3/exp_3.py`
(the derandomization module it imports -- lives under `Archived/` for folder
organization only, it's a live dependency, not deprecated) implement and
benchmark a derandomized version of the SEEQST protocol, analogous to
Huang-Kueng-Preskill's derandomized Pauli shadows. See
`experiments/exp_2/exp_2.md` and `experiments/Archived/exp_3/exp_3.md` for
full explanations, and
`experiments/Archived/exp_3/exp_3_shot_scheduling_suggestions.md` for ideas
on improving how shots are scheduled across the derandomized measurement
groups.

## exp_4: derandomized SEEQST vs. derandomized Pauli shadows, head to head

`experiments/exp_4/exp_4.py` (protocol logic in
`tasks/exp_4_derandomized_comparison.py`) runs exp_2/exp_3's
derandomized-SEEQST protocol *and* Huang-Kueng-Preskill's own derandomized
classical shadows (called directly, unmodified, from
`codes/predicting-quantum-properties/data_acquisition_shadow.derandomized_classical_shadow`)
on identical drawn states and observable sets, to see how much SEEQST's
larger, entangling measurement ensemble buys you once both protocols get to
use derandomization. See `experiments/exp_4/exp_4.md` for the full
explanation, including a structural note on why Huang's algorithm's
measurement schedule has no per-shot randomness at all (directly relevant to
the shot-scheduling suggestions above).

## exp_5: estimating a physical Hamiltonian's energy (lattice Schwinger model)

`experiments/exp_5/exp_5.py` (protocol logic in
`tasks/exp_5_energy_estimation.py`, Hamiltonian construction in
`common/hamiltonians.py`) is the first experiment here to move past *random*
observables and states to a real physical question: how well do Pauli /
Clifford / SEEQST shadows estimate the energy of an actual Hamiltonian? The
Hamiltonian is the lattice Schwinger model of Kokail et al.
(arXiv:1810.03421 Eq. 1-2), the exact system Huang-Kueng-Preskill's own
main-text Fig. 5 (arXiv:2002.08953) uses for this same purpose, and the
system Kokail et al. ran variational quantum simulation on with up to 20
trapped ions. `common/hamiltonians.py`'s construction is verified 7
independent ways in `tests/verify_schwinger_hamiltonian.py` (a from-scratch
plain-numpy cross-check, a hand-solved n=2 special case, Hermiticity, charge
conservation, an HKP-Eq.-S32 structural check, ground-energy cross-check,
and the `neel` reference state's definition).

Full explanation (why this needs a different design than exp_1's two-layer
randomization, exact metric definitions, output format) is in the module
docstring at the top of `exp_5.py` itself, and in
`experiments/exp_5/exp_5.md` — read those first. In short: the Hamiltonian's
own terms (never randomly drawn) are grouped into three physically distinct
types (`z_single`, `zz_long_range`, `hopping`) and every metric is reported
per group, plus a total-energy metric built from the *same* shared snapshot
data the per-term metrics use (so its variance captures cross-term
correlations automatically, no separate derivation needed).

```bash
cd shadow_benchmark
python experiments/exp_5/exp_5.py --quick        # ~20s smoke test
python experiments/exp_5/exp_5.py --n 6           # real run, ~5-6 minutes (already run; see results/exp_5/v1/)
python experiments/exp_5/exp_5.py --n 8 --num-repeats 8 \
    --n-samples 10 30 100 300 1000 3000 10000     # bigger, several minutes
```

**Headline result:** SEEQST wins decisively on the Hamiltonian's Z-type terms
(`z_single`: mass; `zz_long_range`: Gauss law) — exactly as its `β=2`
inverse-map weight predicts, with the largest margin on `zz_long_range`
since Pauli's `3^k` cost grows with weight while SEEQST's pure-Z cost
doesn't. But Pauli wins the `hopping` terms by more than an order of
magnitude (SEEQST is worst of all three ensembles there, its `β=2^{n+1}`
inverse-map weight applying to any X/Y-containing string). Strikingly, the
**aggregate total-energy estimate still favours Pauli**, even though SEEQST
wins on more individual terms (16 of 26 at n=6) — the 10 hopping terms carry
enough weight in the sum that SEEQST's huge hopping-term variance dominates.
Whether shadows help a Hamiltonian's energy estimate depends on the *mix* of
term types it contains, not just which ensemble wins "on average." See
`experiments/exp_5/exp_5.md` for the full numbers, including the
`energy_variance` vs. `energy_variance_naive` comparison showing Pauli and
SEEQST/Clifford disagree even on the *sign* of the shared-snapshot
correlation effect on the aggregate variance.

## exp_6: bounded-X/Y-weight observables, 5-way ensemble comparison

`experiments/exp_6/exp_6.py` (protocol logic in
`tasks/exp_6_bounded_xy_weight.py`) is the empirical companion to the theory
note `QIP_notes/SEEQST_shadows_threshold.tex`: it compares FIVE ensembles --
`pauli`, `clifford`, `seeqst_uniform` (the original flat/uniform SEEQST),
and two new tunable SEEQST variants introduced alongside this experiment,
`seeqst_binomial` (`ensembles/seeqst_binomial_ensemble.py`, per-qubit
Bernoulli(q) inclusion, default `q=1/(n+1)`) and `seeqst_unifsize_l`
(`ensembles/seeqst_unifsize_ensemble.py`, subset size drawn uniformly from
`{0,...,l}`, then a uniformly random subset of that size) -- on Pauli
strings built to isolate exactly the theory note's tunable knob: full
weight `n`, with the *number* of X/Y factors drawn uniformly from
`{1,...,l}` and every other qubit fixed to Z. `l` is a hyperparameter
(`--l`) shared by the observable generator and the `seeqst_unifsize_l`
ensemble's own sampling distribution. Both new ensembles' closed-form
`inverse_weight` formulas, and the qubit/circuit-block bit convention they
(and the refactored `seeqst_ensemble.py`) all now share via
`ensembles/_seeqst_circuits.py`, are verified in
`tests/verify_seeqst_binomial_and_unifsize.py` (closed-form vs. brute-force
from the operational definition, a Monte-Carlo sampling check, and a
before/after regression check against the pre-refactor `seeqst_ensemble.py`).

Like exp_1, this experiment reports only RMSE and MaxError (no
`mean_variance`); unlike exp_1, `l` is swept ACROSS separate runs, not
within one (see exp_6.py's module docstring for why), and
`experiments/exp_6/plot_l_dependence.py` aggregates several such runs into
a metric-vs-`l` view. Full explanation (why `l` governs two things at once,
exact CLI options, output format) is in `exp_6.py`'s own module docstring
and in `experiments/exp_6/exp_6.md` -- read those first.

```bash
cd shadow_benchmark
python experiments/exp_6/exp_6.py --quick              # ~15s smoke test
python experiments/exp_6/exp_6.py --n 6 --l 2           # real run, ~100s (already run; see results/exp_6/v2/)
python experiments/exp_6/plot_l_dependence.py v1 v2 v3 v4   # aggregate several --l runs (already run; see results/exp_6/l_dependence/)
```

**Headline result:** at `n=6`, both new SEEQST variants start out clearly
best at `l=1` (`seeqst_unifsize_l` beats Clifford by ~2.3x, `seeqst_binomial`
by ~1.6x), then degrade monotonically as `l` grows -- `seeqst_binomial`
steeply (consistent with its `q^{-k}` closed form, geometric in the X/Y
weight `k`), becoming the *worst* of all five ensembles by `l=6`;
`seeqst_unifsize_l` degrades more gently and stays competitive with
Clifford out to `l≈4`. Meanwhile `pauli`, `clifford`, and `seeqst_uniform`
are all flat across `l`, exactly as their `l`-independent closed-form `β`
predicts under this full-weight observable family (`plot_l_dependence.py`
checks this automatically). The `seeqst_unifsize_l`/Clifford crossover
falls between `l=2` and `l=4` in every state type tested (weight fraction
`f=l/n` between 1/3 and 2/3), bracketing the theory note's asymptotic
prediction `f*≈0.609` about as closely as 4 sampled `l`-points at `n=6`
can be expected to. See `experiments/exp_6/exp_6.md` for the full
per-state-type numbers. As with exp_5, there is no single best ensemble --
which one wins is governed entirely by where the observable's X/Y-content
sits relative to each SEEQST variant's tuning.

## exp_7: locality-tuned SEEQST, exact-*m* vs. at-least-*m* observables

`experiments/exp_7/exp_7.py` (logic in `tasks/exp_7_tuned_locality.py`) is
the empirical companion to `notes/main_theorem/main_theorem.tex` (Theorem 1 /
Corollary 5): tuning `seeqst_binomial`'s `q` to a *known* locality `m`
(`q*=m/n`) rather than exp_6's one-size-fits-all `q=1/(n+1)`. Six ensembles
are compared -- exp_6's five, plus a second, correctly-tuned Binomial
variant (`seeqst_binomial_tuned`, alongside `seeqst_binomial_untuned` kept
for contrast) -- on two selectable observable families
(`--observable-type`): `exact` (X/Y-weight is exactly `m`, the theorem's own
regime, with `--rest-mode` controlling whether the other qubits are Z, I,
or a fresh per-qubit coin flip) and `at_least` (X/Y-weight `>= m` up to a
cap `--l`, reusing `pauli_utils.random_bounded_xy_spec`'s existing floor
support unchanged). `rest_mode=random` (the default) is more than a style
choice: it makes every SEEQST ensemble's cost provably independent of the
Z/I split (the general eigenvalue theorem's own claim) while Pauli's own
cost is not, and `tests/verify_exp7_observables.py` checks that contrast
directly rather than leaving it as a claim. Building `at_least` mode
surfaced a real bug worth knowing about if you extend this further:
`seeqst_unifsize_tuned` cannot be tuned to the floor `m` the way
`seeqst_binomial_tuned` can -- its `inverse_weight` *raises* (not just
degrades) once weight exceeds its cap, so under `at_least` it must be tuned
to the generator's own upper cap `l` instead, exactly mirroring how exp_6
tuned `seeqst_unifsize_l` to the generator's cap. Full explanation and CLI
options are in `exp_7.py`'s module docstring and `experiments/exp_7/exp_7.md`.

```bash
cd shadow_benchmark
python experiments/exp_7/exp_7.py --quick                      # ~15s smoke test
python experiments/exp_7/exp_7.py --n 6 --m 3                   # real run, ~115s (already run; see results/exp_7/v2/)
python experiments/exp_7/plot_m_dependence.py v0 v1 v2 v3 v4    # aggregate (already run; see results/exp_7/m_dependence/)
```

**Headline result:** at `n=6`, `exact` mode, `rest_mode=random`, both tuned
SEEQST ensembles beat `pauli` at every tested `m=1..5` -- not just
asymptotically -- and `seeqst_binomial_tuned` pulls dramatically ahead of
the untuned baseline as `m` grows (nearly tied at `m=1`, `>15x` better by
`m=5`), the `why_binomial_underperforms.md` mechanism running in reverse.
Both tuned ensembles lose to `clifford` only in a band around `m/n≈1/2`
(`m=2,3,4` here) and beat it at the extremes (`m=1,5`) -- Proposition 4's
narrow Clifford-window prediction, appropriately wider than its asymptotic
`O(n^{-1/2})` shrinkage at this modest `n`. A small `at_least`-mode
comparison (`m=2` floor, cap `l=2` vs. `l=5`) shows the expected
degradation as the true weight is allowed to exceed the tuned floor:
`seeqst_binomial_tuned` gets ~2.2x worse, the untuned baseline ~4.8x worse,
over the same widening. See `experiments/exp_7/exp_7.md` for full numbers.

## What's *not* covered here (possible follow-ups)

- Non-random, structured observables tailored to SEEQST's GHZ-block
  structure (e.g. stabilizers of the GHZ blocks themselves), where SEEQST
  is expected to show a real advantage over generic-random-Pauli tests.
- GHZ-state fidelity estimation (main-text Fig. 2 of the Huang paper) and
  entanglement-entropy prediction (Fig. 4) — natural next `run_*.py` files
  reusing the same `common/` and `ensembles/` code.
- Derandomized Clifford-ensemble schemes (SEEQST and Pauli are both
  derandomized now, in `exp_2`-through-`exp_4`; Clifford has no
  derandomization scheme implemented anywhere in this project yet).
- Smarter shot scheduling for derandomized SEEQST — see
  `experiments/Archived/exp_3/exp_3_shot_scheduling_suggestions.md` (exp_4
  shows Huang's derandomized-Pauli protocol already gets Suggestion A's
  deterministic scheduling "for free").
- exp_5 fixes system size `n` per run (like exp_1); a dedicated `n`-sweep at
  fixed, large `N_sample` (in the style of exp_0's `variance_vs_n` plots)
  would show directly how each ensemble's disadvantage on the Schwinger
  model's `hopping`/Clifford-dominated terms scales with system size.
- exp_6 fixes `n=6` for its real runs (4 `l`-points); a larger-`n` sweep
  would let the `seeqst_unifsize_l`/Clifford crossover `l`-value be compared
  directly against the theory note's asymptotic `f*≈0.609` prediction with
  less finite-size discreteness than 4 points at `n=6` allows.
- Applying exp_2/exp_4's derandomization (currently only benchmarked against
  random Pauli observable sets) to exp_5's *fixed, known-in-advance*
  Hamiltonian terms — derandomization should matter more here than for
  random observables, since the observable set is fixed and small, exactly
  the regime derandomization is designed for.
- VQE-trajectory tracking: instead of just the exact ground state and the
  Neel state, sweep a family of states interpolating between them (e.g. a
  parameterized ansatz circuit) to see how shadow-estimated energy error
  behaves as a simulated VQE run converges — closer still to Kokail et al.'s
  actual "self-verifying" protocol.
- exp_7's real sweep only covers `n=6` under `rest_mode=random`; a larger-`n`
  run (or a `rest_mode=identity` sweep, matching `main_theorem.tex`'s own
  `3^m`-vs-Pauli convention exactly rather than the random-split variant)
  would show a sharper, cleaner crossover, closer to
  `main_theorem_figure.pdf`'s `n=100` panel than 5 points at `n=6` allows.
- exp_7's `at_least` mode is currently illustrated with only two cap values
  (`l=2` vs. `l=5`) at one fixed `m`; a proper cap sweep at fixed `m`
  (mirroring exp_6's own `l`-sweep machinery) would map out the full
  floor-tuned degradation curve rather than two endpoints.
