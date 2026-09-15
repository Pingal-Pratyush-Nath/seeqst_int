# Improving exp_2's shot scheduling

**Context for:** `experiments/exp_2/exp_2.py` / `tasks/exp_2_derandomized_scaling.py` (`run_exp2`)

## The current rule, and its weakness

Once the `M` observables are derandomized and grouped into `C` distinct
SEEQST circuits (group `j` matches `c_j` observables, `Σ_j c_j = M`),
`run_exp2` draws the circuit for **every** shot i.i.d. from a fixed
categorical distribution proportional to demand:

```python
probs = np.array([counts[gi] / M for gi in range(len(distinct_keys))])
...
shot_group_idx = rng.choice(len(distinct_keys), size=max_n_sample, p=probs)
```

This gets the *target* proportions right (`P(S_j) = c_j / M`), but realizing
them via i.i.d. sampling means the *realized* count for group `j` after `t`
shots is `Binomial(t, c_j/M)`, with standard deviation
`√(t · (c_j/M) · (1 − c_j/M))`. Two consequences, both visible in exp_2's own
`coverage` metric:

- Rare groups (small `c_j/M`) can go many shots without being drawn at all —
  effectively a coupon-collector delay — which is exactly why `coverage < 1`
  at small `N_sample` and why some observables are excluded from
  RMSE/MeanVariance/MaxError until `N_sample` is large enough.
- The schedule never adapts: group `j`'s probability is fixed at `c_j/M` for
  the entire run, regardless of how many matching shots it has already
  received or how noisy its current estimate looks.

Below are three ways to improve on this, roughly in order of implementation
effort, and they compose (see "Putting it together").

## A. Deterministic allocation instead of i.i.d. sampling

The simplest fix doesn't touch the target proportions `c_j/M` at all — it
just realizes them **without sampling noise**. Instead of drawing each shot's
group i.i.d., use a *deficit* (largest-remainder-style) rule, the same idea
behind apportionment methods and weighted round-robin schedulers:

1. Maintain, for each group `j`, its fractional entitlement after `t` shots,
   `e_j(t) = t · (c_j / M)`, and its actual realized count so far, `a_j(t)`.
2. For shot `t+1`, assign the group with the largest deficit:
   `j* = argmax_j [ e_j(t) − a_j(t) ]`.
3. Increment `a_{j*}`.

This is a classical result (related to the Bresenham line algorithm / the
"largest remainder method" for proportional representation): it guarantees
`|a_j(t) − t·(c_j/M)| < 1` for **every** `j` and **every** `t` simultaneously
— i.e. every group tracks its target proportion within one shot, at all
times, deterministically. In particular, any group with `c_j ≥ 1` gets its
first shot within `⌈M/c_j⌉` shots, guaranteed — not "in expectation" as under
i.i.d. sampling. This directly attacks the coverage warm-up problem: coverage
would become a deterministic, monotonically-improving function of `N_sample`
instead of a noisy one.

This is a pure implementation change (replace the single `rng.choice(...,
p=probs)` call with a small loop or a vectorized deficit-tracking routine)
with no new hyperparameters and no change to what's being targeted — lowest
risk, and worth doing regardless of whether B or C are pursued.

## B. Confidence-bound-driven greedy scheduler (Hoeffding, à la derandomized shadows)

This one changes *what's being targeted*, not just how it's realized —
analogous to the greedy derandomization in Huang, Kueng & Preskill's
derandomized-shadows algorithm (`papers/derandomized_shadows.pdf`, Algorithm
1), but applied to *scheduling shots across our already-derandomized
circuits* rather than to *choosing single-qubit measurement bases*.

**Setup.** Each hit for observable `i` (a shot using `i`'s own group's
circuit) yields an exact `±1` value — not just bounded, but exactly `±1`,
since the whole point of exp_3's circuits is that `U P_i U†` is diagonal with
`±1` entries (see `exp_3.md`). Because all `c_j` members of group `j` are hit
*together*, on exactly the shots where group `j` is chosen, every member of
group `j` shares the same hit count at any point in the run — call it
`H_j(t)`, the number of times group `j` has been chosen among the first `t`
shots.

**Per-observable tail bound.** For observable `i` in group `j`, its estimate
`ô_i` after `t` shots is a mean of `H_j(t)` i.i.d. `±1`-valued samples with
mean `o_i ∈ [−1, 1]`. Hoeffding's inequality (range `2`, so the standard
`2 exp(−2nε²/(range)²)` form becomes `2 exp(−n ε²/2)`) gives:

```
Pr[ |ô_i − o_i| ≥ ε ]  ≤  2 exp( −H_j(t) ε² / 2 )
```

**Union bound over all M observables.** Summing (grouping by shared `H_j`)
gives a bound on the probability that *any* observable is off by more than
`ε` after `t` total shots:

```
Conf_ε(t)  =  Σ_{i=1}^{M} 2 exp( −H_{j(i)}(t) ε² / 2 )
           =  Σ_{j=1}^{C} c_j · 2 exp( −H_j(t) ε² / 2 )
```

(the second form follows because all `c_j` members of group `j` contribute
an identical term). This is the SEEQST analogue of the confidence potential
that drives HKP's derandomization procedure.

**Greedy rule.** `Conf_ε(t)` can't be minimized globally over the whole
schedule in advance (it's a combinatorial sequencing problem), so — exactly
as HKP does — minimize it **greedily**, one shot at a time. If shot `t+1` is
assigned to group `j*`, only `H_{j*}` changes, so:

```
Conf_ε(t+1 | choose j*)  =  Conf_ε(t)  −  c_{j*}·2exp(−H_{j*}(t)ε²/2)·(1 − exp(−ε²/2))
```

The factor `(1 − exp(−ε²/2))` doesn't depend on `j*`, so minimizing
`Conf_ε(t+1)` is equivalent to maximizing, over `j`:

```
score(j)  =  c_j · exp( −H_j(t) ε² / 2 )
```

**i.e. at every shot, measure the circuit for the group maximizing
`c_j · exp(−H_j(t)·ε²/2)`.** This rewards groups that are large (many
observables riding on that one circuit, `c_j` big) *and* currently
under-sampled relative to the target resolution `ε` (`H_j(t)` small) — unlike
the current rule, it explicitly reacts to how far along each group already
is.

**Two limits worth knowing:**
- As `ε → 0`, `exp(−H_jε²/2) → 1` for any finite `H_j`, so `score(j) → c_j`:
  the rule degenerates to "always fill the currently-largest group" —
  essentially a fairness-blind version of scheme A, useful if you only care
  about eventual coverage, not a specific accuracy target.
- Large `ε` makes `score(j)` insensitive to `H_j` until `H_j` gets fairly
  large, so early shots go almost entirely to the biggest groups first, and
  the schedule only starts "correcting" toward smaller groups once the big
  ones are well-served — pick `ε` close to the resolution you actually care
  about resolving (e.g. tie it to the RMSE target you'd otherwise read off
  an exp_2 plot).

**Where this plugs in:** replace the single vectorized
`shot_group_idx = rng.choice(...)` draw in `run_exp2` with a step-by-step (or
chunked, for speed) loop that maintains `H_j(t)` per group and recomputes
`score(j)` — the shot-by-shot dependency on evolving state makes this
inherently sequential, unlike the current one-shot vectorized draw. `ε`
becomes a new required input to the scheduler (it does not need to match any
`n_sample` checkpoint; it's simply the resolution the *schedule* optimizes
for, while `raw.csv`/`summary.csv` can still report the usual metrics at
whatever checkpoints you like).

## C. Adaptive, variance-aware sequential allocation

B's Hoeffding bound treats every observable as if it had the worst-case
variance (a `±1` Bernoulli-like variable can have variance up to 1, at
`o_i = 0`), but the *actual* per-observable variance is
`Var[x_i] = 1 − o_i²`, which for a near-deterministic observable
(`o_i` close to `±1`) can be far smaller — such an observable needs far fewer
matching shots to reach the same `ε` than one near `o_i = 0`. This is the
classical stratified-sampling ("Neyman allocation") insight: optimal sample
sizes per stratum should be proportional to the stratum's standard
deviation, not just its size.

Since the true `o_i` isn't known in advance, this has to be adaptive:

1. Warm-start with A and/or B (or the current fixed-proportion rule) for a
   small initial batch of shots.
2. Periodically (every batch of shots, not every single shot — recomputing
   after literally every shot is unnecessary overhead) recompute each
   group's **empirical** variance from its matching-shot history so far, and
   fold that into the allocation rule — e.g. swap Hoeffding for an
   **empirical-Bernstein** bound (Maurer & Pontil), which replaces the
   `H_j(t)` in B's `exp(−H_j(t)ε²/2)` term with something like
   `exp(−H_j(t)ε² / (2σ̂_j(t)² + O(ε)))`, so low-variance groups get
   correctly deprioritized once enough shots reveal they're
   already-near-converged, freeing budget for the groups that actually need
   it.
3. Since a whole group's `c_j` observables are always sampled together, the
   right per-group variance proxy is probably the **worst** (max) empirical
   variance among a group's members, not the average — the goal is
   presumably "get every individual observable within `ε`," so the hardest
   member of a group is the one that determines when that group can be
   deprioritized.

This is exactly the spirit of **"Thrifty Shadow Estimation"** (Helsen &
Walter, `papers/shadows_THRIFTY.pdf`): reusing/scheduling measurement
circuits adaptively based on variance information gathered *during* the run,
rather than committing to a static schedule computed once up front. That
paper works out its own variance estimator and (likely tighter,
variance-aware) tail-bound machinery for exactly this kind of sequential
circuit-reuse problem — worth reading closely before implementing this one,
rather than re-deriving the empirical-Bernstein step from scratch, since it's
directly relevant prior art for this exact problem (not just tangentially
related, the way the Hoeffding-based derandomized-shadows paper is for B).

## Putting it together

A and B/C aren't competing — A is a *mechanism* (how to realize any target
allocation with minimal discrepancy, no sampling noise), while B and C are
*policies* (what the target allocation should be, and whether it should
change over time). The natural end state is A's deficit-scheduling mechanism
driven by B's (or C's) time-varying target weights `score(j)` instead of the
current fixed `c_j/M` — i.e. still allocate deterministically rather than by
i.i.d. draw, but let the target proportions adapt shot-by-shot as B/C
recommend.

**Suggested implementation order:** A first (pure mechanism swap, zero new
hyperparameters, immediately improves the coverage-warmup behavior already
visible in exp_2's plots); B second (needs choosing `ε` and a sequential
scheduling loop, but is a direct, well-justified extension with a clean
derivation and a close analogue in prior art); C last (most powerful
asymptotically, but should be implemented after actually reading the Thrifty
paper's specific estimator, rather than improvised).
