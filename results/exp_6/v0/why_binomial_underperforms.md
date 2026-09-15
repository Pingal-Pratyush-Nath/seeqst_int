# Why does `seeqst_binomial` underperform `seeqst_uniform` in this run?

**Run analyzed:** `results/exp_6/v0/` — `n=7, l=4, q_binomial=0.125 (=1/(n+1)), num_observables=100,
num_state_repeats=3, num_observable_repeats=100, states=[haar_random], n_samples up to 25000`
(`hyperparameters.json`).

## TL;DR

Not a bug. `seeqst_binomial(q=1/(n+1))` is tuned for observables with very *few* X/Y factors
(its cost is minimized right around X/Y-weight `k≈1`), and its per-shot cost `β(k)` then grows
**geometrically** in `k`. `seeqst_uniform`'s cost, by contrast, is exactly **flat** in `k` — it
costs the same `β=2^(n+1)` no matter how many X/Y factors an observable has. At `n=7`, the two
cross over at `k*=3`: for `k<3` binomial wins (by a lot, at `k=1`), for `k>3` it loses (by a
*lot more* — `47x` worse at `k=4`). This run's observable generator draws `k` **uniformly from
`{1,2,3,4}`** (`--l 4`), so half of the observables it evaluates already sit past the crossover,
and one quarter of them (`k=4`) sit *deep* past it. Averaged over that mix, `seeqst_binomial`'s
per-shot cost comes out worse than `seeqst_uniform`'s constant cost, which is exactly what the
RMSE curves show. See `experiments/exp_6/exp_6.md`'s "Result headline" — this is the same
crossover phenomenon documented there for the `n=6` sweep; this run just happens to sit further
past it, at `n=7`.

## The mechanism: `β(k)` for each ensemble

Every ensemble's cost per shot is its `inverse_weight` `β(P)`, satisfying `M⁻¹(P)=β(P)·P`. This
run's observables are always full weight `n` (every qubit X, Y, or Z; see
`common/pauli_utils.random_bounded_xy_spec`), so what varies from one observable to the next is
only `k`, the number of X/Y factors:

- `pauli`: `β=3^n` — constant in `k` (total Pauli weight is always `n` here, never less).
- `clifford`: `β=2^n+1` — constant in `k`.
- `seeqst_uniform`: `β=2^(n+1)` for *any* non-identity spec — **constant in `k`**. (The flat
  ensemble samples every qubit subset with equal probability `1/2^n`; this is exactly
  Binomial(n, q=0.5) in seeqst_binomial's own family, and `q=0.5` is precisely the value that
  makes its `q^{-k}(1-q)^{-(n-k)}` factor collapse to the `k`-independent constant `2^n`.)
- `seeqst_binomial(q)`: `β=2·q^{-k}·(1-q)^{-(n-k)}` — **exponential in `k`**, minimized at
  `k≈n·q`. With `q=1/(n+1)`, that minimum sits right at `k≈1`: this ensemble is a "bet" that the
  observable is very sparse in X/Y content, and the further `k` strays from that bet, the more it
  costs — geometrically, since each extra unit of `k` multiplies the cost by `(1-q)/q`, which is
  `7×` per step at `q=1/8`.
- `seeqst_unifsize_l(l)`: `β=2(l+1)C(n,k)` for `k≤l` — grows with `k` too, but only
  *polynomially* (via `C(n,k)`), not geometrically, so it degrades far more gently.

Plugging in this run's actual `n=7, l=4, q=1/8`:

| `k` | `pauli` | `clifford` | `seeqst_uniform` | `seeqst_binomial` | `seeqst_unifsize_l` | binomial / uniform |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2187 | 129 | 256 | **35.7** | 70.0 | 0.14× |
| 2 | 2187 | 129 | 256 | 249.6 | 210.0 | 0.97× |
| 3 | 2187 | 129 | 256 | 1746.9 | 350.0 | 6.82× |
| 4 | 2187 | 129 | 256 | **12228.3** | 350.0 | 47.77× |

(computed directly from the actual `inverse_weight` methods, not hand-derived — see the snippet at
the bottom of this file to reproduce.) See `results/exp_6/v0/beta_vs_k_n7_l4.png` for this as a
plot: a flat line for `seeqst_uniform` against `seeqst_binomial`'s line rocketing upward through it
at `k*=3`.

`seeqst_binomial` beats every other ensemble at `k=1` (`35.7`, versus `70` for unifsize, `129` for
Clifford, `256` for uniform) — it is a genuinely excellent choice *if you already know your
observables are that sparse*. The problem is entirely that this run's observables aren't only
`k=1`.

## Why the aggregate flips: this run's observables span `k=1..4`, not just `k=1`

`--l 4` means the observable generator draws `k` **uniformly from `{1,2,3,4}`**
(`pauli_utils.random_bounded_xy_spec`) — so a quarter of the observables evaluated in every
`obs_repeat` draw have `k=4`, right where `seeqst_binomial` is `48×` worse than `seeqst_uniform`.
Averaging `β` over that `k~Uniform{1,...,4}` mix (the quantity that governs the *aggregate* RMSE,
since every observable contributes `Var[ô^(1)]/N_sample ≈ β_k/N_sample` to the mean-squared
error):

```
mean_k beta_binomial (k=1..4)  =  (35.7 + 249.6 + 1746.9 + 12228.3) / 4  =  3565.1
beta_uniform (constant)                                                 =    256.0
ratio                                                                    =   13.93x
```

A single `k=4` observable's enormous cost (`12228`) dominates this average almost by itself —
it alone is worth `86%` of the sum. Since RMSE scales like `sqrt(mean beta / N_sample)` to
leading order, this `13.9x` cost ratio predicts an RMSE ratio around `sqrt(13.93) ≈ 3.7x` —
which is exactly what `results/exp_6/v0/summary.csv` shows once `N_sample` is large enough to
reach that scaling regime (see next section).

## Cross-check against the actual measured data

`results/exp_6/v0/summary.csv`, `haar_random`, `rmse_mean` (± `rmse_std` across the
`3 state_repeats × 100 obs_repeats`):

| `N_sample` | `seeqst_binomial` | `seeqst_uniform` | ratio |
|---:|---:|---:|---:|
| 10 | 4.19 ± 5.52 | 4.38 ± 1.63 | 0.96× |
| 100 | 3.86 ± 4.21 | 1.50 ± 0.16 | 2.57× |
| 500 | 2.56 ± 1.45 | 0.72 ± 0.06 | 3.57× |
| 1000 | 2.07 ± 0.76 | 0.51 ± 0.05 | 4.07× |
| 5000 | 0.89 ± 0.18 | 0.22 ± 0.02 | 4.00× |
| 10000 | 0.64 ± 0.13 | 0.15 ± 0.01 | 4.26× |
| 25000 | 0.38 ± 0.07 | 0.097 ± 0.007 | 3.93× |

By `N_sample=25000` the ratio (`3.93x`) lands almost exactly on the `sqrt(13.93)≈3.73x` predicted
above. At small `N_sample` the ratio is much smaller (even `<1` at `N_sample=10`) — that's the
`k=4` sub-population *not yet converged*: with `beta=12228`, a single-shot estimate for a `k=4`
observable is `12228·⟨ψ|P|ψ⟩`, i.e. individual "hit" shots contribute values in the thousands, and
you need `N_sample` on the order of `beta≈12228` before the running mean starts behaving like a
well-averaged (CLT) quantity rather than being dominated by whether a rare hit happened at all in
this particular batch. `N_sample=25000` is only about `2×` that — this run is *just* reaching the
asymptotic regime at its largest checkpoint, not comfortably inside it.

The `rmse_std` column is itself direct evidence of this: `seeqst_binomial`'s std is *larger than
its own mean* at `N_sample=10` (`5.52` vs `4.19`) and stays proportionally far noisier than
`seeqst_uniform`'s at every checkpoint (coefficient of variation `std/mean` ≈ `1.1` at
`N_sample=100` for binomial vs `≈0.11` for uniform; still `≈0.17` vs `≈0.07` even at
`N_sample=25000`). A heavy-tailed cost distribution across the `k=1..4` mix is exactly what
produces this: most `state_repeat × obs_repeat` draws see few or no `k=4` "hits" and look fine,
occasional ones see a `k=4` hit and swing wildly, and it takes a lot of repeats/shots for that to
average out. `max_error` shows the same signature even more sharply — at `N_sample=25000`,
`seeqst_binomial`'s `max_error_mean` is `6.2x` worse than `seeqst_uniform`'s (`1.66` vs `0.27`),
a bigger gap than RMSE's `3.9x`, because `max_error` is far more sensitive to a few extreme
per-observable errors than a root-mean-square is.

## Consistent with the earlier `n=6` sweep, not a new/different effect

The same computation at `n=6, q=1/7` (the ensembles used in `experiments/exp_6/exp_6.md`'s
`l∈{1,2,4,6}` runs) gives crossover `k*=2` (vs. `k*=3` here at `n=7`) and:

| `l` | mean `β_binomial` (k=1..l) | `β_uniform` | ratio | predicted RMSE ratio `sqrt(·)` | **actual** measured RMSE ratio (haar_random, N=3000) |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.3 | 128 | 0.24× | 0.49× | 0.68× (binomial *better*, matches `exp_6.md`) |
| 2 | 105.9 | 128 | 0.83× | 0.91× | ≈1.04× (roughly tied, matches `exp_6.md`) |
| 4 | 1959.3 | 128 | 15.31× | 3.91× | 2.78× |
| 6 | 47058.6 | 128 | 367.65× | 19.17× | 5.98× |

(the last two rows undershoot the naive `sqrt(mean-beta-ratio)` prediction because, as above,
`N_sample=3000` is nowhere near the `k=6` sub-population's `beta≈235000` — even further from the
asymptotic regime than this `v0` run's `k=4` case). The direction and the qualitative story are
identical at both `n`: `exp_6.md`'s own headline already says `seeqst_binomial`'s "degradation is
the steeper of the two [SEEQST-tuned ensembles], consistent with its `q^{-k}` closed form,
geometric in `k`" and that it "become[s] the worst of all five ensembles by `l=6`" — `l=4` at
`n=7` is just a point further along that same, already-documented curve. `q=1/(n+1)` is a
sparse-observable bet; this run's `l=4` (`k` up to `4`, i.e. weight fraction up to `4/7≈0.57`)
asks it a question it wasn't tuned to answer.

## If you want `seeqst_binomial` to look good at `l=4`

Use a larger `q`. The crossover happens because `q=1/(n+1)` targets a *mean* X/Y-count of
`n·q≈1`, but `k~Uniform{1,...,l}` has mean `(l+1)/2≈2.5` for `l=4` — a poor match. Retuning to
`q≈(l+1)/(2n)` (so the binomial distribution's mean count matches the observable generator's mean
`k`) would recenter the crossover inside the `l=1..4` range instead of near its low end, and is a
natural `--q-binomial` value to try for a direct comparison.

## Reproducing the `beta(k)` table

```python
from ensembles.seeqst_ensemble import SEEQSTEnsemble
from ensembles.seeqst_binomial_ensemble import SEEQSTBinomialEnsemble

n, l = 7, 4
q = 1.0 / (n + 1)
uni, binom = SEEQSTEnsemble(), SEEQSTBinomialEnsemble(q)
spec = lambda k: {i: ('X' if i < k else 'Z') for i in range(n)}
for k in range(1, l + 1):
    print(k, uni.inverse_weight(spec(k), n), binom.inverse_weight(spec(k), n))
```
