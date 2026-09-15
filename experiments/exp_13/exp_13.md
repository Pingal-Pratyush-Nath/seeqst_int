# exp_13 — long-range Kitaev chain ground-state energy from shadows

## Why this exists

exp_12 asked whether SEEQST/Clifford beat Pauli at recovering a real
Hamiltonian's ground-state energy, and found no — `demo_integrals` is a
nearest-neighbour-only lattice model whose Pauli terms never exceed weight
2, exactly the regime where local random-Pauli shadows are already
near-optimal. exp_13 targets a Hamiltonian designed to have the opposite
property: a long-range-coupled model whose terms genuinely reach weight
`n`, via `common/jw_hamiltonian.py::jw_long_range_kitaev` — the
open-boundary long-range p-wave pairing Kitaev chain (Vodola et al., PRL
113, 156402 (2014)):

```
H = -hopping * sum_j (a_j^dagger a_{j+1} + h.c.)
    - mu * sum_j (n_j - 1/2)
    + (delta/2) * sum_{j<k} (a_j a_k + a_k^dagger a_j^dagger) / (k-j)^alpha
```

Under Jordan-Wigner, a pairing term spanning sites `j` and `k` picks up a
Z-string over every site strictly between them, so its Pauli weight is
`k-j+1` — genuinely up to `n`, with no artificial padding needed.

## The parameter-regime caveat (important)

With this model's own naturally-quoted physics defaults
(`hopping=mu=delta=1, alpha=1`), **Pauli still wins overall.** I checked
this numerically before writing exp_13: even though weight-up-to-6 terms
exist at `n=6`, the O(1) hopping/onsite/short-range-pairing terms (weight
≤ 3) carry essentially all of the coefficient-squared mass (>99.5% at
`alpha=1`), and Pauli is cheap exactly where that mass sits. The rare
high-weight terms are too weak to swing the aggregate.

To make the long-range physics actually dominate, exp_13's **defaults
differ from the model's own physical defaults**: `hopping=0`, `mu=0`,
`delta=1`, `alpha=0` (fully uniform, all-to-all pairing, no decay, no
competing local terms), `n=10`. At those settings (checked numerically):

- SEEQST at `q=1/(n+1)` beats Pauli by roughly **5.7x** in the
  `sum c_i^2 beta(P_i)` proxy.
- Clifford beats Pauli by roughly **2.9x**.
- An aggressively high-weight-tuned `q` (close to 1) is dramatically
  *worse* than the generic `q=1/(n+1)` here — `q=0.8` was over 400x worse
  than Pauli at this same `n=10, alpha=0`. This Hamiltonian's terms are
  spread across many different weights at once, which is exactly the
  regime `q=1/(n+1)` (Equation 7 / Example 2 of `SEEQST_shadows4.pdf`) is
  meant for, not an aggressively high-weight-biased choice.

Override `--hopping/--mu/--delta/--alpha` to explore other regimes,
including the model's own "natural" physics defaults (where Pauli wins
again) — see the CLI's docstring for example commands.

## Ensembles and q-values

Same roster as exp_12: `pauli`, `clifford`, `seeqst_uniform`, plus one
`SEEQSTBinomialEnsemble` per `--q-values` entry. Default `--q-values` (when
omitted) is `[1/(n+1), 0.5]` — computed from whatever `--n` you pass, since
the "generic" untuned value depends on `n`.

**`--tune-q` (on by default)** adds one more ensemble, `seeqst_binomial_tuned`,
whose `q` is computed directly from the actual Hamiltonian via
`jw_hamiltonian.optimal_q` — `argmin_q sum_i coeffs[i]^2 * beta_q(specs[i])`,
a dependency-free grid search (no scipy needed). This is a real "tune q to
the problem" recipe, not a guess: for this chain's default regime
(`hopping=mu=0, delta=1, alpha=0`), every pairing term has X/Y-weight
exactly 2 regardless of separation (only the incidental Z-string length
varies), so the optimum comes out to exactly `q=0.2=2/n` — the textbook
`q=m/n` rule, recovered automatically because the function recognizes every
term shares one X/Y-weight. This gives roughly a **9.9x** improvement over
Pauli in the variance proxy, versus 5.7x at the generic `q=1/(n+1)` and only
1.4x at `q=0.5`. Pass `--no-tune-q` to skip it.

## Repeats and metric

Identical to exp_12: state (ground state) and observable (H) are both
fixed, so `--num-repeats` (default 20) is the sole source of the error
bars — each repeat draws a fresh `max(n_samples)` shadow snapshots per
ensemble. Metric: `abs_error(N_sample) = |E_hat(N_sample) - E_true|`,
averaged ± std-dev across repeats, plotted log-log vs. `N_sample` in
`energy_error_vs_nsample.png`.

## Usage

```bash
python experiments/exp_13/exp_13.py --quick
python experiments/exp_13/exp_13.py --n 10
python experiments/exp_13/exp_13.py --n 10 --q-values 0.0909090909 0.5 --num-repeats 30
python experiments/exp_13/exp_13.py --n 10 --hopping 1 --mu 1 --alpha 1   # model's own "natural" defaults -- Pauli wins again
python experiments/exp_13/exp_13.py --n 12 --alpha 0.25
```

## Validation

Same limitation as exp_12: qiskit is not installable in either sandbox
this was authored from, so the qiskit-touching pieces could not be run
end-to-end from there. What *was* checked with plain numpy before writing
this experiment:

- `jw_long_range_kitaev`'s coefficients come out exactly real (imaginary
  part exactly 0 to machine precision) for several parameter combinations,
  including the working regime — expected, since the hopping and pairing
  terms are each added together with their own exact Hermitian conjugate;
- the resulting Hamiltonian matrix is exactly Hermitian (`max|H-H^dagger|
  = 0` at `n=6`) with a sensible real ground energy;
- the Pauli-weight-up-to-`n` structure, and the specific `sum c_i^2
  beta(P_i)` proxy numbers quoted above (both the working regime and the
  "model's own defaults still favor Pauli" result), were computed with an
  independent from-scratch numeric reimplementation before this file was
  written.

Please run `--quick` first and sanity-check `E_true` before a full run.
Exact diagonalization (dense `2**n x 2**n`) is fine through roughly
`n~10-12` and impractical much beyond that.
