"""SEEQST classical shadows under the SIZE-CAPPED uniform-subset-size
sampling distribution: pick a size s uniformly from {0, ..., l} (l a fixed
cap, not n), then a subset I uniformly among the size-s subsets --

    p(I) = 1 / ((l+1) * C(n, |I|))      for |I| <= l,   0 otherwise.

At l=n this is exactly the (uncapped) "uniform subset size" construction of
``QIP_notes/SEEQST_shadows_threshold.tex`` Definition 4 / Prop. 6.4 of
``SEEQST_shadows_main_results.pdf``. Capping at l<n is the natural
companion to an observable family that is already known to have X/Y-weight
<= l (see ``common/pauli_utils.random_bounded_xy_spec``, exp_6's
observable generator): it puts zero probability mass on subset sizes that
can never matter for such observables, concentrating what the uncapped
construction would have spent on sizes l+1..n instead.

Inverse map, from the general eigenvalue theorem (Theorem 3.3 / Prop. 4.2
of the main-results note; lambda_{a,b} = (1/2) p(a) for X/Y-support a != 0):
for a spec with xy_weight k = |a|,

    beta(P) = 1                              P = identity
    beta(P) = 2 (l+1) C(n, k)                0 < k <= l
    beta(P) = undefined (ValueError)         k > l   -- p(a)=0 there: the
                                              channel is SINGULAR on that
                                              subspace, there is no unbiased
                                              estimator, not merely a
                                              large-variance one.

The pure-Z-type case (a=0) is NOT implemented: its eigenvalue needs
p_hat(b), the Walsh-Hadamard transform of THIS capped distribution, which
-- unlike the product-Bernoulli case in ``seeqst_binomial_ensemble.py`` --
does not collapse to a clean closed form (it is a sum, over s=0..l, of
Krawtchouk polynomials K_s(|b|; n), not a single power). This branch is
unreachable from exp_6 (``common.pauli_utils.random_bounded_xy_spec`` with
its default ``min_xy=1`` never produces a pure-Z spec), so deriving it is
out of scope here; calling ``inverse_weight`` on a pure-Z spec raises
NotImplementedError rather than silently returning a wrong number.

Sampling reuses ``ensembles._seeqst_circuits`` (see that module's docstring
for the real-qubit-subset <-> circuit-index convention).
"""

from __future__ import annotations

from math import comb

import numpy as np
from qiskit.quantum_info import Statevector

from common.pauli_utils import is_pure_z_type, xy_weight
from ensembles import _seeqst_circuits as _circuits
from ensembles.base import ShadowEnsemble, Snapshot


class SEEQSTUniformSizeEnsemble(ShadowEnsemble):
    name = "seeqst_unifsize"

    def __init__(self, l: int) -> None:
        if l < 0:
            raise ValueError(f"l={l} must be >= 0")
        self.l = int(l)

    def _check_l(self, n: int) -> None:
        if self.l > n:
            raise ValueError(
                f"l={self.l} must be <= n={n}: there is no uniform subset of "
                "size s for s>n, so this ensemble's sampling distribution is "
                "not well-defined for l>n."
            )

    def sample_snapshot(self, state: Statevector, n: int, rng: np.random.Generator) -> Snapshot:
        self._check_l(n)
        s = int(rng.integers(0, self.l + 1))  # size ~ Uniform{0, ..., l}
        subset_qubits = rng.choice(n, size=s, replace=False)
        block = _circuits.qubit_subset_to_block(subset_qubits, n)
        branch = int(rng.integers(0, _circuits.num_branches(n, block)))
        qc = _circuits.circuit_for(n, block, branch)
        rotated = state.evolve(qc)
        outcome, _ = rotated.measure()
        psi_pre = Statevector.from_label(str(outcome)).evolve(qc.inverse())
        return Snapshot(psi_pre=psi_pre)

    def inverse_weight(self, spec: dict, n: int) -> float:
        if not spec:
            return 1.0
        if is_pure_z_type(spec):
            raise NotImplementedError(
                "SEEQSTUniformSizeEnsemble.inverse_weight has no closed form on "
                "the pure-Z sector (see module docstring) and is not needed by "
                "exp_6's observable family (random_bounded_xy_spec always has "
                "xy_weight >= 1)."
            )
        k = xy_weight(spec)
        if k > self.l:
            raise ValueError(
                f"spec has xy_weight k={k} > l={self.l}: p(a)=0 for this "
                "ensemble, so the inverse channel is singular here -- there is "
                "no unbiased single-shot estimator for this observable under "
                "this ensemble."
            )
        return 2.0 * (self.l + 1) * comb(n, k)
