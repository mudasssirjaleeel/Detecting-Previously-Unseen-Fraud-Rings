"""Finite-difference check of the NumPy GNN's hand-derived gradients."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from coral.data import SyntheticConfig, generate_synthetic  # noqa: E402
from coral.np_gnn import NumpyTemporalGNN  # noqa: E402
from coral.pairs import sample_negatives, struct_features  # noqa: E402
from coral.windows import HIST_FEATURES, WindowBuilder  # noqa: E402


def test_gradients_match_finite_differences():
    tx = generate_synthetic(SyntheticConfig(n_accounts=300, n_days=10, patterns_per_typology=2, seed=5))
    b = WindowBuilder(tx)
    w = b.build(8)
    m = NumpyTemporalGNN(dim=6, layers=2, hidden=5, seed=1)
    Xs = w.hist_X[HIST_FEATURES].to_numpy(float)
    m.mu, m.sd = Xs.mean(0), Xs.std(0) + 1e-6
    I, J = w.coord["i"].to_numpy()[:40], w.coord["j"].to_numpy()[:40]
    S = struct_features(w, I, J)
    m.s_mu, m.s_sd = S.mean(0), S.std(0) + 1e-6
    m._init(len(HIST_FEATURES) + 1, S.shape[1])
    nI, nJ = sample_negatives(w, 40, np.random.default_rng(0))
    II, JJ = np.r_[I, nI], np.r_[J, nJ]
    y = np.r_[np.ones(len(I)), np.zeros(len(nI))]
    _, G = m._loss_grad(w, II, JJ, y)
    rng = np.random.default_rng(2)
    worst = 0.0
    for k, P in m.P.items():
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + 1e-5; lp, _ = m._loss_grad(w, II, JJ, y)
            P[idx] = old - 1e-5; lm, _ = m._loss_grad(w, II, JJ, y)
            P[idx] = old
            num = (lp - lm) / 2e-5
            ana = G[k][idx]
            rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
            if abs(num) + abs(ana) > 1e-6:
                worst = max(worst, rel)
    assert worst < 1e-3, f"gradient mismatch, worst relative error {worst:.2e}"
    print(f"worst relative gradient error: {worst:.2e}")


if __name__ == "__main__":
    test_gradients_match_finite_differences(); print("PASS test_gradients_match_finite_differences")
