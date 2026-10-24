"""History encoders: map each active account to an embedding computed ONLY from
its history graph (windows before t).

* SIGNEncoder  - parameter-free graph propagation [X, ÂX, Â²X] (Rossi et al., SIGN;
                 Wu et al., SGC). Runs anywhere with NumPy/SciPy; used for smoke tests
                 and as a strong, cheap backbone.
* TorchTemporalEncoder (coral/torch_encoder.py) - a learned temporal message-passing
                 encoder trained end-to-end on the coordination-prediction objective.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .windows import HIST_FEATURES, Window


class SIGNEncoder:
    name = "sign"

    def __init__(self, hops: int = 2):
        self.hops = hops
        self.mu = None
        self.sd = None

    @property
    def dim(self):
        return (len(HIST_FEATURES) + 1) * (self.hops + 1)

    def fit(self, windows: list[Window]):
        rows = [w.hist_X[HIST_FEATURES].to_numpy(float) for w in windows if len(w.hist_X)]
        Xs = np.vstack(rows) if rows else np.zeros((1, len(HIST_FEATURES)))
        self.mu = Xs.mean(0)
        self.sd = Xs.std(0) + 1e-6
        return self

    def embed(self, w: Window) -> np.ndarray:
        """Returns an array aligned with w.accounts."""
        key = f"emb_{id(self)}"
        if key in w.extras:
            return w.extras[key]
        N = w.hist_A.shape[0]
        F = len(HIST_FEATURES)
        Xh = np.zeros((N, F + 1))
        if len(w.hist_X):
            idx = w.hist_X.index.to_numpy(np.int64)
            Xh[idx, :F] = (w.hist_X[HIST_FEATURES].to_numpy(float) - self.mu) / self.sd
            Xh[idx, F] = 1.0  # "seen in history" indicator
        A = w.hist_A + sp.eye(N, format="csr")
        d = np.asarray(A.sum(1)).ravel()
        dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
        Ahat = dinv @ A @ dinv
        feats, cur = [Xh[w.accounts]], Xh
        for _ in range(self.hops):
            cur = Ahat @ cur
            feats.append(cur[w.accounts])
        H = np.hstack(feats)
        w.extras[key] = H
        return H
