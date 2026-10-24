"""Coordination-Predictive Pretraining (CPP): the self-supervised core of CORAL.

Task: from an account pair's HISTORY alone, predict whether the two accounts
will be co-active (synchronized) in the next window. No fraud labels are used.
Positives are observed coordination edges in window t; negatives are random
pairs of accounts active in the same window.

At scoring time an observed coordination edge with low predicted probability
is a *coordination surprise*: two accounts moving together in a way their past
does not explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier

from .windows import Window, pair_key

STRUCT = ["hist_direct", "hist_coord", "cn", "aa", "deg_min", "deg_max",
          "age_min", "age_max", "histn_min", "histn_max", "new_both", "new_any"]


def _binary(A: sp.csr_matrix) -> sp.csr_matrix:
    B = A.copy()
    B.data = np.ones_like(B.data)
    return B


def struct_features(w: Window, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """History-only structural features of pairs (I[k], J[k]) active in window w."""
    N = w.hist_A.shape[0]
    keys = pair_key(I, J, N)
    if "Abin" not in w.extras:
        B = _binary(w.hist_A)
        deg = np.asarray(B.sum(1)).ravel()
        w.extras["Abin"] = B
        w.extras["deg"] = deg
        w.extras["inv_log_deg"] = 1.0 / np.log(np.maximum(deg, 2))
    B, deg, ild = w.extras["Abin"], w.extras["deg"], w.extras["inv_log_deg"]
    common = B[I].multiply(B[J])
    cn = np.asarray(common.sum(1)).ravel()
    aa = np.asarray(common @ ild).ravel()
    hd = np.log1p(w.hist_pairs.reindex(keys).fillna(0).to_numpy(float))
    hc = np.log1p(w.hist_coord.reindex(keys).fillna(0).to_numpy(float))
    pos_i = np.searchsorted(w.accounts, I)
    pos_j = np.searchsorted(w.accounts, J)
    age = w.X["age_days"].to_numpy()
    hn = w.X["hist_n"].to_numpy()
    ai, aj = age[pos_i], age[pos_j]
    di, dj = np.log1p(deg[I]), np.log1p(deg[J])
    ni, nj = hn[pos_i], hn[pos_j]
    new_i, new_j = (ni == 0).astype(float), (nj == 0).astype(float)
    return np.column_stack([hd, hc, np.log1p(cn), aa, np.minimum(di, dj), np.maximum(di, dj),
                            np.minimum(ai, aj), np.maximum(ai, aj), np.minimum(ni, nj),
                            np.maximum(ni, nj), new_i * new_j, np.maximum(new_i, new_j)])


def pair_features(w: Window, I: np.ndarray, J: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Structural history features + symmetric embedding interactions."""
    pos_i = np.searchsorted(w.accounts, I)
    pos_j = np.searchsorted(w.accounts, J)
    hi, hj = H[pos_i], H[pos_j]
    return np.hstack([struct_features(w, I, J), np.abs(hi - hj), hi * hj])


def sample_negatives(w: Window, n: int, rng: np.random.Generator):
    a = w.accounts
    if len(a) < 2 or n <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    I = rng.choice(a, size=n)
    J = rng.choice(a, size=n)
    keep = I != J
    I, J = np.minimum(I[keep], J[keep]), np.maximum(I[keep], J[keep])
    pos = set(w.coord["key"].to_numpy().tolist()) if len(w.coord) else set()
    k = pair_key(I, J, w.hist_A.shape[0])
    m = np.fromiter((x not in pos for x in k), bool, count=len(k))
    return I[m], J[m]


class CoordinationPredictor:
    """p(i and j co-active in window t | history before t)."""

    def __init__(self, encoder, neg_ratio: float = 1.0, max_rows: int = 400_000, seed: int = 0):
        self.encoder = encoder
        self.neg_ratio = neg_ratio
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=seed)

    def dataset(self, windows: list[Window]):
        Xs, ys = [], []
        for w in windows:
            if len(w.coord) == 0:
                continue
            H = self.encoder.embed(w)
            I, J = w.coord["i"].to_numpy(), w.coord["j"].to_numpy()
            nI, nJ = sample_negatives(w, int(len(I) * self.neg_ratio) + 10, self.rng)
            Xs.append(pair_features(w, I, J, H)); ys.append(np.ones(len(I)))
            Xs.append(pair_features(w, nI, nJ, H)); ys.append(np.zeros(len(nI)))
        X, y = np.vstack(Xs), np.concatenate(ys)
        if len(y) > self.max_rows:
            sel = self.rng.choice(len(y), self.max_rows, replace=False)
            X, y = X[sel], y[sel]
        return X, y

    def fit(self, windows: list[Window]):
        X, y = self.dataset(windows)
        self.clf.fit(X, y)
        return self

    def predict_edges(self, w: Window) -> np.ndarray:
        if len(w.coord) == 0:
            return np.empty(0)
        H = self.encoder.embed(w)
        F = pair_features(w, w.coord["i"].to_numpy(), w.coord["j"].to_numpy(), H)
        return self.clf.predict_proba(F)[:, 1]

    def window_auc(self, w: Window, n_neg: int | None = None) -> float:
        """Label-free self-check: how well does the predictor explain this
        window's coordination? Used by the reliability gate."""
        from sklearn.metrics import roc_auc_score
        if len(w.coord) < 5:
            return np.nan
        p_pos = self.predict_edges(w)
        nI, nJ = sample_negatives(w, n_neg or len(p_pos), self.rng)
        if len(nI) < 5:
            return np.nan
        H = self.encoder.embed(w)
        p_neg = self.clf.predict_proba(pair_features(w, nI, nJ, H))[:, 1]
        y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
        return float(roc_auc_score(y, np.r_[p_pos, p_neg]))
