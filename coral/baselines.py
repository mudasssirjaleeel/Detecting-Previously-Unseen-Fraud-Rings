"""Baselines, all trained on exactly the same causal windows as CORAL.

Supervised baselines only ever see labels of the typologies present in training
(the held-out typology is removed from the training data entirely)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)

from .encoders import SIGNEncoder
from .windows import NODE_FEATURES, Window


def _xy(windows: list[Window], emb: SIGNEncoder | None = None):
    X = []
    for w in windows:
        f = w.X[NODE_FEATURES].to_numpy()
        if emb is not None:
            f = np.hstack([f, emb.embed(w)])
        X.append(f)
    return np.vstack(X), np.concatenate([w.y for w in windows])


MAX_NEG = None  # set e.g. 300_000 for large ledgers: keep all positives, subsample negatives


def subsample_negatives(X, y, max_neg, seed=0):
    """Keep every positive and at most max_neg random negatives (rank metrics are
    unaffected by the class-prior shift this introduces)."""
    if max_neg is None or (y == 0).sum() <= max_neg:
        return X, y
    rng = np.random.default_rng(seed)
    neg = np.where(y == 0)[0]
    keep = np.r_[np.where(y == 1)[0], rng.choice(neg, max_neg, replace=False)]
    keep.sort()
    return X[keep], y[keep]


class _Base:
    graph = False

    def fit(self, windows):
        if self.graph:
            self.emb = SIGNEncoder().fit(windows)
        X, y = _xy(windows, getattr(self, "emb", None))
        X, y = subsample_negatives(X, y, MAX_NEG)
        self._fit(X, y)
        return self

    def score(self, w: Window) -> np.ndarray:
        X, _ = _xy([w], getattr(self, "emb", None))
        return self._score(X)


class RandomForest(_Base):
    name = "RF"

    def _fit(self, X, y):
        self.m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=2,
                                        class_weight="balanced_subsample", random_state=0)
        self.m.fit(X, y)

    def _score(self, X):
        return self.m.predict_proba(X)[:, 1]


class GradientBoosting(_Base):
    """XGBoost when installed, otherwise scikit-learn's histogram GBDT (same family)."""
    name = "GBDT"

    def _fit(self, X, y):
        try:
            import xgboost as xgb
            pos = max(1, y.sum())
            self.m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                       subsample=.8, colsample_bytree=.8,
                                       scale_pos_weight=(len(y) - pos) / pos, n_jobs=2,
                                       eval_metric="aucpr", random_state=0)
            self.backend = "xgboost"
        except ImportError:
            self.backend = "sklearn-hgb"
            self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                    class_weight="balanced", random_state=0)
        self.m.fit(X, y)

    def _score(self, X):
        return self.m.predict_proba(X)[:, 1]


class GraphGBDT(GradientBoosting):
    """Supervised graph baseline: node features + SIGN graph embedding -> GBDT.
    Stands in for a supervised GNN when PyTorch is unavailable (see
    torch_encoder.SupervisedTemporalGNN for the neural version)."""
    name = "Graph-GBDT"
    graph = True


class IForest(_Base):
    name = "IsolationForest"

    def _fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.m = IsolationForest(n_estimators=300, random_state=0, n_jobs=2)
        self.m.fit((X - self.mu) / self.sd)

    def _score(self, X):
        return -self.m.decision_function((X - self.mu) / self.sd)


ALL = [RandomForest, GradientBoosting, GraphGBDT, IForest]
