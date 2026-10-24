"""CORAL: Coordination-aware, Reliability-gated Anomaly Learning.

    1. Coordination-Predictive Pretraining  (pairs.CoordinationPredictor, label-free)
    2. Coordination surprise per edge / node
    3. Ring-level scoring by closure of surprising edges (rings.closure_scores)
    4. Drift-aware reliability gate between the graph score and a feature-only
       detector, driven by label-free signals of graph unreliability
    5. Optional supervised head (CORAL-S) stacked on top, trained with cross-fitted
       surprise scores so training-window scores are not optimistically biased.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .encoders import SIGNEncoder
from .pairs import CoordinationPredictor
from .rings import RING_DESC, closure_scores
from .windows import NODE_FEATURES, Window, WindowBuilder


@dataclass
class CORALConfig:
    encoder: str = "sign"         # "sign" (SIGN + GBDT pair head), "gnn" (learned temporal GNN, NumPy) or "torch"
    neg_ratio: float = 1.0
    max_pair_rows: int = 400_000
    eps: float = 1e-3
    ring_pmax: float = 0.5
    ring_min_size: int = 3
    alpha0: float = 0.9           # prior weight on the graph (flow-ring) score; chosen on the dev world
    use_flow: bool = True         # weight rings by exclusivity, conservation and amount coherence
    use_amount_cv: bool = True    # include the amount-coherence factor (generator-sensitive; ablated)
    gate_lambda: float = 0.5      # how fast the gate backs off when the graph looks unreliable
    gate_r0: float = 2.0          # dead zone: react only to shifts beyond ~2 s.d. of validation variation
    use_gate: bool = True
    use_rings: bool = True
    seed: int = 0


def _pct(x: np.ndarray) -> np.ndarray:
    """Percentile rank within a window (ties averaged), in [0, 1]."""
    if len(x) == 0:
        return x
    return pd.Series(x).rank(method="average", pct=True).to_numpy()


def fisher(*ps) -> np.ndarray:
    """Fisher-style combination of within-window percentiles: sum of -log(1 - p).
    Emphasises the upper tail, where rare positives live."""
    return sum(-np.log(np.clip(1.0 - p, 1e-6, 1.0)) for p in ps)


def _make_encoder(cfg: CORALConfig):
    if cfg.encoder == "sign":
        return SIGNEncoder()
    if cfg.encoder == "torch":
        from .torch_encoder import TorchCoordinationModel  # optional dependency
        return TorchCoordinationModel(seed=cfg.seed)
    if cfg.encoder == "gnn":
        from .np_gnn import NumpyTemporalGNN
        return NumpyTemporalGNN(seed=cfg.seed)
    raise ValueError(cfg.encoder)


class CORAL:
    def __init__(self, cfg: CORALConfig | None = None):
        self.cfg = cfg or CORALConfig()

    # ------------------------------------------------------------------ fit
    def fit(self, builder: WindowBuilder, train_t: list[int], val_t: list[int]):
        cfg = self.cfg
        self.builder = builder
        tr = [builder.build(t) for t in train_t]
        va = [builder.build(t) for t in val_t]
        self.encoder = _make_encoder(cfg)
        if cfg.encoder in ("torch", "gnn"):
            # learned encoder + pair head trained jointly on the CPP objective
            self.encoder.fit(tr)
            self.cpp = self.encoder
        else:
            self.encoder.fit(tr)
            self.cpp = CoordinationPredictor(self.encoder, cfg.neg_ratio, cfg.max_pair_rows,
                                             cfg.seed).fit(tr)

        # feature-only detector (label-free)
        Xtr = np.vstack([w.X[NODE_FEATURES].to_numpy() for w in tr])
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        self.iforest = IsolationForest(n_estimators=300, random_state=cfg.seed, n_jobs=2)
        self.iforest.fit((Xtr - self.mu) / self.sd)

        # reliability-gate references
        self.train_logdeg = np.concatenate([self._coord_logdeg(w) for w in tr])
        self.train_emb_mu = np.vstack([self.encoder.embed(w) for w in tr]).mean(0)
        self.train_emb_sd = np.vstack([self.encoder.embed(w) for w in tr]).std(0) + 1e-6
        comps = np.array([self._gate_components(w) for w in va])
        self.gate_mu = np.nanmean(comps, 0)
        self.gate_sd = np.nanstd(comps, 0) + 1e-3
        return self

    # ------------------------------------------------------------ gate bits
    @staticmethod
    def _coord_logdeg(w: Window) -> np.ndarray:
        if len(w.coord) == 0:
            return np.zeros(1)
        d = pd.concat([w.coord["i"], w.coord["j"]]).value_counts()
        return np.log1p(d.to_numpy(float))

    def _gate_components(self, w: Window) -> np.ndarray:
        auc = self.cpp.window_auc(w)
        ks = ks_2samp(self._coord_logdeg(w), self.train_logdeg).statistic
        H = self.encoder.embed(w)
        drift = float(np.mean(np.abs(H.mean(0) - self.train_emb_mu) / self.train_emb_sd))
        # sign convention: larger = graph less trustworthy
        return np.array([-auc if not np.isnan(auc) else np.nan, ks, drift])

    def reliability(self, w: Window) -> tuple[float, float]:
        z = (self._gate_components(w) - self.gate_mu) / self.gate_sd
        r = float(np.nanmean(z)) if np.any(~np.isnan(z)) else 0.0
        alpha = self.cfg.alpha0 * float(np.exp(-self.cfg.gate_lambda * max(0.0, r - self.cfg.gate_r0)))
        return r, alpha

    # ---------------------------------------------------------------- score
    def graph_scores(self, w: Window, cpp=None):
        cfg = self.cfg
        cpp = cpp or self.cpp
        n = len(w.accounts)
        node_s = np.zeros(n); node_max = np.zeros(n); tri = np.zeros(n); ring = np.zeros(n)
        cdeg = np.zeros(n); desc = np.zeros((n, len(RING_DESC)))
        rings = []
        if len(w.coord):
            p = cpp.predict_edges(w)
            aa = w.coord["aa"].to_numpy(float)
            s = -np.log(p + cfg.eps)
            e = pd.DataFrame({"i": w.coord["i"].to_numpy(), "j": w.coord["j"].to_numpy(),
                              "s": s, "p": p, "aa": aa})
            e["sv"] = e["s"] * e["aa"]      # surprise weighted by edge specificity
            both = pd.concat([e[["i", "sv"]].rename(columns={"i": "a"}),
                              e[["j", "sv"]].rename(columns={"j": "a"})])
            g = both.groupby("a")["sv"]
            node_s = g.sum().reindex(w.accounts).fillna(0).to_numpy()
            node_max = g.max().reindex(w.accounts).fillna(0).to_numpy()
            cdeg = g.size().reindex(w.accounts).fillna(0).to_numpy(float)
            if cfg.use_rings:
                tri, ring, rings, desc = closure_scores(e, w.accounts, w.win_tx, cfg.ring_pmax,
                                                        cfg.ring_min_size, return_desc=True)
        D = dict(zip(RING_DESC, desc.T))
        flow = ring.copy()
        if cfg.use_flow:
            flow = flow * (0.5 + D["ring_exclusivity"]) * (0.5 + D["ring_conservation"])
            if cfg.use_amount_cv:
                flow = flow / (0.5 + np.minimum(D["ring_amount_cv"], 2.0))
        if cfg.use_rings:
            # graph evidence: flow-weighted rings, ties broken by surprise closure
            graph = _pct(_pct(flow) + 0.1 * _pct(tri))
        else:
            graph = _pct(node_s)
        return dict(node_surprise=node_s, max_surprise=node_max, triangle=tri, ring=ring,
                    flow_ring=flow, coord_deg=cdeg, graph=graph, **D), rings

    def feature_score(self, w: Window) -> np.ndarray:
        X = (w.X[NODE_FEATURES].to_numpy() - self.mu) / self.sd
        return -self.iforest.decision_function(X)

    def score(self, w: Window) -> tuple[pd.DataFrame, list[dict]]:
        gs, rings = self.graph_scores(w)
        feat = self.feature_score(w)
        r, alpha = self.reliability(w) if self.cfg.use_gate else (0.0, self.cfg.alpha0)
        out = pd.DataFrame({"account": w.accounts, **gs, "feature": feat})
        out["reliability"] = r
        out["alpha"] = alpha
        # The gate trades graph evidence against graph-FREE evidence, so that when the
        # graph becomes unreliable the score falls back to something it cannot corrupt.
        out["coral"] = alpha * gs["graph"] + (1 - alpha) * _pct(feat)
        return out, rings


class CORALSupervised:
    """CORAL-S: a gradient-boosted head on [node features + CORAL graph signals].
    Training-window graph signals are cross-fitted (2 folds by window parity) so
    the head never sees in-sample, optimistically low surprise values."""

    GRAPH_COLS = ["node_surprise", "max_surprise", "triangle", "ring", "flow_ring", "coord_deg",
                  "graph"] + RING_DESC

    def __init__(self, coral: CORAL, seed: int = 0, cpp_override=None):
        self.coral = coral
        self.seed = seed
        self.cpp_override = cpp_override  # ablation: e.g. a heuristic predictor instead of CPP

    def _rows(self, w: Window, cpp):
        gs, _ = self.coral.graph_scores(w, cpp=cpp)
        G = np.column_stack([gs[c] for c in self.GRAPH_COLS])
        return np.hstack([w.X[NODE_FEATURES].to_numpy(), G])

    def fit(self, builder: WindowBuilder, train_t: list[int]):
        cfg = self.coral.cfg
        folds = [[t for t in train_t if t % 2 == k] for k in (0, 1)]
        Xs, ys = [], []
        for k in (0, 1):
            fit_w = [builder.build(t) for t in folds[1 - k]]
            if self.cpp_override is not None:
                cpp = self.cpp_override
            elif cfg.encoder == "sign":
                cpp = CoordinationPredictor(self.coral.encoder, cfg.neg_ratio, cfg.max_pair_rows,
                                            cfg.seed + 1 + k).fit(fit_w)
            elif cfg.encoder == "gnn":
                from .np_gnn import NumpyTemporalGNN
                cpp = NumpyTemporalGNN(seed=cfg.seed + 1 + k).fit(fit_w)   # cross-fitted too
            else:
                cpp = self.coral.cpp  # torch: reuse (documented limitation)
            for t in folds[k]:
                w = builder.build(t)
                Xs.append(self._rows(w, cpp)); ys.append(w.y)
        X, y = np.vstack(Xs), np.concatenate(ys)
        from .baselines import MAX_NEG, subsample_negatives
        X, y = subsample_negatives(X, y, MAX_NEG, self.seed)
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                  class_weight="balanced", random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def score(self, w: Window) -> np.ndarray:
        return self.clf.predict_proba(self._rows(w, self.cpp_override or self.coral.cpp))[:, 1]
