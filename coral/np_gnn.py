"""Learned temporal GNN encoder for CORAL in pure NumPy/SciPy (no PyTorch needed).

Same architecture as coral/torch_encoder.py, with hand-derived backpropagation
(verified against numerical gradients in tests/test_np_gnn.py):

    h0      = ReLU(X W0 + b0)
    h^l     = ReLU(h^{l-1} Ws_l + (Â_cnt h^{l-1}) Wc_l + (Â_rec h^{l-1}) Wr_l + b_l) + h^{l-1}
    z       = h^L
    u_ij    = [ z_i * z_j , |z_i - z_j| , x_ij ]
    logit   = v . ReLU(u_ij U + c) + c2
    loss    = binary cross-entropy (co-active in window t vs random active pair)

Â_cnt / Â_rec are symmetric-normalised history adjacencies with interaction
counts and recency-decayed weights. Each window is encoded on the subgraph of
its active accounts plus their 1-hop history neighbours (keeps cost linear in
activity, so it scales to large ledgers). Trained with Adam, no fraud labels.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .pairs import sample_negatives, struct_features
from .windows import HIST_FEATURES, Window


def _norm_adj(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A + sp.eye(A.shape[0], format="csr")
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    return (dinv @ A @ dinv).tocsr()


class _Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, wd=1e-4):
        self.p, self.lr, self.b1, self.b2, self.wd = params, lr, b1, b2, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class NumpyTemporalGNN:
    name = "np-temporal-gnn"

    def __init__(self, dim=32, layers=2, hidden=32, epochs=20, lr=3e-3, neg_ratio=1.0,
                 max_edges_per_window=20_000, seed=0, verbose=False):
        self.dim, self.L, self.hid = dim, layers, hidden
        self.epochs, self.lr, self.neg_ratio, self.max_e = epochs, lr, neg_ratio, max_edges_per_window
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    # ---------------------------------------------------------------- params
    def _init(self, f_in, n_struct):
        r = self.rng
        def he(a, b):
            return r.normal(0, np.sqrt(2.0 / a), size=(a, b))
        P = {"W0": he(f_in, self.dim), "b0": np.full(self.dim, 0.05)}  # >0: no-history nodes start off the ReLU kink
        for l in range(self.L):
            P[f"Ws{l}"] = he(self.dim, self.dim) * 0.5
            P[f"Wc{l}"] = he(self.dim, self.dim) * 0.5
            P[f"Wr{l}"] = he(self.dim, self.dim) * 0.5
            P[f"bl{l}"] = np.full(self.dim, 0.01)
        P["U"] = he(2 * self.dim + n_struct, self.hid)
        P["c"] = np.zeros(self.hid)
        P["v"] = r.normal(0, np.sqrt(1.0 / self.hid), size=self.hid)
        P["c2"] = np.zeros(1)
        self.P = P

    # --------------------------------------------------------------- graphs
    def _graph(self, w: Window):
        key = f"npgnn_graph_{id(self)}"
        if key in w.extras:
            return w.extras[key]
        A = w.hist_A
        act = w.accounts
        nb = np.unique(A[act].indices) if A.nnz else np.empty(0, np.int64)
        nodes = np.union1d(act, nb).astype(np.int64)
        Ac = _norm_adj(A[nodes][:, nodes])
        Ar = _norm_adj(w.hist_A_rec[nodes][:, nodes])
        F = len(HIST_FEATURES)
        X = np.zeros((len(nodes), F + 1))
        if len(w.hist_X):
            hx = w.hist_X.reindex(nodes)
            seen = ~hx[HIST_FEATURES[0]].isna().to_numpy()
            X[seen, :F] = (hx.loc[seen, HIST_FEATURES].to_numpy(float) - self.mu) / self.sd
            X[seen, F] = 1.0
        g = (nodes, X, Ac, Ar)
        w.extras[key] = g
        return g

    # -------------------------------------------------------------- forward
    def _encode(self, X, Ac, Ar, cache=None):
        P = self.P
        a0 = X @ P["W0"] + P["b0"]
        h = np.maximum(a0, 0)
        if cache is not None:
            cache["a0"], cache["hs"], cache["pre"], cache["Ah"] = a0, [h], [], []
        for l in range(self.L):
            Ach, Arh = Ac @ h, Ar @ h
            pre = h @ P[f"Ws{l}"] + Ach @ P[f"Wc{l}"] + Arh @ P[f"Wr{l}"] + P[f"bl{l}"]
            h = np.maximum(pre, 0) + h
            if cache is not None:
                cache["pre"].append(pre); cache["Ah"].append((Ach, Arh)); cache["hs"].append(h)
        return h

    def _head(self, z, ii, jj, S, cache=None):
        P = self.P
        zi, zj = z[ii], z[jj]
        diff = zi - zj
        u = np.hstack([zi * zj, np.abs(diff), S])
        q = u @ P["U"] + P["c"]
        r = np.maximum(q, 0)
        logit = r @ P["v"] + P["c2"][0]
        if cache is not None:
            cache.update(zi=zi, zj=zj, diff=diff, u=u, q=q, r=r)
        return logit

    # ------------------------------------------------------------- backward
    def _loss_grad(self, w: Window, I, J, y):
        nodes, X, Ac, Ar = self._graph(w)
        ii, jj = np.searchsorted(nodes, I), np.searchsorted(nodes, J)
        S = (struct_features(w, I, J) - self.s_mu) / self.s_sd
        ce, ch = {}, {}
        z = self._encode(X, Ac, Ar, ce)
        logit = self._head(z, ii, jj, S, ch)
        p = 1.0 / (1.0 + np.exp(-logit))
        n = len(y)
        loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        P, G = self.P, {}
        dlog = (p - y) / n
        G["c2"] = np.array([dlog.sum()])
        G["v"] = ch["r"].T @ dlog
        dq = np.outer(dlog, P["v"]) * (ch["q"] > 0)
        G["U"] = ch["u"].T @ dq
        G["c"] = dq.sum(0)
        du = dq @ P["U"].T
        d = self.dim
        dprod, dabs = du[:, :d], du[:, d:2 * d]
        sgn = np.sign(ch["diff"])
        dzi = dprod * ch["zj"] + dabs * sgn
        dzj = dprod * ch["zi"] - dabs * sgn
        dz = np.zeros_like(z)
        np.add.at(dz, ii, dzi)
        np.add.at(dz, jj, dzj)
        dh = dz
        for l in reversed(range(self.L)):
            h_prev = ce["hs"][l]
            Ach, Arh = ce["Ah"][l]
            dpre = dh * (ce["pre"][l] > 0)
            G[f"Ws{l}"] = h_prev.T @ dpre
            G[f"Wc{l}"] = Ach.T @ dpre
            G[f"Wr{l}"] = Arh.T @ dpre
            G[f"bl{l}"] = dpre.sum(0)
            dh = dh + dpre @ P[f"Ws{l}"].T + Ac.T @ (dpre @ P[f"Wc{l}"].T) + Ar.T @ (dpre @ P[f"Wr{l}"].T)
        da0 = dh * (ce["a0"] > 0)
        G["W0"] = X.T @ da0
        G["b0"] = da0.sum(0)
        return loss, G

    # ------------------------------------------------------------------ fit
    def fit(self, windows: list[Window]):
        rows = [w.hist_X[HIST_FEATURES].to_numpy(float) for w in windows if len(w.hist_X)]
        Xs = np.vstack(rows)
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-6
        S = np.vstack([struct_features(w, w.coord["i"].to_numpy(), w.coord["j"].to_numpy())
                       for w in windows if len(w.coord)])
        self.s_mu, self.s_sd = S.mean(0), S.std(0) + 1e-6
        self._init(len(HIST_FEATURES) + 1, S.shape[1])
        opt = _Adam(self.P, lr=self.lr)
        for ep in range(self.epochs):
            tot, cnt = 0.0, 0
            for wi in self.rng.permutation(len(windows)):
                w = windows[wi]
                if len(w.coord) < 5:
                    continue
                I, J = w.coord["i"].to_numpy(), w.coord["j"].to_numpy()
                if len(I) > self.max_e:
                    sel = self.rng.choice(len(I), self.max_e, replace=False)
                    I, J = I[sel], J[sel]
                nI, nJ = sample_negatives(w, int(len(I) * self.neg_ratio) + 10, self.rng)
                y = np.r_[np.ones(len(I)), np.zeros(len(nI))]
                loss, G = self._loss_grad(w, np.r_[I, nI], np.r_[J, nJ], y)
                opt.step(G)
                tot += loss; cnt += 1
            if self.verbose:
                print(f"[CPP-npGNN] epoch {ep + 1}/{self.epochs} loss {tot / max(cnt, 1):.4f}")
        self.final_loss = tot / max(cnt, 1)
        return self

    # ---------------------------------------------------- inference interface
    def _z(self, w):
        key = f"npgnn_z_{id(self)}"
        if key not in w.extras:
            nodes, X, Ac, Ar = self._graph(w)
            w.extras[key] = self._encode(X, Ac, Ar)
        return w.extras[key]

    def embed(self, w: Window) -> np.ndarray:
        nodes = self._graph(w)[0]
        return self._z(w)[np.searchsorted(nodes, w.accounts)]

    def _proba(self, w, I, J):
        nodes = self._graph(w)[0]
        S = (struct_features(w, I, J) - self.s_mu) / self.s_sd
        lg = self._head(self._z(w), np.searchsorted(nodes, I), np.searchsorted(nodes, J), S)
        return 1.0 / (1.0 + np.exp(-lg))

    def predict_edges(self, w: Window) -> np.ndarray:
        if len(w.coord) == 0:
            return np.empty(0)
        return self._proba(w, w.coord["i"].to_numpy(), w.coord["j"].to_numpy())

    def window_auc(self, w: Window, n_neg: int | None = None) -> float:
        from sklearn.metrics import roc_auc_score
        if len(w.coord) < 5:
            return np.nan
        p_pos = self.predict_edges(w)
        nI, nJ = sample_negatives(w, n_neg or len(p_pos), self.rng)
        if len(nI) < 5:
            return np.nan
        p_neg = self._proba(w, nI, nJ)
        y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
        return float(roc_auc_score(y, np.r_[p_pos, p_neg]))
