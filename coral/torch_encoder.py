"""Learned temporal encoder for CORAL (requires PyTorch; no PyG dependency).

Architecture (per window t, history graph only):
    h0  = MLP(x_hist)                                   node history features
    h^l = ReLU( W_s h^{l-1} + W_c Â_count h^{l-1} + W_r Â_recent h^{l-1} )   l = 1..L
          two edge channels: interaction counts and recency-decayed weights,
          so the encoder sees *when* relationships were active, not just *if*
    z   = LayerNorm(h^L)
Pair head:
    p_ij = σ( MLP([ z_i ⊙ z_j , |z_i − z_j| , struct_ij ]) )

Trained end-to-end on Coordination-Predictive Pretraining (positives: observed
co-activity in window t; negatives: random active pairs). No fraud labels.

Exposes the same interface as SIGNEncoder + CoordinationPredictor:
    fit(windows), embed(w), predict_edges(w), window_auc(w)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .pairs import sample_negatives, struct_features
from .windows import HIST_FEATURES, Window


def _to_torch_sparse(A: sp.csr_matrix, device) -> torch.Tensor:
    A = A + sp.eye(A.shape[0], format="csr")
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    A = (dinv @ A @ dinv).tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A.shape, device=device).coalesce()


class _Encoder(nn.Module):
    def __init__(self, f_in, dim=64, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, dim), nn.ReLU(), nn.Dropout(dropout))
        self.self_ = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.cnt = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        self.rec = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A_cnt, A_rec):
        h = self.inp(x)
        for ws, wc, wr in zip(self.self_, self.cnt, self.rec):
            h = torch.relu(ws(h) + wc(torch.sparse.mm(A_cnt, h)) + wr(torch.sparse.mm(A_rec, h)))
            h = self.drop(h)
        return self.norm(h)


class _PairHead(nn.Module):
    def __init__(self, dim, n_struct, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + n_struct, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, zi, zj, s):
        return self.mlp(torch.cat([zi * zj, (zi - zj).abs(), s], dim=1)).squeeze(1)


class TorchCoordinationModel:
    name = "torch-temporal"

    def __init__(self, dim=64, layers=2, epochs=15, lr=2e-3, neg_ratio=1.0,
                 max_edges_per_window=50_000, device=None, seed=0):
        self.dim, self.layers, self.epochs, self.lr = dim, layers, epochs, lr
        self.neg_ratio, self.max_e = neg_ratio, max_edges_per_window
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    # -------------------------------------------------------------- tensors
    def _graph(self, w: Window):
        key = f"torch_graph_{id(self)}"
        if key in w.extras:
            return w.extras[key]
        N, F = w.hist_A.shape[0], len(HIST_FEATURES)
        X = np.zeros((N, F + 1), np.float32)
        if len(w.hist_X):
            idx = w.hist_X.index.to_numpy(np.int64)
            X[idx, :F] = (w.hist_X[HIST_FEATURES].to_numpy(float) - self.mu) / self.sd
            X[idx, F] = 1.0
        g = (torch.tensor(X, device=self.device),
             _to_torch_sparse(w.hist_A, self.device),
             _to_torch_sparse(w.hist_A_rec, self.device))
        w.extras[key] = g
        return g

    def _z(self, w: Window) -> torch.Tensor:
        x, Ac, Ar = self._graph(w)
        return self.enc(x, Ac, Ar)

    def _logits(self, w: Window, z: torch.Tensor, I, J):
        S = (struct_features(w, I, J) - self.s_mu) / self.s_sd
        S = torch.tensor(S, dtype=torch.float32, device=self.device)
        It = torch.tensor(I, dtype=torch.long, device=self.device)
        Jt = torch.tensor(J, dtype=torch.long, device=self.device)
        return self.head(z[It], z[Jt], S)

    # ------------------------------------------------------------------ fit
    def fit(self, windows: list[Window]):
        rows = [w.hist_X[HIST_FEATURES].to_numpy(float) for w in windows if len(w.hist_X)]
        Xs = np.vstack(rows)
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-6
        S = np.vstack([struct_features(w, w.coord["i"].to_numpy(), w.coord["j"].to_numpy())
                       for w in windows if len(w.coord)])
        self.s_mu, self.s_sd = S.mean(0), S.std(0) + 1e-6
        self.enc = _Encoder(len(HIST_FEATURES) + 1, self.dim, self.layers).to(self.device)
        self.head = _PairHead(self.dim, S.shape[1]).to(self.device)
        opt = torch.optim.AdamW(list(self.enc.parameters()) + list(self.head.parameters()),
                                lr=self.lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(self.epochs):
            self.enc.train(); self.head.train()
            tot = 0.0
            for wi in self.rng.permutation(len(windows)):
                w = windows[wi]
                if len(w.coord) == 0:
                    continue
                I, J = w.coord["i"].to_numpy(), w.coord["j"].to_numpy()
                if len(I) > self.max_e:
                    sel = self.rng.choice(len(I), self.max_e, replace=False)
                    I, J = I[sel], J[sel]
                nI, nJ = sample_negatives(w, int(len(I) * self.neg_ratio) + 10, self.rng)
                z = self._z(w)
                logit = torch.cat([self._logits(w, z, I, J), self._logits(w, z, nI, nJ)])
                y = torch.cat([torch.ones(len(I)), torch.zeros(len(nI))]).to(self.device)
                loss = lossf(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            print(f"[CPP-torch] epoch {ep + 1}/{self.epochs} loss {tot / max(1, len(windows)):.4f}")
        self.enc.eval(); self.head.eval()
        return self

    # ---------------------------------------------------- inference interface
    @torch.no_grad()
    def embed(self, w: Window) -> np.ndarray:
        key = f"torch_emb_{id(self)}"
        if key not in w.extras:
            z = self._z(w)
            w.extras[key] = z[torch.tensor(w.accounts, device=self.device)].cpu().numpy()
        return w.extras[key]

    @torch.no_grad()
    def predict_edges(self, w: Window) -> np.ndarray:
        if len(w.coord) == 0:
            return np.empty(0)
        z = self._z(w)
        lg = self._logits(w, z, w.coord["i"].to_numpy(), w.coord["j"].to_numpy())
        return torch.sigmoid(lg).cpu().numpy()

    @torch.no_grad()
    def window_auc(self, w: Window, n_neg: int | None = None) -> float:
        from sklearn.metrics import roc_auc_score
        if len(w.coord) < 5:
            return np.nan
        p_pos = self.predict_edges(w)
        nI, nJ = sample_negatives(w, n_neg or len(p_pos), self.rng)
        if len(nI) < 5:
            return np.nan
        z = self._z(w)
        p_neg = torch.sigmoid(self._logits(w, z, nI, nJ)).cpu().numpy()
        y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
        return float(roc_auc_score(y, np.r_[p_pos, p_neg]))
