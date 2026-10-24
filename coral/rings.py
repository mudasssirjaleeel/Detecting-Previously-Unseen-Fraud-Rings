"""Ring-level detection via *closure of surprising coordination*.

Insight: a single unexplained edge is common (people meet new counterparties
every day). What a ring adds is that its unexplained edges close on each other:
the members are mutually, simultaneously new to one another. An ordinary new
contact creates a *star* of unexplained edges around one account; a ring
creates a dense, triangle-rich cluster of them.

Procedure per window:
  1. keep "surprising" edges: predicted co-activity p < p_max
  2. edge strength v = specificity (Adamic-Adar, direct flow = 1) x surprise
  3. weighted triangle intensity per account:  T_i = 1/2 * sum_jk  c_ij c_jk c_ki,
     with c = v^(1/3) (so each triangle contributes the geometric mean of its edges)
  4. candidate rings = connected components of edges that sit in >= 1 triangle;
     ring score = mean member intensity x log(1 + size) / (1 + time spread in hours)
  5. ring descriptors (what the money does inside the candidate ring):
       exclusivity  = share of the members' transactions that stay inside the ring
       conservation = sum_v min(in_v, out_v) / sum_v max(in_v, out_v) over members
                      (money passed on rather than consumed; cf. FlowScope, KDD'20)
       amount_cv    = coefficient of variation of the internal transaction amounts
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


RING_DESC = ["ring_size", "ring_spread_h", "ring_exclusivity", "ring_conservation", "ring_amount_cv"]


def closure_scores(edges: pd.DataFrame, accounts: np.ndarray, win_tx: pd.DataFrame,
                   p_max: float = 0.5, min_size: int = 3, return_desc: bool = False):
    """edges: columns i, j, s (surprise), p (predicted prob), aa (specificity).
    Returns (triangle intensity per account, ring score per account, rings) and, with
    return_desc, an (n, 5) array of RING_DESC for each account's best ring (0 if none)."""
    n = len(accounts)
    tri = np.zeros(n); ring = np.zeros(n); rings = []
    desc = np.zeros((n, len(RING_DESC)))
    out = lambda: (tri, ring, rings, desc) if return_desc else (tri, ring, rings)  # noqa: E731
    e = edges[edges["p"] < p_max]
    if len(e) < 3:
        return out()
    I = np.searchsorted(accounts, e["i"].to_numpy())
    J = np.searchsorted(accounts, e["j"].to_numpy())
    c = np.cbrt(e["aa"].to_numpy() * e["s"].to_numpy())
    C = sp.coo_matrix((np.r_[c, c], (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    C.sum_duplicates()
    closed = (C @ C).multiply(C)               # entry (i,j) > 0 iff edge ij closes a triangle
    tri = np.asarray(closed.sum(1)).ravel() / 2.0
    if closed.nnz == 0:
        return out()
    B = closed.copy(); B.data[:] = 1.0
    ncomp, lab = connected_components(B, directed=False)
    in_tri = np.asarray(B.sum(1)).ravel() > 0
    src, dst, ts = win_tx["src"].to_numpy(), win_tx["dst"].to_numpy(), win_tx["ts"].to_numpy()
    amt = win_tx["amount"].to_numpy()
    for k in range(ncomp):
        idx = np.where((lab == k) & in_tri)[0]
        if len(idx) < min_size:
            continue
        members = accounts[idx]
        inside = np.isin(src, members) & np.isin(dst, members)
        touch = np.isin(src, members) | np.isin(dst, members)
        tsel = ts[inside] if inside.sum() >= 2 else ts[touch]
        spread_h = float(np.std(tsel) / 3600) if len(tsel) >= 2 else 0.0
        score = float(tri[idx].mean() * np.log1p(len(idx)) / (1.0 + spread_h))
        excl = float(inside.sum() / max(1, touch.sum()))
        fs = pd.Series(amt[touch & np.isin(src, members)], index=src[touch & np.isin(src, members)])
        fd = pd.Series(amt[touch & np.isin(dst, members)], index=dst[touch & np.isin(dst, members)])
        o = fs.groupby(level=0).sum().reindex(members).fillna(0).to_numpy()
        i_ = fd.groupby(level=0).sum().reindex(members).fillna(0).to_numpy()
        cons = float(np.minimum(o, i_).sum() / max(np.maximum(o, i_).sum(), 1e-9))
        ia = amt[inside]
        cv = float(np.std(ia) / np.mean(ia)) if len(ia) >= 2 and np.mean(ia) > 0 else 1.0
        d = np.array([len(idx), spread_h, excl, cons, cv])
        better = score > ring[idx]
        desc[idx[better]] = d
        ring[idx] = np.maximum(ring[idx], score)
        rings.append({"members": members, "score": score, "size": len(idx),
                      "spread_h": spread_h, "exclusivity": excl, "conservation": cons,
                      "amount_cv": cv})
    return out()
