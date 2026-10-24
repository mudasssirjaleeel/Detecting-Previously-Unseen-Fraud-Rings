"""Strictly causal windowing.

For window t (time range [t*W, (t+1)*W)) everything is built from:
  * the CURRENT window's transactions  -> what we score (detection at window end)
  * the HISTORY [ (t-L)*W, t*W )        -> what the model may condition on
Nothing from after (t+1)*W is ever read. `Window.max_ts_used` records the
latest timestamp touched so the leakage tests can assert this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

DAY = 86400

NODE_FEATURES = [
    "n_out", "n_in", "amt_out", "amt_in", "uniq_out", "uniq_in",
    "amt_mean", "amt_std", "time_std_h", "frac_new_cp", "age_days",
    "hist_n", "hist_uniq_cp", "pass_through", "round_frac", "amt_dev",
]
HIST_FEATURES = ["h_n", "h_uniq", "h_amt_mean", "h_amt_std", "h_out_ratio", "h_active_days"]


@dataclass
class Window:
    t: int
    start: int
    end: int
    accounts: np.ndarray            # active account ids in this window
    X: pd.DataFrame                 # node features (index = accounts)
    y: np.ndarray                   # 1 if account takes part in laundering in window
    typology: np.ndarray            # typology of that participation ("" if none)
    pattern: np.ndarray             # pattern id (-1 if none)
    coord: pd.DataFrame             # coordination edges: i, j, w, dt_min
    hist_A: sp.csr_matrix           # history adjacency (N x N), log1p counts, symmetric
    hist_A_rec: sp.csr_matrix       # same graph, recency-decayed weights (temporal channel)
    hist_X: pd.DataFrame            # history node features (index = accounts seen in history)
    hist_pairs: pd.Series           # pair key -> # past direct transactions
    hist_coord: pd.Series           # pair key -> # past windows co-active
    win_tx: pd.DataFrame            # this window's transactions (for ring tightness)
    max_ts_used: int = 0
    extras: dict = field(default_factory=dict)


def pair_key(i, j, N):
    i = np.asarray(i, dtype=np.int64); j = np.asarray(j, dtype=np.int64)
    a, b = np.minimum(i, j), np.maximum(i, j)
    return a * np.int64(N) + b


class WindowBuilder:
    def __init__(self, tx: pd.DataFrame, window: int = DAY, lookback: int = 7,
                 sync_tau: int = 3600, hub_cap: int = 50, relay_tau: int = 0):
        self.tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
        self.W, self.L, self.tau, self.cap = window, lookback, sync_tau, hub_cap
        self.relay_tau = relay_tau  # time-respecting relays i->c->j (0 disables)
        self.N = int(max(tx["src"].max(), tx["dst"].max())) + 1
        self.n_windows = int(self.tx["ts"].max() // window) + 1
        self._ts = self.tx["ts"].to_numpy()
        self._coord_cache: dict[int, pd.DataFrame] = {}
        self._cache: dict[int, Window] = {}

    # ---------------------------------------------------------------- slices
    def _slice(self, lo, hi):
        a, b = np.searchsorted(self._ts, [lo, hi], side="left")
        return self.tx.iloc[a:b]

    def window_tx(self, t):
        return self._slice(t * self.W, (t + 1) * self.W)

    def history_tx(self, t):
        return self._slice(max(0, t - self.L) * self.W, t * self.W)

    # ---------------------------------------------------------- coordination
    def coordination(self, t) -> pd.DataFrame:
        """Accounts i,j are co-active in window t if they transact directly, or
        both touch a common counterparty within `sync_tau` seconds of each other
        (direction-agnostic, so cycles/stacks/fans are all covered). Very large
        hubs (> hub_cap distinct accounts) are skipped: they carry no ring signal
        and would make this quadratic."""
        if t in self._coord_cache:
            return self._coord_cache[t]
        w = self.window_tx(t)
        N = self.N
        if len(w) == 0:
            out = pd.DataFrame({"i": [], "j": [], "w": [], "dt_min": [], "aa": [], "key": []})
            self._coord_cache[t] = out
            return out
        inc = pd.concat([
            pd.DataFrame({"a": w["src"].values, "cp": w["dst"].values, "ts": w["ts"].values}),
            pd.DataFrame({"a": w["dst"].values, "cp": w["src"].values, "ts": w["ts"].values}),
        ]).drop_duplicates(["a", "cp"])
        deg = inc.groupby("cp")["a"].transform("size")
        inc = inc.assign(deg=deg)
        inc = inc[(deg >= 2) & (deg <= self.cap)]
        m = inc.merge(inc[["a", "cp", "ts"]], on="cp", suffixes=("1", "2"))
        m = m[m["a1"] < m["a2"]]
        m["dt"] = (m["ts1"] - m["ts2"]).abs()
        m = m[m["dt"] <= self.tau]
        # Adamic-Adar style specificity: sharing a busy counterparty (a shop) says
        # little; sharing a low-degree intermediary says a lot.
        m["aa"] = 1.0 / np.log2(1.0 + m["deg"])
        two_hop = m.groupby(["a1", "a2"]).agg(w=("cp", "size"), dt_min=("dt", "min"),
                                              aa=("aa", "sum")).reset_index()
        two_hop.columns = ["i", "j", "w", "dt_min", "aa"]
        parts = [two_hop]
        if self.relay_tau > 0:
            parts.append(self._relays(w))
        d = pd.DataFrame({"i": np.minimum(w["src"].values, w["dst"].values),
                          "j": np.maximum(w["src"].values, w["dst"].values)})
        direct = d.groupby(["i", "j"]).size().rename("w").reset_index()
        direct["dt_min"] = 0
        direct["aa"] = 1.0  # a direct money flow is maximally specific
        c = pd.concat(parts + [direct]).groupby(["i", "j"]).agg(
            w=("w", "sum"), dt_min=("dt_min", "min"), aa=("aa", "sum")).reset_index()
        c["aa"] = c["aa"].clip(upper=2.0)
        c["key"] = pair_key(c["i"], c["j"], N)
        self._coord_cache[t] = c
        return c

    def _relays(self, w: pd.DataFrame) -> pd.DataFrame:
        """Time-respecting relays: money enters intermediary c from i and leaves c
        to j within relay_tau seconds (in that order). Then i and j are co-active
        even if the two hops are far apart from the point of view of sync_tau.
        This is the temporal-path notion of co-activity; it links consecutive-but-one
        members of cycles, stacks and layered chains."""
        din = w.groupby("dst")["src"].nunique()
        dout = w.groupby("src")["dst"].nunique()
        deg = din.add(dout, fill_value=0)
        ok = deg[(deg <= self.cap)].index
        fin = w.loc[w["dst"].isin(ok), ["src", "dst", "ts"]].rename(columns={"src": "i", "dst": "c", "ts": "t1"})
        fout = w.loc[w["src"].isin(ok), ["src", "dst", "ts"]].rename(columns={"src": "c", "dst": "j", "ts": "t2"})
        m = fin.merge(fout, on="c")
        m = m[(m["i"] != m["j"]) & (m["t2"] >= m["t1"]) & (m["t2"] - m["t1"] <= self.relay_tau)]
        if len(m) == 0:
            return pd.DataFrame({"i": [], "j": [], "w": [], "dt_min": [], "aa": []})
        m["aa"] = 1.0 / np.log2(1.0 + deg.reindex(m["c"]).to_numpy())
        a, b = np.minimum(m["i"].to_numpy(), m["j"].to_numpy()), np.maximum(m["i"].to_numpy(), m["j"].to_numpy())
        r = pd.DataFrame({"i": a, "j": b, "dt": (m["t2"] - m["t1"]).to_numpy(), "aa": m["aa"].to_numpy()})
        r = r.drop_duplicates(["i", "j", "aa"])
        return r.groupby(["i", "j"]).agg(w=("aa", "size"), dt_min=("dt", "min"), aa=("aa", "sum")).reset_index()

    # --------------------------------------------------------------- features
    @staticmethod
    def _hist_node_features(h: pd.DataFrame, W: int) -> pd.DataFrame:
        if len(h) == 0:
            return pd.DataFrame(columns=HIST_FEATURES)
        inc = pd.concat([
            pd.DataFrame({"a": h["src"].values, "cp": h["dst"].values, "amt": h["amount"].values,
                          "out": 1, "day": h["ts"].values // W}),
            pd.DataFrame({"a": h["dst"].values, "cp": h["src"].values, "amt": h["amount"].values,
                          "out": 0, "day": h["ts"].values // W}),
        ])
        inc["la"] = np.log1p(inc["amt"])
        g = inc.groupby("a")
        f = pd.DataFrame({
            "h_n": np.log1p(g.size()),
            "h_uniq": np.log1p(g["cp"].nunique()),
            "h_amt_mean": g["la"].mean(),
            "h_amt_std": g["la"].std().fillna(0),
            "h_out_ratio": g["out"].mean(),
            "h_active_days": g["day"].nunique(),
        })
        return f

    def build(self, t: int) -> Window:
        if t in self._cache:
            return self._cache[t]
        N, W = self.N, self.W
        w = self.window_tx(t)
        h = self.history_tx(t)
        start, end = t * W, (t + 1) * W

        inc = pd.concat([
            pd.DataFrame({"a": w["src"].values, "cp": w["dst"].values, "amt": w["amount"].values,
                          "ts": w["ts"].values, "out": 1, "lab": w["is_laundering"].values,
                          "typ": w["typology"].values, "pid": w["pattern_id"].values}),
            pd.DataFrame({"a": w["dst"].values, "cp": w["src"].values, "amt": w["amount"].values,
                          "ts": w["ts"].values, "out": 0, "lab": w["is_laundering"].values,
                          "typ": w["typology"].values, "pid": w["pattern_id"].values}),
        ])
        accounts = np.sort(inc["a"].unique())
        inc["la"] = np.log1p(inc["amt"])
        g = inc.groupby("a")
        outg = inc[inc["out"] == 1].groupby("a")
        ing = inc[inc["out"] == 0].groupby("a")

        # history relationships
        hk = pair_key(h["src"].values, h["dst"].values, N)
        hist_pairs = pd.Series(hk).value_counts()
        inc_key = pair_key(inc["a"].values, inc["cp"].values, N)
        inc["new_cp"] = ~np.isin(inc_key, hist_pairs.index.values)

        hist_X = self._hist_node_features(h, W)
        # account age: first time the account was seen at or before this window
        first = self._first_seen_upto(end)
        age = (start - first.reindex(accounts).values) / DAY
        age = np.clip(np.nan_to_num(age, nan=0.0), 0, None)

        X = pd.DataFrame(index=accounts)
        X["n_out"] = np.log1p(outg.size()).reindex(accounts).fillna(0)
        X["n_in"] = np.log1p(ing.size()).reindex(accounts).fillna(0)
        X["amt_out"] = np.log1p(outg["amt"].sum()).reindex(accounts).fillna(0)
        X["amt_in"] = np.log1p(ing["amt"].sum()).reindex(accounts).fillna(0)
        X["uniq_out"] = np.log1p(outg["cp"].nunique()).reindex(accounts).fillna(0)
        X["uniq_in"] = np.log1p(ing["cp"].nunique()).reindex(accounts).fillna(0)
        X["amt_mean"] = g["la"].mean().reindex(accounts)
        X["amt_std"] = g["la"].std().reindex(accounts).fillna(0)
        X["time_std_h"] = (g["ts"].std() / 3600).reindex(accounts).fillna(0)
        X["frac_new_cp"] = g["new_cp"].mean().reindex(accounts)
        X["age_days"] = age
        X["hist_n"] = hist_X["h_n"].reindex(accounts).fillna(0).values if len(hist_X) else 0.0
        X["hist_uniq_cp"] = hist_X["h_uniq"].reindex(accounts).fillna(0).values if len(hist_X) else 0.0
        ai = np.expm1(X["amt_in"]); ao = np.expm1(X["amt_out"])
        X["pass_through"] = (np.minimum(ai, ao) / np.maximum(np.maximum(ai, ao), 1e-9)).values
        X["round_frac"] = g["amt"].apply(lambda s: np.mean(np.isclose(s % 100, 0, atol=.5))).reindex(accounts)
        hm = hist_X["h_amt_mean"].reindex(accounts).values if len(hist_X) else np.full(len(accounts), np.nan)
        X["amt_dev"] = np.nan_to_num(X["amt_mean"].values - hm, nan=0.0)
        X = X[NODE_FEATURES].astype(float).fillna(0.0)

        # labels (laundering participation in this window)
        lab = inc[inc["lab"] == 1].drop_duplicates("a").set_index("a")
        y = np.isin(accounts, lab.index.values).astype(np.int8)
        typ = lab["typ"].reindex(accounts).fillna("").values
        pid = lab["pid"].reindex(accounts).fillna(-1).astype(np.int64).values

        # history adjacency
        if len(h):
            hh = pd.DataFrame({"k": hk}).value_counts().reset_index()
            a_ = (hh["k"].values // N).astype(np.int64)
            b_ = (hh["k"].values % N).astype(np.int64)
            v = np.log1p(hh["count"].values.astype(float))
            A = sp.coo_matrix((np.r_[v, v], (np.r_[a_, b_], np.r_[b_, a_])), shape=(N, N)).tocsr()
            dec = np.exp(-(start - h["ts"].to_numpy()) / (2.0 * W))
            rr = pd.DataFrame({"k": hk, "d": dec}).groupby("k")["d"].sum()
            ra = (rr.index.to_numpy() // N).astype(np.int64)
            rb = (rr.index.to_numpy() % N).astype(np.int64)
            rv = rr.to_numpy()
            Arec = sp.coo_matrix((np.r_[rv, rv], (np.r_[ra, rb], np.r_[rb, ra])), shape=(N, N)).tocsr()
        else:
            A = sp.csr_matrix((N, N))
            Arec = sp.csr_matrix((N, N))

        # past co-activity (from cached coordination of previous windows only)
        past = [self.coordination(s)[["key", "w"]] for s in range(max(0, t - self.L), t)]
        if past:
            pc = pd.concat(past)
            hist_coord = pc.groupby("key").size()
        else:
            hist_coord = pd.Series(dtype=float)

        coord = self.coordination(t)
        max_ts = int(max(w["ts"].max() if len(w) else 0, h["ts"].max() if len(h) else 0))
        win = Window(t=t, start=start, end=end, accounts=accounts, X=X, y=y, typology=typ,
                     pattern=pid, coord=coord, hist_A=A, hist_A_rec=Arec, hist_X=hist_X, hist_pairs=hist_pairs,
                     hist_coord=hist_coord, win_tx=w, max_ts_used=max_ts)
        self._cache[t] = win
        return win

    def _first_seen_upto(self, end) -> pd.Series:
        sub = self._slice(0, end)
        s = pd.concat([sub[["src", "ts"]].rename(columns={"src": "a"}),
                       sub[["dst", "ts"]].rename(columns={"dst": "a"})])
        return s.groupby("a")["ts"].min()
