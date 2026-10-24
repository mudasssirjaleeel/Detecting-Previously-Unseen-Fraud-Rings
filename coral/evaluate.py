"""Leave-One-Typology-Out (LOTO) evaluation.

For each typology h:
  1. Split windows in time: train | val | test (no shuffling, ever).
  2. Delete every transaction of typology h from the train and val periods, so
     h is genuinely unseen: not in the labels, not in the graph, not in the
     self-supervised pretraining data.
  3. Fit CORAL and all baselines on train (val is used only for the gate's
     label-free reference statistics).
  4. Score test windows. Evaluate on: negatives + positives of typology h.

Metrics: AUPRC, ROC-AUC, recall at 1% FPR, ring recall at 1% FPR (a pattern
counts as found if >= 50% of its member account-windows exceed the threshold),
and AUPRC on the typologies that WERE seen (to show the trade-off)."""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import baselines as B
from .model import CORAL, CORALConfig, CORALSupervised, _pct, fisher
from .windows import DAY, WindowBuilder


def split_windows(n: int, train_frac=0.57, val_frac=0.14, warmup=2):
    a = int(round(n * train_frac)); b = int(round(n * (train_frac + val_frac)))
    return list(range(warmup, a)), list(range(a, b)), list(range(b, n))


def holdout(tx: pd.DataFrame, typology: str, test_start_ts: int) -> pd.DataFrame:
    drop = (tx["typology"] == typology) & (tx["ts"] < test_start_ts)
    return tx[~drop].reset_index(drop=True)


class HeuristicPredictor:
    """Ablation: replace learned CPP with 'is this relationship new?'."""

    def predict_edges(self, w):
        if len(w.coord) == 0:
            return np.empty(0)
        k = w.coord["key"]
        known = (w.hist_pairs.reindex(k).fillna(0).to_numpy() > 0) | \
                (w.hist_coord.reindex(k).fillna(0).to_numpy() > 0)
        return np.where(known, 0.9, 0.1)


def metrics(score, y, typ, pat, h, seen):
    score = np.asarray(score, float)
    ev = (y == 0) | (typ == h)
    out = {}
    yy, ss = y[ev], score[ev]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return {k: np.nan for k in ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr",
                                    "seen_auprc", "n_pos"]}
    out["auprc"] = average_precision_score(yy, ss)
    out["roc_auc"] = roc_auc_score(yy, ss)
    thr = np.quantile(score[y == 0], 0.99)
    out["recall@1%fpr"] = float(np.mean(score[(y == 1) & (typ == h)] > thr))
    m = (y == 1) & (typ == h)
    g = pd.DataFrame({"p": pat[m], "hit": score[m] > thr}).groupby("p")["hit"].mean()
    out["ring_recall@1%fpr"] = float(np.mean(g >= 0.5)) if len(g) else np.nan
    sv = (y == 0) | np.isin(typ, seen)
    out["seen_auprc"] = average_precision_score(y[sv], score[sv]) if y[sv].sum() else np.nan
    out["n_pos"] = int(m.sum())
    out["base_rate"] = float(yy.mean())
    return out


def run_loto(tx: pd.DataFrame, typologies: list[str] | None = None,
             cfg: CORALConfig | None = None, window: int = DAY, lookback: int = 7,
             sync_tau: int = 3600, hub_cap: int = 50, relay_tau: int = 0, verbose: bool = True,
             with_supervised_head: bool = True, cpp_ablation: bool = False, score_dir: str | None = None):
    cfg = cfg or CORALConfig()
    all_typ = sorted(t for t in tx["typology"].unique() if t and t != "UNKNOWN")
    typologies = typologies or all_typ
    n_windows = int(tx["ts"].max() // window) + 1
    train_t, val_t, test_t = split_windows(n_windows)
    test_start = test_t[0] * window
    results, gate_log = [], []

    for h in typologies:
        t0 = time.time()
        seen = [t for t in all_typ if t != h]
        txh = holdout(tx, h, test_start)
        bld = WindowBuilder(txh, window, lookback, sync_tau, hub_cap, relay_tau)
        tr = [bld.build(t) for t in train_t]
        assert not any((w.typology == h).any() for w in tr), "held-out typology leaked into train"

        coral = CORAL(cfg).fit(bld, train_t, val_t)
        sup = CORALSupervised(coral, cfg.seed).fit(bld, train_t) if with_supervised_head else None
        heur = HeuristicPredictor()
        sup_h = CORALSupervised(coral, cfg.seed, cpp_override=heur).fit(bld, train_t) \
            if (with_supervised_head and cpp_ablation) else None
        base = {cls.name: cls().fit(tr) for cls in B.ALL}

        rows = []
        for t in test_t:
            w = bld.build(t)
            sc, _ = coral.score(w)
            hs, _ = coral.graph_scores(w, cpp=heur)
            a = sc["alpha"].iloc[0]
            pf = _pct(sc["feature"].to_numpy())
            G = lambda r, tri: _pct(_pct(r) + 0.1 * _pct(np.asarray(tri)))  # noqa: E731
            mix = lambda g, al=a: al * g + (1 - al) * pf  # noqa: E731
            ring, tri = sc["ring"].to_numpy(), sc["triangle"].to_numpy()
            ex, co = sc["ring_exclusivity"].to_numpy(), sc["ring_conservation"].to_numpy()
            d = pd.DataFrame({"window": t, "account": w.accounts, "y": w.y,
                              "typology": w.typology, "pattern": w.pattern})
            d["CORAL-U"] = sc["coral"].to_numpy()
            d["CORAL-U (no gate)"] = mix(sc["graph"].to_numpy(), cfg.alpha0)
            d["CORAL-U (no flow: closure only)"] = mix(G(ring, tri))
            d["CORAL-U (no amount coherence)"] = mix(G(ring * (0.5 + ex) * (0.5 + co), tri))
            d["CORAL-U (no rings)"] = mix(_pct(sc["node_surprise"].to_numpy()))
            d["CORAL-U (graph only)"] = sc["graph"].to_numpy()
            d["CORAL-U (heuristic, no CPP)"] = mix(hs["graph"])
            if sup is not None:
                d["CORAL-S"] = sup.score(w)
            if sup_h is not None:
                d["CORAL-S (heuristic, no CPP)"] = sup_h.score(w)
            if sup is not None:
                # Gated CORAL-S: the same label-free gate, falling back to the graph-free
                # supervised model. a/alpha0 == 1 whenever the gate is inside its dead zone,
                # so in stable windows this is exactly CORAL-S.
                aS = a / cfg.alpha0
                d["CORAL-S (gated)"] = aS * d["CORAL-S"] + (1 - aS) * base["GBDT"].score(w)
            for name, m in base.items():
                d[name] = m.score(w)
            rows.append(d)
            gate_log.append({"heldout": h, "window": t, "reliability": sc["reliability"].iloc[0],
                             "alpha": a})
        D = pd.concat(rows, ignore_index=True)
        if score_dir:
            import os
            os.makedirs(score_dir, exist_ok=True)
            D.to_csv(f"{score_dir}/scores_{h}_seed{cfg.seed}.csv.gz", index=False)
        methods = [c for c in D.columns if c not in ("window", "account", "y", "typology", "pattern")]
        y, typ, pat = D["y"].to_numpy(), D["typology"].to_numpy(), D["pattern"].to_numpy()
        for mname in methods:
            r = metrics(D[mname].to_numpy(), y, typ, pat, h, seen)
            r.update({"heldout": h, "method": mname,
                      "label_free": mname.startswith("CORAL-U") or mname == "IsolationForest"})
            results.append(r)
        if verbose:
            best = pd.DataFrame([r for r in results if r["heldout"] == h]).sort_values(
                "auprc", ascending=False)[["method", "auprc", "recall@1%fpr", "ring_recall@1%fpr"]]
            print(f"\n=== held-out {h}  ({time.time() - t0:.1f}s, positives={int(((y==1)&(typ==h)).sum())})")
            print(best.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    res = pd.DataFrame(results)
    meta = {"config": asdict(cfg), "window": window, "lookback": lookback, "sync_tau": sync_tau,
            "hub_cap": hub_cap, "relay_tau": relay_tau, "train_windows": train_t, "val_windows": val_t,
            "test_windows": test_t}
    return res, pd.DataFrame(gate_log), meta


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr", "seen_auprc"]
    s = res.groupby("method")[cols].mean().sort_values("auprc", ascending=False)
    s.insert(0, "label_free", res.groupby("method")["label_free"].first())
    return s
