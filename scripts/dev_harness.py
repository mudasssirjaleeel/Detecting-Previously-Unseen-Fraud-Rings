#!/usr/bin/env python3
"""Development harness: fit CORAL once per held-out typology on the DEVELOPMENT
world (generator seed 11, never used for reported numbers) and compare scoring
variants on its test windows. Prints mean AUPRC / recall@1%FPR per variant."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from coral.data import SyntheticConfig, generate_synthetic  # noqa: E402
from coral.evaluate import holdout, metrics, split_windows  # noqa: E402
from coral.model import CORAL, CORALConfig, _pct  # noqa: E402
from coral.rings import RING_DESC, closure_scores  # noqa: E402
from coral.windows import DAY, WindowBuilder  # noqa: E402


def fisher(*ps):
    """Fisher-style tail combination of within-window percentiles."""
    return sum(-np.log(np.clip(1 - p, 1e-6, 1)) for p in ps)


def variants(c: CORAL, w):
    gs, _ = c.graph_scores(w)
    feat = _pct(c.feature_score(w))
    p = c.cpp.predict_edges(w)
    e = pd.DataFrame({"i": w.coord["i"].to_numpy(), "j": w.coord["j"].to_numpy(),
                      "s": -np.log(p + 1e-3), "p": p, "aa": w.coord["aa"].to_numpy(float)})
    tri, ring, _, desc = closure_scores(e, w.accounts, w.win_tx, c.cfg.ring_pmax, 3, True)
    D = pd.DataFrame(desc, columns=RING_DESC)
    pn, pt, pr = _pct(gs["node_surprise"]), _pct(tri), _pct(ring)
    excl, cons, cv = D["ring_exclusivity"].to_numpy(), D["ring_conservation"].to_numpy(), D["ring_amount_cv"].to_numpy()
    flow = ring * (0.5 + excl) * (0.5 + cons) / (0.5 + np.minimum(cv, 2))
    pf = _pct(flow)
    a = 0.6
    fl = lambda e_, c_, v_: _pct(ring * ((0.5 + excl) if e_ else 1) * ((0.5 + cons) if c_ else 1) / ((0.5 + np.minimum(cv, 2)) if v_ else 1))  # noqa
    out = {
        "flow: excl only": fl(1, 0, 0), "flow: cons only": fl(0, 1, 0), "flow: cv only": fl(0, 0, 1),
        "flow: excl+cons (no cv)": fl(1, 1, 0), "flow: all three": fl(1, 1, 1),
        "flow(excl+cons) + 0.1*fisher(tri,feat)": fl(1, 1, 0) + 0.1 * _pct(fisher(pt, feat)),
        "flow(all) + 0.1*fisher(tri,feat)": pf + 0.1 * _pct(fisher(pt, feat)),
    }
    out0 = {
        "U current (rank-avg, a=.6)": a * _pct((pn + pt + pr) / 3) + (1 - a) * feat,
        "graph only (current)": (pn + pt + pr) / 3,
        "fisher(node,tri,ring)": fisher(pn, pt, pr),
        "fisher(node,tri,ring,feat)": fisher(pn, pt, pr, feat),
        "flow ring only": pf,
        "fisher(node,tri,flowring)": fisher(pn, pt, pf),
        "fisher(node,tri,flowring,feat)": fisher(pn, pt, pf, feat),
        "fisher(tri,flowring,feat)": fisher(pt, pf, feat),
        "feature only (IForest)": feat,
    }
    out.update({k: out0[k] for k in ["U current (rank-avg, a=.6)", "feature only (IForest)"]})
    return out


def main(world_seed=11, typs=None):
    tx = generate_synthetic(SyntheticConfig(seed=world_seed))
    n = int(tx.ts.max() // DAY) + 1
    tr, va, te = split_windows(n)
    all_typ = sorted(t for t in tx.typology.unique() if t and t != "UNKNOWN")
    rows = []
    for h in typs or all_typ:
        b = WindowBuilder(holdout(tx, h, te[0] * DAY))
        c = CORAL(CORALConfig()).fit(b, tr, va)
        S, Y, T, P = {}, [], [], []
        for t in te:
            w = b.build(t)
            for k, v in variants(c, w).items():
                S.setdefault(k, []).append(v)
            Y.append(w.y); T.append(w.typology); P.append(w.pattern)
        y, typ, pat = np.concatenate(Y), np.concatenate(T), np.concatenate(P)
        seen = [x for x in all_typ if x != h]
        for k, v in S.items():
            m = metrics(np.concatenate(v), y, typ, pat, h, seen)
            rows.append({"heldout": h, "variant": k, **m})
        print("done", h, flush=True)
    R = pd.DataFrame(rows)
    print(R.groupby("variant")[["auprc", "recall@1%fpr", "ring_recall@1%fpr", "roc_auc"]]
          .mean().sort_values("auprc", ascending=False).round(3).to_string())
    print(R.pivot_table(index="heldout", columns="variant", values="auprc").round(3).T.to_string())
    return R


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
