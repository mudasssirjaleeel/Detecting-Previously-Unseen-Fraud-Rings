#!/usr/bin/env python3
"""Run the Leave-One-Typology-Out benchmark.

Examples
  # synthetic smoke test (minutes on a laptop)
  python scripts/run_loto.py --data synthetic --out results/synthetic

  # IBM AMLworld (download HI-Small_Trans.csv and HI-Small_Patterns.txt from Kaggle)
  python scripts/run_loto.py --data ibm --trans data/HI-Small_Trans.csv \
      --patterns data/HI-Small_Patterns.txt --out results/hi_small

  # learned temporal encoder (needs PyTorch; uses GPU if available)
  python scripts/run_loto.py --data ibm ... --encoder torch

  # your own ledger (columns ts,src,dst,amount[,is_laundering,pattern_id,typology])
  python scripts/run_loto.py --data csv --trans my_ledger.csv --out results/mine
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from coral.data import SyntheticConfig, generate_synthetic, load_amlsim, load_generic_csv, load_ibm_aml  # noqa
from coral.evaluate import run_loto, summarize  # noqa
from coral.model import CORALConfig  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "ibm", "amlsim", "csv"], default="synthetic")
    ap.add_argument("--alerts", help="AMLSim alert_members.csv (with --data amlsim --trans tx_log.csv)")
    ap.add_argument("--trans"); ap.add_argument("--patterns")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    ap.add_argument("--typologies", nargs="*", default=None)
    ap.add_argument("--encoder", choices=["sign", "gnn", "torch"], default="sign")
    ap.add_argument("--window_hours", type=float, default=24)
    ap.add_argument("--window_days", type=float, default=None, help="overrides --window_hours")
    ap.add_argument("--lookback", type=int, default=7, help="history windows")
    ap.add_argument("--sync_minutes", type=float, default=60)
    ap.add_argument("--hub_cap", type=int, default=50)
    ap.add_argument("--ring_pmax", type=float, default=0.5, help="surprise threshold p_max for closure")
    ap.add_argument("--relay_hours", type=float, default=0, help="time-respecting relay co-activity (0 = off)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--no_supervised_head", action="store_true")
    ap.add_argument("--cpp_ablation", action="store_true", help="also run CORAL-S with a heuristic predictor")
    ap.add_argument("--world_seed", type=int, default=7,
                    help="synthetic generator seed (7 = held-out report world, 11 = development world)")
    ap.add_argument("--drift", action="store_true", help="synthetic: inject graph drift in the test period")
    ap.add_argument("--max_neg", type=int, default=None,
                    help="supervised models: keep all positives, subsample negatives to this many")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()

    if a.max_neg:
        import coral.baselines as _B
        _B.MAX_NEG = a.max_neg
    if a.data == "synthetic":
        tx = generate_synthetic(SyntheticConfig(seed=a.world_seed, drift=a.drift))
    elif a.data == "amlsim":
        tx = load_amlsim(a.trans, a.alerts)
    elif a.data == "ibm":
        tx = load_ibm_aml(a.trans, a.patterns, nrows=a.nrows)
    else:
        tx = load_generic_csv(a.trans)
    print(f"{len(tx):,} transactions, {tx[['src','dst']].max().max()+1:,} accounts")
    print(tx[tx.is_laundering == 1].typology.value_counts().to_string())

    os.makedirs(a.out, exist_ok=True)
    allres = []
    for seed in a.seeds:
        cfg = CORALConfig(encoder=a.encoder, seed=seed, ring_pmax=a.ring_pmax)
        res, gate, meta = run_loto(tx, a.typologies, cfg, window=int(a.window_days * 86400) if a.window_days else int(a.window_hours * 3600),
                                   lookback=a.lookback, sync_tau=int(a.sync_minutes * 60),
                                   hub_cap=a.hub_cap, relay_tau=int(a.relay_hours * 3600),
                                   with_supervised_head=not a.no_supervised_head,
                                   cpp_ablation=a.cpp_ablation, score_dir=f"{a.out}/scores")
        res["seed"] = seed
        allres.append(res)
        gate.to_csv(f"{a.out}/gate_seed{seed}.csv", index=False)
    import pandas as pd
    res = pd.concat(allres)
    res.to_csv(f"{a.out}/results.csv", index=False)
    s = summarize(res)
    s.to_csv(f"{a.out}/summary.csv")
    with open(f"{a.out}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print("\n==== mean over held-out typologies (and seeds) ====")
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
