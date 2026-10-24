# Detecting Previously Unseen Fraud Rings with Self-Supervised Temporal Graph Neural Networks under Leakage-Free Evaluation

Code, data and results for the paper by **Mudassir Jaleel** and **Qaiser Farooq** (independent researchers, Pakistan), submitted to *Expert Systems with Applications*.

**Quick start**
```bash
pip install -r requirements.txt
python tests/test_leakage.py && python tests/test_np_gnn.py
python scripts/run_loto.py --data synthetic --seeds 0 --out results/demo
```
Or open `CORAL_fraud_rings.ipynb` (runs on Kaggle, Colab or locally).

| Folder | Contents |
|---|---|
| `coral/` | The CORAL package: data loaders, causal windows, co-activity predictor, closure + flow ring scoring, reliability gate, baselines, LOTO evaluation |
| `scripts/` | Command-line runner and the scripts that generate every table in the paper from `results/` |
| `results/final/` | All reported runs (`results.csv`, gate logs, run logs) |
| `results/sensitivity/` | Sensitivity analysis |
| `dev/` | Development-world runs used for design decisions |
| `datasets/` | Synthetic benchmark (development, held-out, drift) and the two IBM AMLSim worlds |
| `amlsim_tools/` | Everything needed to regenerate the AMLSim data, including the disclosed patches |
| `tests/` | Leakage tests and the GNN gradient check |

---

# CORAL — Coordination-aware, Reliability-gated Anomaly Learning

Reference implementation for the paper

> **Detecting Previously Unseen Fraud Rings with Self-Supervised Temporal Graph Neural Networks under Leakage-Free Evaluation**

CORAL detects fraud rings, including fraud *typologies it has never seen*, without fraud labels. It is evaluated under a protocol where the held-out typology is absent from training labels, the training graph, and the self-supervised pretraining data, and where every score is provably computed from the past only.

---

## 1. Method

| # | Component | Idea | Code |
|---|---|---|---|
| 1 | **Coordination-Predictive Pretraining (CPP)** | Self-supervised task: from an account pair's *history only*, predict whether the two accounts will be **co-active** (synchronized: direct flow, or touching a common counterparty within τ) in the next window. No fraud labels. | `coral/pairs.py`, `coral/torch_encoder.py` |
| 2 | **Coordination surprise** | An observed co-activity edge the predictor did not expect: `s_ij = −log p_ij`, weighted by edge **specificity** (Adamic–Adar: sharing a busy shop says little; a direct flow or a low-degree intermediary says a lot). | `coral/model.py`, `coral/windows.py` |
| 3 | **Ring detection by closure** | One unexplained edge is common: people meet new counterparties every day, and that produces a *star* of new edges. A ring's unexplained edges **close on each other** (members are mutually and simultaneously new to one another). CORAL scores weighted triangle closure of surprising edges, then groups triangle-supported edges into candidate rings. | `coral/rings.py` |
| 4 | **Flow-aware ring scoring** | Each candidate ring is weighted by what its money does: **exclusivity** (share of members' transactions that stay inside), **conservation** (money passed on rather than consumed, cf. FlowScope) and **amount coherence**. | `coral/rings.py`, `CORAL.graph_scores` |
| 5 | **Drift-aware reliability gate** | Blends the graph score with a **graph-free** feature detector. The graph's weight backs off when label-free signals say the graph is unreliable this window: (a) the predictor's own AUC on this window's coordination, (b) KS shift of the coordination-degree distribution, (c) embedding drift. A 2-s.d. dead zone means it costs nothing when nothing has changed. | `CORAL.reliability` |
| 6 | **CORAL-S** (optional) | A supervised GBDT head on node features + CORAL graph signals + ring descriptors. Training-window graph signals are **cross-fitted** (2 folds by window) so the head never sees in-sample, optimistically low surprise. | `CORALSupervised` |

Optional **time-respecting relay co-activity** (`--relay_hours`): money i→c→j through an intermediary within a few hours links i and j. Use it on ledgers whose hops are hours or days apart (AMLworld, AMLSim).

Two encoder backends share one interface:
- `sign`: parameter-free propagation `[X, ÂX, Â²X]` over the history graph plus a gradient-boosted pair head. NumPy/SciPy only; fast and strong.
- `gnn`: a learned temporal message-passing encoder with **two edge channels** (interaction counts and recency-decayed weights), residual connections and an MLP pair head, trained end-to-end on CPP. Pure NumPy with hand-derived gradients, checked against finite differences in `tests/test_np_gnn.py`.
- `torch`: the same architecture in PyTorch (GPU if available).

## 2. Evaluation protocol: Leave-One-Typology-Out (LOTO)

For each typology *h*:
1. Time-ordered split into train | val | test windows (no shuffling).
2. **Delete every transaction of typology *h* from train and val.** So *h* is not in the labels, not in the graph, and not in the self-supervised data.
3. Fit everything on train. Validation is used only for the gate's label-free reference statistics, never for labels.
4. Score test windows and evaluate on negatives + positives of *h*.

Metrics: AUPRC, ROC-AUC, recall at 1% FPR, **ring recall** at 1% FPR (a pattern counts as found if ≥ 50% of its member account-windows exceed the threshold), and AUPRC on the typologies that *were* seen.

**Leakage guarantees (tested in `tests/test_leakage.py`):**
- Each window's features use only its own transactions plus the history before it (`Window.max_ts_used < Window.end`).
- Scores for window *t* are **bit-identical** when every transaction after *t* is scrambled.
- All normalization statistics and models are fit on training windows only.
- The held-out typology is verifiably absent from every training window.

## 3. Install & run

```bash
pip install -r requirements.txt          # + torch / xgboost optionally
python tests/test_leakage.py             # leakage tests (~10 s)
python tests/test_np_gnn.py              # gradient check of the NumPy GNN
python scripts/run_loto.py --data synthetic --seeds 0 1 2 --out results/synthetic
```

**IBM AMLworld** (Altman et al., NeurIPS 2023 Datasets & Benchmarks; on Kaggle as *"IBM Transactions for Anti Money Laundering (AML)"*): download e.g. `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`, then:

```bash
python scripts/run_loto.py --data ibm \
    --trans data/HI-Small_Trans.csv --patterns data/HI-Small_Patterns.txt \
    --seeds 0 1 2 3 4 --out results/hi_small
# learned encoder:
python scripts/run_loto.py --data ibm ... --encoder torch
```

The patterns file supplies the typology of each laundering attempt (FAN-OUT, FAN-IN, CYCLE, SCATTER-GATHER, …); the loader reads whatever typology names the file contains. Laundering transactions that can't be matched to a pattern are labelled `UNKNOWN` and are never treated as normal.

**Your own ledger:** `--data csv --trans ledger.csv` with columns `ts, src, dst, amount` and optionally `is_laundering, pattern_id, typology`.

Useful flags: `--window_hours`, `--lookback` (history windows), `--sync_minutes` (τ), `--hub_cap`, `--relay_hours`, `--encoder sign|gnn|torch`, `--max_neg` (subsample negatives for big ledgers), `--typologies FAN-IN CYCLE`, `--nrows` (quick look at a big file).
Synthetic only: `--world_seed` (7 = held-out report world, 11 = development world), `--drift` (stress test).

Outputs in `--out`: `results.csv` (every method × held-out typology × seed), `summary.csv` (means), `gate_seed*.csv` (per-window reliability and α), `meta.json` (exact splits and config).

## 4. Baselines and ablations (all run automatically)

| Method | Labels? | What it tests |
|---|---|---|
| RF, GBDT (XGBoost if installed) | seen typologies | strong tabular baselines that beat GNNs under strict inductive evaluation |
| Graph-GBDT | seen typologies | supervised graph baseline (node features + SIGN embedding) |
| IsolationForest | none | label-free feature baseline |
| **CORAL-U** | **none** | full method |
| CORAL-U (no gate) | none | value of the reliability gate |
| CORAL-U (no rings) | none | value of closure-based ring scoring |
| CORAL-U (graph only) | none | graph signal alone |
| CORAL-U (heuristic, no CPP) | none | learned CPP vs "is this relationship new?" |
| **CORAL-S** | seen typologies | CORAL signals + supervised head |

## 5. Code map

```
coral/data.py           loaders (IBM AMLworld, generic CSV) + synthetic world with legit decoys
coral/windows.py        strictly causal windows, node features, coordination graph
coral/pairs.py          Coordination-Predictive Pretraining (pair features, predictor)
coral/encoders.py       SIGN history encoder
coral/torch_encoder.py  learned temporal encoder + pair head (PyTorch)
coral/rings.py          closure-based ring detection
coral/model.py          CORAL-U (surprise, rings, gate) and CORAL-S
coral/baselines.py      RF, GBDT/XGBoost, Graph-GBDT, IsolationForest
coral/evaluate.py       LOTO protocol + metrics
scripts/run_loto.py     CLI
tests/test_leakage.py   leakage tests
```

## 6. Synthetic world (for development, not for the paper's main claims)

`generate_synthetic()` includes **legitimate look-alikes** so the task is not trivial:
- weekly payroll: recurring synchronized fan-out, predictable from history
- friend group-splits: synchronized fan-in among people with shared history
- novel legitimate bursts: new relationships, e.g. a new seller
- account churn: 15% of accounts open mid-period, so "new account" alone is not a fraud tell

Laundering amounts overlap normal amounts. Ring members mix fresh and existing accounts.

## 7. Results (synthetic benchmark)

All design choices were made on a **development world** (`--world_seed 11`) and frozen. Reported numbers come from the **held-out world** (`--world_seed 7`), which was not inspected during development. Mean over 7 held-out typologies.

| Method | Labels | AUPRC | Recall @1% FPR | Ring recall @1% FPR |
|---|---|---|---|---|
| **CORAL-S** | seen typologies | **0.708 ± 0.026** | **0.774** | **0.879** |
| RF | seen typologies | 0.541 | 0.596 | 0.631 |
| GBDT | seen typologies | 0.514 | 0.558 | 0.686 |
| Graph-GBDT | seen typologies | 0.499 | 0.547 | 0.634 |
| **CORAL-U** | **none** | **0.173 ± 0.011** | **0.445** | **0.495** |
| IsolationForest | none | 0.034 | 0.062 | 0.024 |

5 seeds; baselines are deterministic. CORAL-S is the best supervised method on 6 of 7 typologies (RF wins cycle). CORAL-U is the best label-free method on 6 of 7.

Honest findings:
- **Rings + flow carry the label-free signal**: closure-only rings give 0.107, flow factors give 0.173, no rings gives 0.028.
- **The learned co-activity predictor (CPP) helps CORAL-S** (0.720 vs 0.664 with a "never interacted" heuristic) but **not CORAL-U**, where the heuristic scores 0.252.
- **Drift stress test** (`--drift`): the gate detects the change (graph weight 0.9 → 0.1) and keeps CORAL-U from collapsing (0.028 vs 0.022 ungated; recall 0.097 vs 0.030 for IsolationForest). But CORAL-S drops to 0.518, below RF (0.561); a gated CORAL-S helped on dev and not on held-out, so it is not adopted.
- **GNN encoder** (`--encoder gnn`): CORAL-S 0.659, CORAL-U 0.167, slightly below SIGN.

Reproduce: `results/final/` holds every run (`results.csv`, gate logs, per-account scores). `scripts/fill_paper.py results/final` regenerates the paper's results section, and every number in it is computed from these CSVs.
