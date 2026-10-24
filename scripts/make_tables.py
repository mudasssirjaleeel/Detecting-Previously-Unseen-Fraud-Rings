#!/usr/bin/env python3
"""Generate the paper's LaTeX tables directly from results CSVs (no hand-copying).
Usage: python scripts/make_tables.py results/final > ../paper/tables_generated.tex"""
import sys

import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "results/final"
COLS = ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr", "seen_auprc"]


def load(name):
    return pd.read_csv(f"{D}/{name}/results.csv")


def agg(r):
    m = r.groupby("method")[COLS].mean()
    sd = r.groupby(["method", "seed"])[COLS].mean().groupby("method").std().fillna(0)
    return m, sd


def cell(m, sd, meth, c, bold=False, show_sd=True):
    v = f"{m.loc[meth, c]:.3f}"
    if bold:
        v = r"\textbf{" + v + "}"
    s = sd.loc[meth, c]
    if show_sd and s > 0:
        v += r"$\pm$" + f"{s:.3f}".lstrip("0")
    return v


def main_table(r, label, caption):
    m, sd = agg(r)
    sup = ["CORAL-S", "RF", "GBDT", "Graph-GBDT"]
    uns = ["CORAL-U", "IsolationForest"]
    name = {"CORAL-S": r"\coral{}-S", "CORAL-U": r"\coral{}-U", "IsolationForest": "Isolation Forest"}
    out = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
           r"\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccc}", r"\toprule",
           r"Method & Labels & AUPRC & ROC-AUC & Rec@1\% & Ring rec@1\% & Seen AUPRC \\", r"\midrule"]
    for grp, lab in ((sup, "seen"), (uns, "none")):
        best = {c: m.loc[grp, c].idxmax() for c in COLS}
        for meth in grp:
            cells = [cell(m, sd, meth, c, best[c] == meth) for c in COLS]
            out.append(f"{name.get(meth, meth)} & {lab} & " + " & ".join(cells) + r" \\")
        if grp is sup:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def per_typ(r):
    p = r.pivot_table(index="heldout", columns="method", values="auprc")
    n = r.groupby("heldout")[["n_pos", "base_rate"]].first()
    meths = ["CORAL-S", "RF", "GBDT", "Graph-GBDT", "CORAL-U", "IsolationForest"]
    out = []
    for h in p.index:
        best = p.loc[h, meths[:4]].idxmax()
        bu = p.loc[h, meths[4:]].idxmax()
        vals = [(r"\textbf{%.3f}" if m in (best, bu) else "%.3f") % p.loc[h, m] for m in meths]
        out.append(f"{h.title()} & {int(n.loc[h, 'n_pos'])} & {n.loc[h, 'base_rate']:.4f} & " + " & ".join(vals) + r" \\")
    return "\n".join(out), p


def ablation(r, variants):
    m, sd = agg(r)
    rows = []
    for v, lab in variants:
        cells = [cell(m, sd, v, c) for c in ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr"]]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


if __name__ == "__main__":
    normal = load("normal")
    print("% ---- main table (held-out world) ----")
    print(main_table(normal, "tab:main",
                     "Leave-One-Typology-Out results on the held-out world, averaged over seven held-out "
                     r"typologies (mean $\pm$ std over five seeds). ``Labels'' refers to labels of the "
                     r"\emph{seen} typologies only; no method ever sees the held-out one. Bold: best within "
                     "each group. Baselines are deterministic, so their std is zero."))
    rows, p = per_typ(normal)
    print("\n% ---- per-typology rows ----\n" + rows)
    print("\n% ---- bar chart coordinates ----")
    for meth in ["CORAL-S", "RF", "GBDT", "CORAL-U"]:
        short = {"GATHER-SCATTER": "Gather-sc.", "SCATTER-GATHER": "Scatter-g."}
        co = " ".join(f"({short.get(h, h.title())},{p.loc[h, meth]:.3f})" for h in p.index)
        print(f"% {meth}: \\addplot coordinates {{{co}}};")
    V = [("CORAL-U", r"\coral{}-U (full)"), ("CORAL-U (no gate)", r"\quad no gate"),
         ("CORAL-U (no amount coherence)", r"\quad no amount coherence ($\nu$)"),
         ("CORAL-U (no flow: closure only)", r"\quad no flow factors (closure only)"),
         ("CORAL-U (no rings)", r"\quad no rings"), ("CORAL-U (graph only)", r"\quad graph only (no feature fallback)"),
         ("CORAL-U (heuristic, no CPP)", r"\quad heuristic predictor instead of CPP"),
         ("IsolationForest", "Isolation Forest (reference)")]
    print("\n% ---- ablation rows (held-out world) ----\n" + ablation(normal, V))
    for extra in ("drift", "gnn"):
        try:
            r = load(extra)
        except FileNotFoundError:
            continue
        m, sd = agg(r)
        print(f"\n% ---- {extra}: means ----")
        print((m.round(3).astype(str) + " +- " + sd.round(3).astype(str)).sort_values("auprc", ascending=False)
              [["auprc", "recall@1%fpr", "ring_recall@1%fpr"]].to_string())
        if extra == "drift":
            print("\n% ---- drift ablation rows ----\n" + ablation(r, V))
