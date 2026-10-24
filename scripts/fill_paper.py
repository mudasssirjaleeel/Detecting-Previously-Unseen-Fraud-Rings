#!/usr/bin/env python3
"""Fill the paper's results section from result CSVs. Every number in the
generated LaTeX is computed here; nothing is typed by hand.
Usage: python scripts/fill_paper.py results/final > ../paper/results_section.tex"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "results/final"
COLS = ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr", "seen_auprc"]


def load(name):
    return pd.read_csv(f"{D}/{name}/results.csv")


def agg(r):
    m = r.groupby("method")[COLS].mean()
    sd = r.groupby(["method", "seed"])[COLS].mean().groupby("method").std().fillna(0)
    return m, sd


def f3(x):
    return f"{x:.3f}"


def cell(m, sd, meth, c, bold=False):
    v = f3(m.loc[meth, c])
    if bold:
        v = r"\textbf{" + v + "}"
    if sd.loc[meth, c] > 0:
        v += r"$\pm$" + f"{sd.loc[meth, c]:.3f}".lstrip("0")
    return v


NAME = {"CORAL-S": r"\coral{}-S", "CORAL-U": r"\coral{}-U", "IsolationForest": "Isolation Forest",
        "CORAL-S (gated)": r"\coral{}-S (gated)"}


def table(r, label, caption, sup, uns):
    m, sd = agg(r)
    out = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
           r"\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccc}", r"\toprule",
           r"Method & Labels & AUPRC & ROC-AUC & Rec@1\% & Ring rec@1\% & Seen AUPRC \\", r"\midrule"]
    for grp, lab in ((sup, "seen"), (uns, "none")):
        best = {c: m.loc[grp, c].idxmax() for c in COLS}
        for meth in grp:
            out.append(f"{NAME.get(meth, meth)} & {lab} & " +
                       " & ".join(cell(m, sd, meth, c, best[c] == meth) for c in COLS) + r" \\")
        if grp is sup:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def ablation_rows(r, V, cols=("auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr")):
    m, sd = agg(r)
    return "\n".join(f"{lab} & " + " & ".join(cell(m, sd, v, c) for c in cols) + r" \\"
                     for v, lab in V if v in m.index)


def main():
    N = load("normal")            # 5 seeds, held-out world
    Dr = load("drift_v2")         # 3 seeds, held-out world + drift, incl. gated CORAL-S
    C = load("cpp_ablation")      # 3 seeds, held-out world, CORAL-S with heuristic predictor
    G = load("gnn")               # 3 seeds, GNN encoder
    mN, sN = agg(N)
    mD, sD = agg(Dr)
    mC, _ = agg(C)
    mG, sG = agg(G)
    br = N.groupby("heldout")["base_rate"].first().mean()
    pN = N.pivot_table(index="heldout", columns="method", values="auprc")
    nwin_s = int((pN["CORAL-S"] > pN[["RF", "GBDT", "Graph-GBDT"]].max(axis=1)).sum())
    nwin_u = int((pN["CORAL-U"] > pN["IsolationForest"]).sum())
    losers_s = [h.title() for h in pN.index if pN.loc[h, "CORAL-S"] <= pN.loc[h, ["RF", "GBDT", "Graph-GBDT"]].max()]
    gate = pd.concat([pd.read_csv(f"{D}/drift_v2/gate_seed{s}.csv") for s in range(3)])
    ga = gate.groupby("window")["alpha"].mean()
    gr = gate.groupby("window")["reliability"].mean()
    x = {}
    x["S"], x["RF"], x["GB"], x["GG"] = (f3(mN.loc[k, "auprc"]) for k in ["CORAL-S", "RF", "GBDT", "Graph-GBDT"])
    x["Srec"], x["GBrec"] = f3(mN.loc["CORAL-S", "recall@1%fpr"]), f3(mN.loc["GBDT", "recall@1%fpr"])
    x["Sring"], x["GBring"], x["RFring"] = (f3(mN.loc[k, "ring_recall@1%fpr"]) for k in ["CORAL-S", "GBDT", "RF"])
    x["Sseen"], x["RFseen"] = f3(mN.loc["CORAL-S", "seen_auprc"]), f3(mN.loc["RF", "seen_auprc"])
    x["U"], x["IF"] = f3(mN.loc["CORAL-U", "auprc"]), f3(mN.loc["IsolationForest", "auprc"])
    x["U_ring"] = f3(mN.loc["CORAL-U", "ring_recall@1%fpr"])
    x["Urec"], x["IFrec"] = f3(mN.loc["CORAL-U", "recall@1%fpr"]), f3(mN.loc["IsolationForest", "recall@1%fpr"])
    x["Uring"], x["IFring"] = f3(mN.loc["CORAL-U", "ring_recall@1%fpr"]), f3(mN.loc["IsolationForest", "ring_recall@1%fpr"])
    x["Ux"] = f"{mN.loc['CORAL-U', 'auprc'] / br:.0f}"
    x["UIFx"] = f"{mN.loc['CORAL-U', 'auprc'] / mN.loc['IsolationForest', 'auprc']:.1f}"
    x["br"] = f"{br:.4f}"
    x["nwin_s"], x["nwin_u"], x["losers"] = nwin_s, nwin_u, ", ".join(losers_s)
    for k, v in {"noflow": "CORAL-U (no flow: closure only)", "nocv": "CORAL-U (no amount coherence)",
                 "norings": "CORAL-U (no rings)", "gonly": "CORAL-U (graph only)",
                 "heur": "CORAL-U (heuristic, no CPP)", "nogate": "CORAL-U (no gate)"}.items():
        x[k] = f3(mN.loc[v, "auprc"])
        x[k + "_ring"] = f3(mN.loc[v, "ring_recall@1%fpr"])
    x["heur_roc"], x["U_roc"] = f3(mN.loc["CORAL-U (heuristic, no CPP)", "roc_auc"]), f3(mN.loc["CORAL-U", "roc_auc"])
    x["S3"], x["Sh"] = f3(mC.loc["CORAL-S", "auprc"]), f3(mC.loc["CORAL-S (heuristic, no CPP)", "auprc"])
    x["S3ring"], x["Shring"] = f3(mC.loc["CORAL-S", "ring_recall@1%fpr"]), f3(mC.loc["CORAL-S (heuristic, no CPP)", "ring_recall@1%fpr"])
    x["GS"], x["GU"] = f3(mG.loc["CORAL-S", "auprc"]), f3(mG.loc["CORAL-U", "auprc"])
    x["GSsd"], x["GUsd"] = f"{sG.loc['CORAL-S', 'auprc']:.3f}", f"{sG.loc['CORAL-U', 'auprc']:.3f}"
    for k, v in {"dS": "CORAL-S", "dSg": "CORAL-S (gated)", "dRF": "RF", "dGB": "GBDT", "dU": "CORAL-U",
                 "dUng": "CORAL-U (no gate)", "dIF": "IsolationForest", "dheur": "CORAL-U (heuristic, no CPP)",
                 "dgonly": "CORAL-U (graph only)"}.items():
        x[k] = f3(mD.loc[v, "auprc"])
        x[k + "_rec"] = f3(mD.loc[v, "recall@1%fpr"])
        x[k + "_ring"] = f3(mD.loc[v, "ring_recall@1%fpr"])
    x["a_first"], x["a_last"] = f"{ga.iloc[:3].mean():.2f}", f"{ga.iloc[-3:].mean():.2f}"
    x["r_first"], x["r_last"] = f"{gr.iloc[:3].mean():.1f}", f"{gr.iloc[-3:].mean():.1f}"
    x["a_min"] = f"{ga.min():.2f}"
    gn = pd.concat([pd.read_csv(f"{D}/normal/gate_seed{s}.csv") for s in range(5)])
    assert (gn["alpha"] >= 0.8999).all(), "gate fired in a normal window; revise the text"
    x["n_gate_windows"] = len(gn)

    main_tab = table(N, "tab:main",
                     "Leave-One-Typology-Out results on the held-out world, averaged over seven held-out "
                     r"typologies (mean $\pm$ std over five seeds). ``Labels'' refers to labels of the "
                     r"\emph{seen} typologies only; no method ever sees the held-out one. Bold: best within "
                     f"each group. Baselines are deterministic, so their std is zero. Mean held-out base rate {x['br']}.",
                     ["CORAL-S", "RF", "GBDT", "Graph-GBDT"], ["CORAL-U", "IsolationForest"])
    drift_tab = table(Dr, "tab:drift",
                      r"Drift stress test on the held-out world (mean $\pm$ std over three seeds): a bill-splitting "
                      "product makes legitimate behaviour ring-like throughout the test period.",
                      ["CORAL-S (gated)", "CORAL-S", "RF", "GBDT", "Graph-GBDT"], ["CORAL-U", "IsolationForest"])
    # per typology
    meths = ["CORAL-S", "RF", "GBDT", "Graph-GBDT", "CORAL-U", "IsolationForest"]
    n = N.groupby("heldout")[["n_pos", "base_rate"]].first()
    rows = []
    for h in pN.index:
        b1 = pN.loc[h, meths[:4]].idxmax(); b2 = pN.loc[h, meths[4:]].idxmax()
        vals = [(r"\textbf{%.3f}" if mm in (b1, b2) else "%.3f") % pN.loc[h, mm] for mm in meths]
        rows.append(f"{h.title()} & {int(n.loc[h, 'n_pos'])} & {n.loc[h, 'base_rate']:.4f} & " + " & ".join(vals) + r" \\")
    short = {"GATHER-SCATTER": "Gather-sc.", "SCATTER-GATHER": "Scatter-g."}
    def coords(mm):
        return " ".join(f"({short.get(h, h.title())},{pN.loc[h, mm]:.3f})" for h in pN.index)
    sym = ",".join(short.get(h, h.title()) for h in pN.index)
    V = [("CORAL-U", r"\coral{}-U (full)"), ("CORAL-U (no gate)", r"\quad no gate"),
         ("CORAL-U (no amount coherence)", r"\quad no amount coherence ($\nu$)"),
         ("CORAL-U (no flow: closure only)", r"\quad no flow factors (closure only)"),
         ("CORAL-U (no rings)", r"\quad no rings"),
         ("CORAL-U (graph only)", r"\quad graph only (no feature fallback)"),
         ("CORAL-U (heuristic, no CPP)", r"\quad heuristic predictor instead of CPP"),
         ("IsolationForest", "Isolation Forest (reference)")]
    abl = ablation_rows(N, V)
    T = open(__file__.replace("fill_paper.py", "results_template.tex")).read()
    T = T.replace("<<MAIN_TABLE>>", main_tab).replace("<<DRIFT_TABLE>>", drift_tab)
    T = T.replace("<<PERTYP_ROWS>>", "\n".join(rows)).replace("<<ABLATION_ROWS>>", abl)
    T = T.replace("<<SYMS>>", sym)
    for mm, key in [("CORAL-S", "S"), ("RF", "RF"), ("GBDT", "GB"), ("CORAL-U", "U")]:
        T = T.replace(f"<<COORDS_{key}>>", coords(mm))
    for k, v in x.items():
        T = T.replace("{{" + k + "}}", str(v))
    assert "{{" not in T and "<<" not in T, "unfilled placeholder: " + T[T.find("{{") - 40:T.find("{{") + 40]
    print(T)


if __name__ == "__main__":
    main()
