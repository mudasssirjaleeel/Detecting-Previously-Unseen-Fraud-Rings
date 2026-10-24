#!/usr/bin/env python3
"""Generate the AMLSim results subsection (tables + prose numbers) from results/final/amlsim."""
import pandas as pd

R = pd.read_csv("results/final/amlsim/results.csv")
COLS = ["auprc", "roc_auc", "recall@1%fpr", "ring_recall@1%fpr", "seen_auprc"]
m = R.groupby("method")[COLS].mean()
sd = R.groupby(["method", "seed"])[COLS].mean().groupby("method").std().fillna(0)
p = R.pivot_table(index="heldout", columns="method", values="auprc")
n = R.groupby("heldout")[["n_pos", "base_rate"]].first()
gate = pd.concat([pd.read_csv(f"results/final/amlsim/gate_seed{s}.csv") for s in range(3)])
NAME = {"CORAL-S": r"\coral{}-S", "CORAL-U": r"\coral{}-U", "IsolationForest": "Isolation Forest"}
f3 = lambda v: f"{v:.3f}"  # noqa: E731


def cell(meth, c, bold):
    v = f3(m.loc[meth, c])
    v = r"\textbf{" + v + "}" if bold else v
    return v + (r"$\pm$" + f"{sd.loc[meth, c]:.3f}".lstrip("0") if sd.loc[meth, c] >= 0.0005 else "")


rows = []
for grp, lab in ((["CORAL-S", "GBDT", "Graph-GBDT", "RF"], "seen"), (["CORAL-U", "IsolationForest"], "none")):
    best = {c: m.loc[grp, c].idxmax() for c in COLS}
    for meth in grp:
        rows.append(f"{NAME.get(meth, meth)} & {lab} & " + " & ".join(cell(meth, c, best[c] == meth) for c in COLS) + r" \\")
    if lab == "seen":
        rows.append(r"\midrule")
meths = ["CORAL-S", "RF", "GBDT", "Graph-GBDT", "CORAL-U", "IsolationForest"]
prow = []
for h in p.index:
    b1 = p.loc[h, meths[:4]].idxmax(); b2 = p.loc[h, meths[4:]].idxmax()
    vals = [(r"\textbf{%.3f}" if mm in (b1, b2) else "%.3f") % p.loc[h, mm] for mm in meths]
    prow.append(f"{h.title()} & {int(n.loc[h, 'n_pos'])} & {n.loc[h, 'base_rate']:.4f} & " + " & ".join(vals) + r" \\")
wins = int((p["CORAL-S"] > p[["RF", "GBDT", "Graph-GBDT"]].max(axis=1)).sum())
losses = [h.title() for h in p.index if p.loc[h, "CORAL-S"] <= p.loc[h, ["RF", "GBDT", "Graph-GBDT"]].max()]
uwin = int((p["CORAL-U"] > p["IsolationForest"]).sum())
X = dict(S=f3(m.loc["CORAL-S", "auprc"]), Ssd=f"{sd.loc['CORAL-S','auprc']:.3f}", GB=f3(m.loc["GBDT", "auprc"]),
         GG=f3(m.loc["Graph-GBDT", "auprc"]), RF=f3(m.loc["RF", "auprc"]),
         Srec=f3(m.loc["CORAL-S", "recall@1%fpr"]), GBrec=f3(m.loc["GBDT", "recall@1%fpr"]),
         Sring=f3(m.loc["CORAL-S", "ring_recall@1%fpr"]), GBring=f3(m.loc["GBDT", "ring_recall@1%fpr"]),
         RFring=f3(m.loc["RF", "ring_recall@1%fpr"]),
         Sseen=f3(m.loc["CORAL-S", "seen_auprc"]), GBseen=f3(m.loc["GBDT", "seen_auprc"]),
         U=f3(m.loc["CORAL-U", "auprc"]), Ung=f3(m.loc["CORAL-U (no gate)", "auprc"]), IF=f3(m.loc["IsolationForest", "auprc"]),
         Uroc=f3(m.loc["CORAL-U", "roc_auc"]), IFroc=f3(m.loc["IsolationForest", "roc_auc"]),
         heur=f3(m.loc["CORAL-U (heuristic, no CPP)", "auprc"]),
         wins=wins, losses=", ".join(losses), uwin=uwin, ntyp=len(p.index),
         gfrac=f"{(gate.alpha < 0.8999).mean() * 100:.0f}", amean=f"{gate.alpha.mean():.2f}",
         delta=f"{m.loc['CORAL-S','auprc'] - m.loc['GBDT','auprc']:.3f}")

T = r"""\subsection{IBM AMLSim}
\label{sec:amlsim}

\begin{table}[t]
\centering
\caption{Leave-One-Typology-Out results on the held-out IBM AMLSim world (mean $\pm$ std over three seeds, averaged over eight held-out typologies). Bold: best within each group.}
\label{tab:amlsim}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
Method & Labels & AUPRC & ROC-AUC & Rec@1\% & Ring rec@1\% & Seen AUPRC \\
\midrule
<<ROWS>>
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{AUPRC per held-out typology on the held-out AMLSim world (mean over three seeds). Bold: best supervised and best label-free method per row.}
\label{tab:amlsim-typ}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrccccccc}
\toprule
Held-out typology & $n^+$ & Base rate & \coral{}-S & RF & GBDT & Graph-GBDT & \coral{}-U & IForest \\
\midrule
<<PROWS>>
\bottomrule
\end{tabular}
\end{table}

Tables~\ref{tab:amlsim} and~\ref{tab:amlsim-typ} report the held-out AMLSim world. The configuration (weekly windows, $\tau=1$ day, 3-day relays) was chosen on the AMLSim development world, where relays were marginally better than no relays, and then frozen; all other settings are those of the synthetic benchmark.

\paragraph{The supervised gain transfers, but it is small}
\coral{}-S is again the best method on AUPRC ({{S}}$\pm${{Ssd}} vs.\ {{GB}} for GBDT, {{GG}} for Graph-GBDT and {{RF}} for RF), with higher recall at 1\% FPR ({{Srec}} vs.\ {{GBrec}}) and higher AUPRC on the seen typologies ({{Sseen}} vs.\ {{GBseen}}). The margin over GBDT, however, is only {{delta}} AUPRC, against 0.194 on the synthetic benchmark. \coral{}-S is the best supervised method on {{wins}} of {{ntyp}} typologies, and the exceptions are {{losses}}. On ring recall, GBDT and RF ({{GBring}} and {{RFring}}) are slightly ahead of \coral{}-S ({{Sring}}).

\paragraph{The label-free signal does not transfer}
\coral{}-U reaches {{U}} AUPRC ({{Ung}} without the gate), no better than Isolation Forest ({{IF}}), and it beats Isolation Forest on only {{uwin}} of {{ntyp}} typologies. The reason is visible in how AMLSim builds rings. Its alert members are existing accounts that keep their normal activity, and an alert unfolds over days to weeks rather than in a burst. A group of laundering accounts is therefore neither mutually new nor tightly synchronised, which are the two properties that closure and flow detect. The learned predictor does help here, unlike on the synthetic benchmark ({{U}} vs.\ {{heur}} for the heuristic), but not enough to make structure-only detection competitive. The gate also left its dead zone in {{gfrac}}\% of test windows although no drift was injected, which cost a little AUPRC.

\paragraph{What the two benchmarks say together}
The synthetic benchmark models rings of partly fresh accounts acting in bursts; AMLSim models rings of established accounts acting slowly. \coral{}'s relational signals add most when rings look like the former, and they still add a little to a supervised detector when rings look like the latter. Label-free detection of slow rings built from established accounts remains an open problem.
"""
T = T.replace("<<ROWS>>", "\n".join(rows)).replace("<<PROWS>>", "\n".join(prow))
for k, v in X.items():
    T = T.replace("{{" + k + "}}", str(v))
assert "{{" not in T and "<<" not in T
print(T)
