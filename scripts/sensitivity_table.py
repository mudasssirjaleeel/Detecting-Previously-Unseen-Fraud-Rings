#!/usr/bin/env python3
"""LaTeX sensitivity table from results/sensitivity/* and results/final/normal (seed 0)."""
import pandas as pd

base = pd.read_csv("results/final/normal/results.csv")
runs = [(r"$\tau=1$\,h, $p_{\max}=0.5$ (default)", base[base.seed == 0]),
        (r"$\tau=30$\,min", pd.read_csv("results/sensitivity/tau30/results.csv")),
        (r"$\tau=2$\,h", pd.read_csv("results/sensitivity/tau120/results.csv")),
        (r"$p_{\max}=0.3$", pd.read_csv("results/sensitivity/pmax0.3/results.csv")),
        (r"$p_{\max}=0.7$", pd.read_csv("results/sensitivity/pmax0.7/results.csv"))]
rows = []
for lab, r in runs:
    m = r.groupby("method")[["auprc", "ring_recall@1%fpr"]].mean()
    rows.append(f"{lab} & {m.loc['CORAL-S','auprc']:.3f} & {m.loc['CORAL-S','ring_recall@1%fpr']:.3f} & "
                f"{m.loc['CORAL-U','auprc']:.3f} & {m.loc['CORAL-U','ring_recall@1%fpr']:.3f} \\\\")
rf = runs[0][1].groupby("method")["auprc"].mean()["RF"]
print(r"""\begin{table}[t]
\centering
\caption{Sensitivity to the synchronisation tolerance $\tau$ and the surprise threshold $p_{\max}$ on the held-out world (one seed, averaged over seven held-out typologies). Other settings at their defaults. For reference, RF reaches %.3f AUPRC and does not depend on these settings.}
\label{tab:sens}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\coral{}-S} & \multicolumn{2}{c}{\coral{}-U} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Setting & AUPRC & Ring rec@1\%% & AUPRC & Ring rec@1\%% \\
\midrule
""" % rf + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")
