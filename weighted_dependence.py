"""Output-weighted vs. mean individual dependence (Results, 'AI makes individuals independent').

Re-runs the P1 baseline (mutual consent) at alpha = 0 and alpha = 6 with the same seeds as run_all.py
and reports the mean of 1 - Y_alone/Y over agents and the output-weighted share sum(Y - Y_alone)/sum(Y).
Usage: python weighted_dependence.py   -> results/weighted_dependence.csv
"""
import os
import numpy as np
import pandas as pd
import run_all as ra

rows = []
for d_i, dom in enumerate(ra.DOMS3):
    for al in [0.0, 6.0]:
        for s in ra.seeds(ra.REPS, 10 + d_i):
            _, ag = ra.run_and_classify(dom, al, s, dict(transfers=False), keep_agents=True)
            rows.append(dict(domain=dom, alpha=al, seed=s,
                             mean_individual_dependence=ag.abhaengigkeit.mean(),
                             output_weighted_dependence=(ag.abhaengigkeit * ag.output).sum() / ag.output.sum()))
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv(os.path.join("results", "weighted_dependence.csv"), index=False)
print(df.groupby(["domain", "alpha"])[["mean_individual_dependence", "output_weighted_dependence"]].mean().round(3))
