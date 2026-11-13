#!/usr/bin/env python3
"""Regenerate value-equivalent Figures 7.1 and 7.2 from archival CSV.

Series = max verified q=240 L1-loss per (m,n,mechanism), among all K/restarts.
For the original image bytes see results/full_hill_m4.png and full_hill_m3.png.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / 'results' / 'q240_refinement.csv')
figdir = HERE / 'figures_reproduced'
figdir.mkdir(exist_ok=True)
styles = [('im','o-','Independent Markets'),('ladder','s--','Ladder'),
          ('utilprop','D-.','UtilProp'),('piecewise','^:','Piecewise Uniform')]
for m in (4, 3):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sub = raw.loc[raw.m.eq(m)]
    for mech, style, label in styles:
        rows = (sub.loc[sub.mechanism.eq(mech)]
                .groupby('n', as_index=False).loss_q240.max().sort_values('n'))
        ax.plot(rows.n, rows.loss_q240, style, label=label)
    ax.axhline(1-1/m, color='grey', ls='--', lw=.9, label=f'1−1/m = {1-1/m:.3f}')
    if m == 4:
        ax.axhline(1, color='grey', ls=':', lw=.9, label='L1-loss = 1')
    ax.set_xscale('log')
    ax.set_xticks([4,6,10,20,50,100,1000])
    ax.set_xticklabels(['4','6','10','20','50','100','1000'])
    ax.set_xlabel('Number of voters (n)')
    ax.set_ylabel('Best observed L1-loss')
    ax.set_title(f'Weighted-voter hill climbing · m={m}, 20 restarts × 3000 steps')
    ax.grid(alpha=.2)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    destination = figdir / f'figure_7_{1 if m == 4 else 2}.png'
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(destination)
