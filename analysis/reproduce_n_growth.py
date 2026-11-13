"""Regenerate Figures 5.1–5.8 from original per-profile n-growth runs.

Reads 15,000 source evaluations and validates all stored Student-t 95% CIs
before plotting. Figure order is mapped explicitly in FIGURES below.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'n_growth'
OUT = ROOT / 'reproduced_figures' / 'chapter_5'
OUT.mkdir(parents=True, exist_ok=True)
MECHS = ['Mean Rule','Util','Independent Markets','Ladder','UtilProp','Piecewise Uniform']
FIGURES = [(5.1,2,'sw_ratio'),(5.2,3,'sw_ratio'),(5.3,10,'sw_ratio'),
           (5.4,100,'sw_ratio'),(5.5,10,'l1_loss'),(5.6,100,'l1_loss'),
           (5.7,10,'welfare_gap'),(5.8,100,'welfare_gap')]
raw = pd.read_csv(SOURCE / 'detailed_results_full.csv')
check = pd.read_csv(SOURCE / 'student_t_95_ci_full.csv')
assert len(raw) == 15000 and len(check) == 720
# Independent validation of stored point means/CI from per-profile results.
for (m,n,mechanism,metric), group in check.groupby(['m','n','mechanism','metric']):
    if metric not in ('sw_ratio', 'l1_loss'):
        continue
    sample = raw[(raw.m == m) & (raw.n == n) & (raw.mechanism == mechanism)][metric]
    if len(sample) < 2:
        raise AssertionError((m,n,mechanism))
    x = sample.to_numpy(dtype=float)
    mean = x.mean()
    half = t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    stored = group.iloc[0]
    assert abs(stored['mean']-mean) < 1e-9, (m,n,mechanism,metric)
    assert abs(stored['ci_low']-(mean-half)) < 1e-9
    assert abs(stored['ci_high']-(mean+half)) < 1e-9

for number,m,metric in FIGURES:
    fig,ax=plt.subplots(figsize=(10,5.4))
    base = 'sw_ratio' if metric == 'welfare_gap' else metric
    subset=check[(check.m == m)&(check.metric == base)]
    for mechanism in MECHS:
        df=subset[subset.mechanism == mechanism].sort_values('n')
        x=df.n.to_numpy()
        if metric == 'welfare_gap':
            y=1-df['mean'].to_numpy(); lower=1-df['ci_high'].to_numpy(); upper=1-df['ci_low'].to_numpy()
        else:
            y=df['mean'].to_numpy();lower=df['ci_low'].to_numpy();upper=df['ci_high'].to_numpy()
        ax.plot(x,y,marker='o',label=mechanism,markersize=3.5,linewidth=1.2)
        color=ax.lines[-1].get_color()
        ax.fill_between(x,lower,upper,alpha=.12,color=color)
    ax.set_title(f'Figure {number:.1f} · random culture, m={m} · Student-t 95% CI')
    ax.set_xlabel('Number of voters (n)')
    ax.set_ylabel({'sw_ratio':'SW/SW*','l1_loss':'L1-loss to mean','welfare_gap':'1−SW/SW*'}[metric])
    ax.grid(alpha=.18)
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout()
    target=OUT/f'figure_{str(number).replace(".","_")}.png'
    fig.savefig(target,dpi=150);plt.close(fig)
    print(target)
