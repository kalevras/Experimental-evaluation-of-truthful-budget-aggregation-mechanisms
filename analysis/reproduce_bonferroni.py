#!/usr/bin/env python3
"""Bonferroni paired t intervals from archived R=50 matched-profile raw data.

Each family: fixed experimental dataset, culture, metric, all 15 mechanisms pairs.
Familywise alpha=0.05; this does NOT adjust over metrics or cultures.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from statistics import mean, stdev
import csv
import json
from scipy.stats import t

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'supplementary'
MECHS=('Mean Rule','Util','Independent Markets','Ladder','UtilProp','Piecewise Uniform')
F_METRICS=('avg_utility','utility_std','min_utility','p10_utility','utility_gini','l1_loss')
R_METRICS=('allocation_l1_shift','avg_utility_abs_change','l1_loss_abs_change','gini_abs_change','social_welfare_abs_change')

def compute(label, source, metrics, eps=None):
    """Recompute paired Student-t confidence intervals with Bonferroni adjustment."""
    by=defaultdict(dict)
    with open(source,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if eps is not None and abs(float(r['epsilon'])-eps)>1e-12:continue
            k=(r['culture'],int(r['repetition']))
            if r['mechanism'] in by[k]:raise AssertionError(('duplicate',k,r['mechanism']))
            by[k][r['mechanism']]=r
    cultures=sorted(set(c for c,_ in by))
    out=[]
    for culture in cultures:
        reps=sorted(z for c,z in by if c==culture)
        assert len(reps)==50,(label,culture,len(reps))
        assert all(set(by[culture,r])==set(MECHS) for r in reps)
        for metric in metrics:
            for a,b in combinations(MECHS,2):
                ds=[float(by[culture,r][a][metric])-float(by[culture,r][b][metric]) for r in reps]
                n=len(ds); avg=mean(ds); sd=stdev(ds)
                tcrit=float(t.ppf(1-0.05/(2*15),n-1))
                margin=tcrit*sd/(n**0.5)
                lo,hi=avg-margin,avg+margin
                judgement='+' if lo>0 else '-' if hi<0 else '='
                out.append({'dataset':label,'culture':culture,'metric':metric,'mechanism_A':a,'mechanism_B':b,
                   'repetitions':n,'mean_A_minus_B':avg,'sd_of_paired_difference':sd,
                   't_critical_bonferroni':tcrit,'ci_lower':lo,'ci_upper':hi,
                   'decision':judgement, 'family_alpha':0.05,'pairs_in_family':15})
    name=OUT/f'bonferroni_{label}_R50.csv'
    with name.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    return {'file':name.name,'rows':len(out),'cultures':len(cultures),'metrics':len(metrics),
            'R':50,'decisions':{ch:sum(r['decision']==ch for r in out) for ch in '+-='},
            't_critical':out[0]['t_critical_bonferroni']}

if __name__=='__main__':
    results=[compute('fairness',BASE/'data/fairness/fairness_detailed.csv',F_METRICS),
    compute('robustness_eps_010',BASE/'data/robustness/robustness_detailed.csv',R_METRICS,.1)]
    (OUT/'bonferroni_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
