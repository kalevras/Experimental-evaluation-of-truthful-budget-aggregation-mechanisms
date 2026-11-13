"""Rebuild fairness and robustness visualizations from saved CSV files."""

import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
work=str(ROOT/'reproduced_figures'/'chapters_5_6');os.makedirs(work,exist_ok=True)
f=pd.read_csv(str(ROOT/'data'/'fairness'/'fairness_detailed.csv'))
r=pd.read_csv(str(ROOT/'data'/'robustness'/'robustness_detailed.csv'))
cc=['random','balanced','concentrated','single_minded','polarized','majority_minority','mixed','sparse'];mechs=['Mean Rule','Util','Independent Markets','Ladder','UtilProp','Piecewise Uniform'];colors=dict(zip(mechs,plt.get_cmap('tab10').colors[:6]));
f['SW_ratio']=f['avg_utility']/f[f.mechanism.eq('Util')].set_index(['culture','repetition'])['avg_utility'].reindex(f.set_index(['culture','repetition']).index).to_numpy()
def ci(x):
 """Compute the Student-t interval from independent profile values."""
 x=np.array(x,dtype=float);x=x[np.isfinite(x)];avg=x.mean();err=t.ppf(.975,len(x)-1)*x.std(ddof=1)/len(x)**.5 if len(x)>1 else 0;return avg,max(avg-err,0),avg+err

def base(ax,title='',ylabel='',xlabel=''):
 """Apply consistent axes, title and grid styling to a figure."""
 ax.set_title(title,fontsize=10); ax.grid(alpha=.16,axis='y');ax.set_ylabel(ylabel,fontsize=9);ax.set_xlabel(xlabel,fontsize=9)
def catplot(ax,df,metric,title='',legend=False):
 """Draw culture-specific means with confidence-range shading."""
 for j,m in enumerate(mechs):
  x=np.arange(len(cc))+(j-2.5)*.115
  dat=[df[(df.culture==c)&(df.mechanism==m)][metric] for c in cc]
  vals=np.array([ci(z) for z in dat]);means=vals[:,0];bounds=vals[:,1:]
  # Individual CI as subtle shaded vertical rectangles around markers, not error bars; no invented continuity among categories
  ax.scatter(x,means,c=[colors[m]],s=24,label=m,zorder=5)
  for xx,lo,hi in zip(x,bounds[:,0],bounds[:,1]):
   if hi>lo:ax.fill_between([xx-.045,xx+.045],[lo,lo],[hi,hi],color=colors[m],alpha=.22,linewidth=0,zorder=2)
 ax.set_xticks(np.arange(len(cc)),cc,rotation=35,ha='right',fontsize=8);base(ax,title)
 if legend:ax.legend(ncol=3,fontsize=7,loc='best')

def save(fig,name):
 """Save the generated figure to its reproducible target filename."""
 fig.savefig(os.path.join(work,name),dpi=170,bbox_inches='tight');plt.close(fig)

for num,metric,title in [(10,'min_utility','Mean minimum utility'),(12,'avg_utility','Mean average utility'),(13,'l1_loss','Mean L1-loss'),(14,'p10_utility','Mean P10 utility'),(15,'SW_ratio','Mean SW/SW*'),(21,'utility_gini','Mean Gini')]:
 fig,ax=plt.subplots(figsize=(11,5.8));catplot(ax,f,metric,title+' · n=100, m=3, R=50, seed=456',True);fig.tight_layout();save(fig,f'image{num}.png')
fig,axs=plt.subplots(2,3,figsize=(12,7));metrics=['SW_ratio','avg_utility','min_utility','l1_loss','p10_utility'];titles=['SW/SW*','Average utility','Minimum utility','L1-loss','P10 utility'];
for ax,metric,title in zip(axs.flat,metrics,titles):catplot(ax,f,metric,title,False)
axs.flat[-1].axis('off');handles=[plt.Line2D([0],[0],color=colors[m],marker='o',linestyle='None',label=m) for m in mechs];fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9);fig.suptitle('Fairness & welfare by culture · n=100, m=3, R=50, seed=456');fig.tight_layout(rect=[0,.07,1,.94]);save(fig,'image11.png')
# 6.1 multi culture small multiples
fig,axs=plt.subplots(2,4,figsize=(13,6.5),sharey=True)
for ax,c in zip(axs.flat,cc):
 for m in mechs:
  subset=r[(r.culture==c)&(r.mechanism==m)]
  vals=subset.groupby('epsilon').allocation_l1_shift.apply(lambda z:ci(z)).sort_index();x=np.array(vals.index);y=np.array([a[0] for a in vals]);lo=np.array([a[1] for a in vals]);hi=np.array([a[2] for a in vals]);ax.plot(x,y,'o-',ms=3,lw=1,color=colors[m]);ax.fill_between(x,lo,hi,color=colors[m],alpha=.12)
 base(ax,c,xlabel='ε')
fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8);fig.suptitle('Allocation L1 shift by culture · n=100, m=10, R=50, seed=789');fig.tight_layout(rect=[0,.08,1,.95]);save(fig,'image16.png')
fix=r[np.isclose(r.epsilon,.1)].copy()
for num,metric,title in [(17,'allocation_l1_shift','Mean allocation L1 shift'),(19,'avg_utility_abs_change','Mean |Δ average utility|'),(20,'gini_abs_change','Mean |Δ Gini|')]:
 fig,ax=plt.subplots(figsize=(11,5.8));catplot(ax,fix,metric,title+' · ε=0.10, n=100, m=10, R=50',True);fig.tight_layout();save(fig,f'image{num}.png')
# paired differences 6.3, exactly matched by culture,repetition,epsilon
pairs=[('Mean Rule','Independent Markets'),('Independent Markets','Ladder'),('Ladder','UtilProp'),('UtilProp','Piecewise Uniform'),('Mean Rule','Util')]
fig,ax=plt.subplots(figsize=(11,5.8));pivot=fix.pivot_table(index=['culture','repetition'],columns='mechanism',values='allocation_l1_shift')
for j,(a,b) in enumerate(pairs):
 xs=np.arange(len(cc))+(j-2)*.13
 for x,c in zip(xs,cc):
  s=pivot.loc[c,a]-pivot.loc[c,b];v,lo,hi=ci(s);ax.scatter([x],[v],s=22,color=plt.get_cmap('tab10')(j),zorder=3);ax.fill_between([x-.045,x+.045],[lo,lo],[hi,hi],alpha=.25,color=plt.get_cmap('tab10')(j))
ax.axhline(0,ls='--',lw=.9,color='grey');ax.set_xticks(np.arange(len(cc)),cc,rotation=35,ha='right',fontsize=8);base(ax,'Paired allocation L1 shift differences · ε=0.10, R=50');ax.legend([plt.Line2D([0],[0],marker='o',linestyle='None',color=plt.get_cmap('tab10')(j)) for j in range(len(pairs))],[f'{a} − {b}' for a,b in pairs],fontsize=7,ncol=2);fig.tight_layout();save(fig,'image18.png')
# report equality and accuracy
records=[]
for metric in ['min_utility','p10_utility','l1_loss','avg_utility','SW_ratio']:
 for c in cc:
  d=f[(f.culture==c)].groupby('mechanism')[metric].mean();pairs_equal=[(a,b) for i,a in enumerate(mechs) for b in mechs[i+1:] if abs(d[a]-d[b])<1e-9]
  if pairs_equal:records.append({'metric':metric,'culture':c,'equal_pairs':'; '.join('/'.join(x) for x in pairs_equal)})
pd.DataFrame(records).to_csv(os.path.join(work,'confirmed_mean_equalities.csv'),index=False)
# verify saved metrics
for grp,metric_list,df in [('fairness',['min_utility','avg_utility','p10_utility','l1_loss'],f),('robustness',['allocation_l1_shift','avg_utility_abs_change','gini_abs_change'],r)]:
 summary=df.groupby(['culture','mechanism']+(['epsilon'] if grp=='robustness' else []))[metric_list].mean();assert len(summary)>0


print("Επαναδημιουργήθηκαν τα γραφήματα image10.png έως image20.png στο figures/")
