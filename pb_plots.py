"""Plotting functions for controlled preference and runtime experiments."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

METRIC_INFO={
 'avg_utility':('Average Utility','Mean voter overlap utility','Higher values indicate greater average voter satisfaction.'),
 'min_utility':('Minimum Utility','Utility of the least-satisfied voter','Higher values indicate a better worst individual outcome.'),
 'p10_utility':('P10 Utility','10th percentile of voter utility','About 10% of voters are at or below this value; higher is better for the lower tail.'),
 'utility_std':('Utility Std. Dev.','Standard deviation of voter utility','Lower values indicate less dispersion; this is not a complete fairness criterion by itself.'),
 'utility_gini':('Utility Gini','Gini coefficient of voter utilities','0 means equal utilities; larger values indicate greater inequality.'),
 'l1_loss':('L1-loss to Mean','L1 distance from the coordinate-wise mean allocation','Lower values mean closer to the mean/proportional benchmark.'),
 'sw_ratio':('SW / Optimal SW','Social-welfare ratio','1.0 means the welfare optimum is attained.'),
 'welfare_gap_pct':('Welfare Gap (%)','Percentage welfare shortfall','Lower is better; 0 means no welfare loss.'),
}

def _finish(fig,path,show):
    """Save a chart and optionally open the rendered figure."""
    fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(path,dpi=180,bbox_inches='tight');
    if show: plt.show()
    plt.close(fig)

def plot_allocations(allocations,outdir,show=False):
    """Compare the resulting budget allocations in a figure."""
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    labels={'mean':'Mean Rule','util':'Util','im':'Independent Markets','ladder':'Ladder','utilprop':'UtilProp','piecewise':'Piecewise Uniform'}
    for key,a in allocations.items():
        fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(1,len(a)+1); ax.bar(x,a); ax.set_title(labels.get(key,key)); ax.set_xlabel('Project'); ax.set_ylabel('Budget share'); ax.set_xticks(x); ax.set_ylim(0,max(1.0,float(max(a))*1.08)); fig.text(.5,.01,'Allocation shares sum to 1.',ha='center',fontsize=9); _finish(fig,outdir/f'allocation_{key}.png',show)

def plot_metric_table(rows,metric,outdir,zoom=True,show=False):
    """Render selected per-mechanism metric values as a chart."""
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); names=[r['mechanism'] for r in rows]; vals=np.array([r[metric] for r in rows],float); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(names,vals); title,ylabel,note=METRIC_INFO.get(metric,(metric.replace('_',' ').title(),metric.replace('_',' ').title(),'')); ax.set_ylabel(ylabel); ax.tick_params(axis='x',rotation=25); ax.set_title(title); fig.text(.5,.01,note,ha='center',fontsize=9)
    if zoom and np.all(np.isfinite(vals)):
        lo,hi=float(vals.min()),float(vals.max())
        if hi>lo: pad=max((hi-lo)*.2,.001); ax.set_ylim(max(0,lo-pad),min(1.005,hi+pad) if metric=='sw_ratio' else hi+pad)
    _finish(fig,outdir/f'{metric}.png',show)

def plot_controlled_experiment(summary,outdir,metrics=None,show=False):
    """One-factor-at-a-time plots: only n OR m changes in each figure."""
    outdir=Path(outdir); metrics=metrics or ['avg_utility','l1_loss','min_utility','p10_utility']
    for metric in metrics:
        if metric not in summary.columns: continue
        title_metric,ylabel,note=METRIC_INFO.get(metric,(metric.replace('_',' ').title(),metric.replace('_',' ').title(),'')); metric_dir=outdir/metric; metric_dir.mkdir(parents=True,exist_ok=True)
        for (culture,mechanism),sub in summary.groupby(['culture','mechanism'],sort=True):
            safe_c=str(culture).replace(' ','_').replace('/','_'); safe_mech=str(mechanism).replace(' ','_').replace('/','_')
            for n,g in sub.groupby('n',sort=True):
                g=g.sort_values('m'); fig,ax=plt.subplots(figsize=(8.2,5.2)); ax.plot(g['m'],g[metric],marker='o'); ax.set_xlabel('Number of projects, m'); ax.set_ylabel(ylabel); ax.set_title(f'{title_metric} as m changes\n{mechanism} | culture={culture} | n={n} fixed'); ax.grid(True,alpha=.25); fig.text(.5,.025,f'Only m changes. Fixed: mechanism={mechanism}, culture={culture}, n={n}. Each point = mean over repetitions.',ha='center',fontsize=8.5); fig.text(.5,.008,note,ha='center',fontsize=8.5); _finish(fig,metric_dir/f'vary_m__{safe_c}__{safe_mech}__n{n}.png',show)
            for m,g in sub.groupby('m',sort=True):
                g=g.sort_values('n'); fig,ax=plt.subplots(figsize=(8.2,5.2)); ax.plot(g['n'],g[metric],marker='o'); ax.set_xlabel('Number of voters, n'); ax.set_ylabel(ylabel); ax.set_title(f'{title_metric} as n changes\n{mechanism} | culture={culture} | m={m} fixed'); ax.grid(True,alpha=.25); fig.text(.5,.025,f'Only n changes. Fixed: mechanism={mechanism}, culture={culture}, m={m}. Each point = mean over repetitions.',ha='center',fontsize=8.5); fig.text(.5,.008,note,ha='center',fontsize=8.5); _finish(fig,metric_dir/f'vary_n__{safe_c}__{safe_mech}__m{m}.png',show)


def plot_runtime_controlled(summary, outdir, show=False):
    """Controlled runtime plots using median execution time.

    Each vary-n figure fixes m, culture and mechanism. Each vary-m figure fixes
    n, culture and mechanism. No averaging is performed across n or m.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if summary is None or len(summary) == 0 or 'median_seconds' not in summary.columns:
        return
    for (culture, mechanism), sub in summary.groupby(['culture','mechanism'], sort=True):
        safe_c = str(culture).replace(' ','_').replace('/','_')
        safe_mech = str(mechanism).replace(' ','_').replace('/','_')
        for m, g in sub.groupby('m', sort=True):
            g = g.sort_values('n')
            if len(g) < 2: continue
            fig, ax = plt.subplots(figsize=(8.2,5.2))
            ax.plot(g['n'], g['median_seconds'], marker='o')
            ax.set_xlabel('Number of voters, n'); ax.set_ylabel('Median runtime (seconds)')
            ax.set_title(f'Runtime as n changes\n{mechanism} | culture={culture} | m={m} fixed')
            ax.grid(True, alpha=.25)
            fig.text(.5,.025,f'Only n changes. Fixed: mechanism={mechanism}, culture={culture}, m={m}.',ha='center',fontsize=8.5)
            fig.text(.5,.008,'Each point is the median of the timed repetitions; profile generation and file export are excluded.',ha='center',fontsize=8.5)
            _finish(fig,outdir/f'vary_n__{safe_c}__{safe_mech}__m{m}.png',show)
        for n, g in sub.groupby('n', sort=True):
            g = g.sort_values('m')
            if len(g) < 2: continue
            fig, ax = plt.subplots(figsize=(8.2,5.2))
            ax.plot(g['m'], g['median_seconds'], marker='o')
            ax.set_xlabel('Number of projects, m'); ax.set_ylabel('Median runtime (seconds)')
            ax.set_title(f'Runtime as m changes\n{mechanism} | culture={culture} | n={n} fixed')
            ax.grid(True, alpha=.25)
            fig.text(.5,.025,f'Only m changes. Fixed: mechanism={mechanism}, culture={culture}, n={n}.',ha='center',fontsize=8.5)
            fig.text(.5,.008,'Each point is the median of the timed repetitions; profile generation and file export are excluded.',ha='center',fontsize=8.5)
            _finish(fig,outdir/f'vary_m__{safe_c}__{safe_mech}__n{n}.png',show)

def plot_overall_comparisons(df, outdir, metrics=None, show=False):
    """Compact thesis-ready mechanism comparisons, aggregated only over rows explicitly supplied."""
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    metrics=metrics or ['avg_utility','l1_loss','sw_ratio','welfare_gap_pct','min_utility','p10_utility','utility_std','utility_gini']
    for metric in metrics:
        if metric not in df.columns: continue
        g=df.groupby('mechanism',as_index=False)[metric].mean().sort_values(metric)
        title,ylabel,note=METRIC_INFO.get(metric,(metric,metric,''))
        fig,ax=plt.subplots(figsize=(9,5.2)); ax.bar(g['mechanism'],g[metric]); ax.set_title(f'Overall mechanism comparison — {title}'); ax.set_ylabel(ylabel); ax.tick_params(axis='x',rotation=25)
        fig.text(.5,.015,'Aggregate over the experiment grid shown in the report; exact controlled results remain available separately. '+note,ha='center',fontsize=8.3)
        _finish(fig,outdir/f'overall__{metric}.png',show)
    if 'l1_loss' in df.columns and 'sw_ratio' in df.columns:
        g=df.groupby('mechanism',as_index=False)[['l1_loss','sw_ratio']].mean()
        fig,ax=plt.subplots(figsize=(8,5.5)); ax.scatter(g['l1_loss'],g['sw_ratio'])
        for _,r in g.iterrows(): ax.annotate(str(r['mechanism']),(r['l1_loss'],r['sw_ratio']),xytext=(5,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Mean L1-loss to Mean'); ax.set_ylabel('Mean SW / Optimal SW'); ax.set_title('Welfare–proportionality trade-off'); ax.grid(True,alpha=.25)
        fig.text(.5,.015,'Each point is one mechanism, aggregated over the same experiment grid.',ha='center',fontsize=8.5); _finish(fig,outdir/'tradeoff__l1_vs_sw_ratio.png',show)

def plot_robustness(summary,outdir,show=False):
    """Robustness plots where epsilon is the only x-axis variable; exact n,m,culture are fixed."""
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    metrics=['allocation_l1_shift','social_welfare_abs_change','l1_loss_abs_change','avg_utility_abs_change','gini_abs_change']
    labels={'allocation_l1_shift':'Allocation L1 shift','social_welfare_abs_change':'Absolute change in Social Welfare','l1_loss_abs_change':'Absolute change in L1-loss','avg_utility_abs_change':'Absolute change in Average Utility','gini_abs_change':'Absolute change in Utility Gini'}
    for metric in metrics:
        if metric not in summary.columns: continue
        md=outdir/metric; md.mkdir(parents=True,exist_ok=True)
        for (culture,n,m),sub in summary.groupby(['culture','n','m'],sort=True):
            fig,ax=plt.subplots(figsize=(8.5,5.4))
            for mech,g in sub.groupby('mechanism',sort=True):
                g=g.sort_values('epsilon'); ax.plot(g['epsilon'],g[metric],marker='o',label=mech)
            ax.set_xlabel('Perturbation size epsilon (exact per-voter L1)'); ax.set_ylabel(labels[metric]); ax.set_title(f'{labels[metric]} vs epsilon\nculture={culture} | n={n} | m={m}'); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
            fig.text(.5,.012,'Only epsilon changes; culture, n and m are fixed. Points are means over repetitions.',ha='center',fontsize=8.2)
            safe=str(culture).replace(' ','_').replace('/','_'); _finish(fig,md/f'{safe}__n{n}__m{m}.png',show)
        # useful global overview by epsilon and mechanism; clearly labelled aggregate
        g=summary.groupby(['epsilon','mechanism'],as_index=False)[metric].mean()
        fig,ax=plt.subplots(figsize=(8.5,5.4))
        for mech,s in g.groupby('mechanism',sort=True): s=s.sort_values('epsilon'); ax.plot(s['epsilon'],s[metric],marker='o',label=mech)
        ax.set_xlabel('Perturbation size epsilon'); ax.set_ylabel(labels[metric]); ax.set_title(f'Overall robustness — {labels[metric]}'); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
        fig.text(.5,.012,'Supplementary aggregate across the requested culture/n/m grid; exact controlled plots are authoritative.',ha='center',fontsize=8.2); _finish(fig,md/'OVERALL.png',show)
