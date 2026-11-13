"""Recreate workbooks from stored numerical experiment outputs."""

from pathlib import Path
import argparse
import pandas as pd
from pb_excel import export_experiment, export_fairness, export_robustness, export_benchmark

def rebuild(folder: Path):
    """Regenerate Excel summaries from the available experiment CSVs."""
    if (folder/'detailed_results.csv').exists():
        df=pd.read_csv(folder/'detailed_results.csv')
        numeric=[c for c in ['l1_loss','social_welfare','avg_utility','validity_error','utility_std','min_utility','p10_utility','utility_gini','optimal_sw','sw_ratio','welfare_gap_pct'] if c in df]
        summary=df.groupby(['culture','n','m','mechanism'],as_index=False)[numeric].mean()
        l1=df.groupby(['culture','mechanism'],as_index=False)['l1_loss'].mean()
        export_experiment(df,summary,l1,folder/'experiment_report.xlsx')
        print('Rebuilt experiment_report.xlsx')
    if (folder/'fairness_detailed.csv').exists():
        df=pd.read_csv(folder/'fairness_detailed.csv')
        primary=['avg_utility','utility_std','min_utility','p10_utility','l1_loss']; secondary=['utility_gini']; metrics=primary+secondary
        summary=df.groupby(['culture','n','m','mechanism'],as_index=False)[metrics].mean()
        byculture=df.groupby(['culture','mechanism'],as_index=False)[metrics].mean()
        prim=df.groupby(['culture','n','m','mechanism'],as_index=False)[primary].mean()
        gini=df.groupby(['culture','n','m','mechanism'],as_index=False)[secondary].mean()
        export_fairness(df,summary,byculture,prim,gini,folder/'fairness_report.xlsx')
        print('Rebuilt fairness_report.xlsx')
    if (folder/'robustness_detailed.csv').exists():
        df=pd.read_csv(folder/'robustness_detailed.csv')
        metrics=['mean_voter_profile_l1_shift','allocation_l1_shift','avg_utility_abs_change','gini_abs_change']
        summary=df.groupby(['culture','n','m','epsilon','mechanism'],as_index=False)[metrics].mean()
        byculture=df.groupby(['culture','epsilon','mechanism'],as_index=False)[metrics].mean()
        export_robustness(df,summary,byculture,folder/'robustness_report.xlsx')
        print('Rebuilt robustness_report.xlsx')
    if (folder/'scaling_runtime.csv').exists():
        df=pd.read_csv(folder/'scaling_runtime.csv'); export_benchmark(df,folder/'benchmark_report.xlsx'); print('Rebuilt benchmark_report.xlsx')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Rebuild safe Excel reports from existing CSV results without rerunning experiments.')
    ap.add_argument('folder',help='Results folder')
    a=ap.parse_args(); rebuild(Path(a.folder))
