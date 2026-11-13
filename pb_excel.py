"""Reproducible Excel exports of experimental results and metrics."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.drawing.image import Image as XLImage

NAVY='1F4E78'; BLUE='D9EAF7'; PALE='F4F8FB'; WHITE='FFFFFF'; GREY='666666'; GREEN='E2F0D9'; THIN=Side(style='thin',color='D9E1F2')

METRICS = {
 'avg_utility': ('Average Utility','Mean overlap utility across voters. Higher means greater average voter satisfaction.'),
 'social_welfare': ('Social Welfare','Sum of all individual overlap utilities. Higher means greater total welfare.'),
 'sw_ratio': ('SW / Optimal SW','Social welfare divided by the independently computed welfare optimum. 1.000 means welfare-optimal.'),
 'welfare_gap_pct': ('Welfare Gap (%)','Percentage shortfall from the welfare optimum. Lower is better; 0 means no welfare loss.'),
 'min_utility': ('Minimum Utility','Utility of the least-satisfied voter. Higher means a better worst individual outcome.'),
 'p10_utility': ('P10 Utility','10th percentile of individual utility: about 10% of voters are at or below this value. Higher is better for the lower tail.'),
 'utility_std': ('Utility Std. Dev.','Standard deviation of individual utilities. Lower means less dispersion; it is not by itself a complete fairness criterion.'),
 'utility_gini': ('Utility Gini','Secondary inequality diagnostic for individual utilities. 0 means equal utilities; larger values mean greater inequality.'),
 'l1_loss': ('L1-loss to Mean','L1 distance between the mechanism allocation and the coordinate-wise mean preference allocation. Lower means closer to the mean/proportional benchmark; Mean Rule is 0 by construction.'),
 'validity_error': ('Validity Error','Numerical error in the allocation simplex constraint. Values near 0 indicate the allocation sums to 1.'),
 'optimal_sw': ('Optimal Social Welfare','Maximum social welfare attainable for the profile under the overlap-utility objective.'),
 'mean_voter_profile_l1_shift': ('Mean Input L1 Shift','Mean per-voter L1 change in the preference profile. Under controlled robustness this should equal epsilon up to numerical tolerance.'),
 'allocation_l1_shift': ('Allocation L1 Shift','L1 distance between the base and perturbed mechanism allocations. Lower means less empirical sensitivity.'),
 'avg_utility_abs_change': ('Average Utility |Change|','Absolute change in average voter utility after perturbation.'),
 'social_welfare_abs_change': ('Social Welfare |Change|','Absolute change in total social welfare after perturbation.'),
 'l1_loss_abs_change': ('L1-loss |Change|','Absolute change in L1-loss to the mean/proportional benchmark after perturbation.'),
 'gini_abs_change': ('Gini |Change|','Absolute change in utility Gini after perturbation.'),
}

RENAMES = {k:v[0] for k,v in METRICS.items()}
RENAMES.update({'culture':'Preference Culture','n':'Voters (n)','m':'Projects (m)','repetition':'Repetition','mechanism':'Mechanism','allocation':'Allocation'})


def _pretty(df):
    """Replace machine-oriented dataframe names with readable labels."""
    return df.rename(columns={c:RENAMES.get(c,c.replace('_',' ').title()) for c in df.columns})

def _write_title(ws,title,subtitle=None,width=10):
    """Insert a report heading and contextual subtitle into a worksheet."""
    ws.sheet_view.showGridLines=False
    ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=max(4,width))
    c=ws.cell(1,1,title); c.font=Font(size=18,bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY); c.alignment=Alignment(vertical='center')
    for j in range(2,max(4,width)+1): ws.cell(1,j).fill=PatternFill('solid',fgColor=NAVY)
    ws.row_dimensions[1].height=28
    if subtitle:
        ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=max(4,width))
        ws.cell(2,1,subtitle).alignment=Alignment(wrap_text=True,vertical='top'); ws.row_dimensions[2].height=36

def _number_format(source_col):
    """Choose Excel number formatting appropriate to the source metric."""
    c = str(source_col).lower()
    if c in {'n','m','repetition','seed','voter_id','type_id','base_archetype_id','voter_rows'}:
        return '0'
    if 'gap_pct' in c or c.endswith('_pct'):
        return '0.0000"%"'  # values are already stored as 0..100 percentages
    if c in {'epsilon'}:
        return '0.000'
    if 'seconds' in c or 'runtime' in c:
        return '0.000000'
    if c in {'validity_error'} or 'error' in c:
        return '0.00E+00'
    return '0.000000'


def _write_df(ws,df,start_row=1,freeze=None,filter=True):
    """Write a dataframe into a worksheet with headers, filters and formats."""
    raw_cols=list(df.columns)
    df=_pretty(df.copy()); ncols=len(df.columns)
    ws.sheet_view.zoomScale=90
    ws.sheet_properties.pageSetUpPr.fitToPage=True
    ws.page_setup.fitToWidth=1; ws.page_setup.fitToHeight=0; ws.page_layout_view=False
    ws.print_options.horizontalCentered=True
    for j,col in enumerate(df.columns,1):
        c=ws.cell(start_row,j,str(col)); c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY); c.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True); c.border=Border(bottom=THIN)
    for i,row in enumerate(df.itertuples(index=False,name=None),start_row+1):
        for j,val in enumerate(row,1):
            if pd.isna(val): val=None
            c=ws.cell(i,j,val); c.border=Border(bottom=THIN); c.fill=PatternFill('solid',fgColor=PALE if i%2==0 else WHITE)
            c.alignment=Alignment(horizontal='left' if isinstance(val,str) else 'center',vertical='center')
            if isinstance(val,float): c.number_format=_number_format(raw_cols[j-1])
    if freeze: ws.freeze_panes=freeze
    if filter and len(df): ws.auto_filter.ref=f'A{start_row}:{get_column_letter(ncols)}{start_row+len(df)}'
    ws.row_dimensions[start_row].height=34
    for j,col in enumerate(df.columns,1):
        sample=[str(col)]+[str(x) for x in df.iloc[:150,j-1].tolist() if x is not None]
        w=min(max(max(map(len,sample),default=8)+2,11),38)
        if 'Mechanism' in str(col) or 'Culture' in str(col): w=max(w,22)
        if 'Allocation' in str(col) or 'Preferences' in str(col): w=38
        ws.column_dimensions[get_column_letter(j)].width=w
    # Gentle heat maps on headline numerical metrics improve scanability without
    # changing any stored value.
    heat_names={'Average Utility','SW / Optimal SW','P10 Utility','Minimum Utility',
                'L1-Loss To Mean','Allocation L1 Shift','Median Seconds'}
    for j,col in enumerate(df.columns,1):
        if str(col) in heat_names and len(df) >= 2:
            letter=get_column_letter(j); rng=f'{letter}{start_row+1}:{letter}{start_row+len(df)}'
            ws.conditional_formatting.add(rng, ColorScaleRule(start_type='min',start_color='FDE9D9',mid_type='percentile',mid_value=50,mid_color='FFF2CC',end_type='max',end_color='E2F0D9'))

def _readme(wb,title,df,kind='experiment'):
    """Add a report-description worksheet with relevant methodological notes."""
    ws=wb.create_sheet('00_READ_ME'); _write_title(ws,title,'How to read this workbook. All reported averages keep the experimental setting fixed and average repetitions only.',8)
    info=[('What is one row?', 'One exact experimental setting: preference culture + n voters + m projects + mechanism. Repetitions are averaged only within that same setting.'),
          ('Voters (n)','Number of voters/agents in the profile.'),('Projects (m)','Number of divisible budget projects.'),('Repetition','Independent repetition of the same parameter setting using the seeded generator.'),
          ('Mechanism','Budget aggregation rule evaluated on the same input profile as the other selected mechanisms.'),
          ('Comparison rule','When studying n, m is held fixed. When studying m, n is held fixed. Culture and mechanism are also held fixed.')]
    ws['A4']='Field / rule'; ws['B4']='Meaning'
    for c in ws[4]: c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY)
    r=5
    for a,b in info:
        ws.cell(r,1,a).font=Font(bold=True); ws.cell(r,1).fill=PatternFill('solid',fgColor=BLUE); ws.cell(r,2,b).alignment=Alignment(wrap_text=True,vertical='top'); r+=1
    r+=1; ws.cell(r,1,'Metrics').font=Font(size=14,bold=True,color=NAVY); r+=1
    ws.cell(r,1,'Metric'); ws.cell(r,2,'Meaning / interpretation')
    for c in ws[r]: c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY)
    r+=1
    present=set(df.columns)
    for key,(name,desc) in METRICS.items():
        if key in present:
            ws.cell(r,1,name).font=Font(bold=True); ws.cell(r,1).fill=PatternFill('solid',fgColor=BLUE); ws.cell(r,2,desc).alignment=Alignment(wrap_text=True,vertical='top'); r+=1
    r+=1; ws.cell(r,1,'Important interpretation note').font=Font(bold=True,color=NAVY)
    ws.cell(r+1,1,'No single metric is treated as an overall ranking of mechanisms. Welfare, proportionality-related distance, lower-tail utility and inequality describe different aspects of the outcome.'); ws.merge_cells(start_row=r+1,start_column=1,end_row=r+2,end_column=8); ws.cell(r+1,1).alignment=Alignment(wrap_text=True,vertical='top')
    ws.column_dimensions['A'].width=28; ws.column_dimensions['B'].width=95
    for c in 'CDEFGH': ws.column_dimensions[c].width=12
    return ws

def _overview(wb,title,df,exact,primary):
    """Generate an overview worksheet from experiment aggregates."""
    ws=wb.create_sheet('01_OVERVIEW'); _write_title(ws,title,'Human-readable overview. Exact n,m combinations are never merged.',9)
    stats=[('Mechanism-runs',len(df)),('Mechanisms',df['mechanism'].nunique() if 'mechanism' in df else ''),('Preference cultures',df['culture'].nunique() if 'culture' in df else ''),('n values',', '.join(map(str,sorted(df['n'].unique()))) if 'n' in df else ''),('m values',', '.join(map(str,sorted(df['m'].unique()))) if 'm' in df else ''),('Repetitions per exact setting',int(df['repetition'].nunique()) if 'repetition' in df else '')]
    ws['A4']='Dataset'; ws['B4']='Value'
    for c in ws[4]: c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY)
    for i,(a,b) in enumerate(stats,5): ws.cell(i,1,a).font=Font(bold=True); ws.cell(i,1).fill=PatternFill('solid',fgColor=BLUE); ws.cell(i,2,b)
    ws['A13']='Exact results preview'; ws['A13'].font=Font(size=14,bold=True,color=NAVY)
    cols=[c for c in ['culture','n','m','mechanism']+primary if c in exact.columns]
    _write_df(ws,exact[cols],start_row=14,freeze='A15')
    return ws

def _new_book(title,df,exact,primary):
    """Create a styled workbook for a reproducible experiment report."""
    wb=Workbook(); wb.remove(wb.active); _readme(wb,title,df); _overview(wb,title,df,exact,primary); return wb

def export_experiment(df,summary,l1,path):
    """Export full and summarized controlled-experiment results."""
    exact=summary.sort_values(['culture','n','m','mechanism']).reset_index(drop=True)
    primary=[c for c in ['avg_utility','social_welfare','sw_ratio','welfare_gap_pct','l1_loss','min_utility','p10_utility','utility_std','utility_gini','validity_error'] if c in exact]
    wb=_new_book('Main Experimental Results',df,exact,primary)
    ws=wb.create_sheet('02_EXACT_RESULTS'); _write_title(ws,'Exact Results by n and m','Each row fixes culture, n, m and mechanism; values are means over repetitions only.',len(exact.columns)); _write_df(ws,exact,start_row=4,freeze='A5')
    ws=wb.create_sheet('03_DETAILED_RUNS'); _write_title(ws,'Detailed Mechanism Runs','One row per repetition and mechanism. Use this sheet for traceability, not for headline comparisons.',len(df.columns)); _write_df(ws,df.sort_values(['culture','n','m','repetition','mechanism']),start_row=4,freeze='A5')
    if len(l1):
        ws=wb.create_sheet('04_L1_BY_CULTURE'); _write_title(ws,'Mean L1-loss by Preference Culture','Supplementary aggregate across the requested grid; exact n,m results remain authoritative.',len(l1.columns)); _write_df(ws,l1,start_row=4,freeze='A5')
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)

def export_fairness(df,summary,by_culture,primary,gini,path):
    """Export culture-specific individual-utility and fairness summaries."""
    exact=summary.sort_values(['culture','n','m','mechanism']).reset_index(drop=True)
    prim=[c for c in ['avg_utility','min_utility','p10_utility','utility_std','l1_loss','utility_gini'] if c in exact]
    wb=_new_book('Fairness Analysis',df,exact,prim)
    ws=wb.create_sheet('02_EXACT_FAIRNESS'); _write_title(ws,'Exact Fairness Results','Culture, n, m and mechanism are fixed; only repetitions are averaged.',len(exact.columns)); _write_df(ws,exact,start_row=4,freeze='A5')
    ws=wb.create_sheet('03_DETAILED_RUNS'); _write_title(ws,'Detailed Fairness Runs','One row per repetition and mechanism.',len(df.columns)); _write_df(ws,df.sort_values(['culture','n','m','repetition','mechanism']),start_row=4,freeze='A5')
    ws=wb.create_sheet('04_BY_CULTURE'); _write_title(ws,'Fairness by Preference Culture','Supplementary aggregate across n and m; use exact results for controlled comparisons.',len(by_culture.columns)); _write_df(ws,by_culture.sort_values(['culture','mechanism']),start_row=4,freeze='A5')
    ws=wb.create_sheet('05_GINI_SECONDARY'); _write_title(ws,'Gini — Secondary Inequality Diagnostic','Gini is supplementary; it is not used as a stand-alone overall fairness verdict.',len(gini.columns)); _write_df(ws,gini.sort_values(['culture','n','m','mechanism']),start_row=4,freeze='A5')
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)

def export_inputs(input_df,type_df,path):
    """Export externally supplied profiles and their type metadata."""
    wb=Workbook(); wb.remove(wb.active)
    ws=wb.create_sheet('00_READ_ME'); _write_title(ws,'Characteristic Input Profiles','This workbook exposes the actual generated voter inputs used by the experiment.',8)
    explanations=[('Profile identity','Culture + n + m + repetition + seed uniquely identify a generated profile.'),('Voter input','The final p1...pm values are the preference vector passed to every selected mechanism for that profile.'),('Type / archetype','Characteristic profiles are generated from representative preference types. Permutation/shuffle and jitter create structured variations.'),('Simplex requirement','Every final voter preference vector is nonnegative and sums to 1, up to floating-point tolerance.'),('Why this file exists','It makes the experiment auditable: the reader can trace generated types to the exact voter preferences used as mechanism inputs.')]
    ws['A4']='Item'; ws['B4']='Explanation'
    for c in ws[4]: c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY)
    for i,(a,b) in enumerate(explanations,5): ws.cell(i,1,a).font=Font(bold=True); ws.cell(i,1).fill=PatternFill('solid',fgColor=BLUE); ws.cell(i,2,b).alignment=Alignment(wrap_text=True)
    ws.column_dimensions['A'].width=24; ws.column_dimensions['B'].width=100
    # Profile index
    profile_cols=[c for c in ['culture','n','m','repetition','seed'] if c in input_df]
    prof=input_df.groupby(profile_cols,as_index=False).size().rename(columns={'size':'voter_rows'}) if len(input_df) else pd.DataFrame()
    ws=wb.create_sheet('01_PROFILE_INDEX'); _write_title(ws,'Profile Index','One row per generated input profile. voter_rows should equal n.',max(6,len(prof.columns))); _write_df(ws,prof,start_row=4,freeze='A5')
    if len(type_df):
        ws=wb.create_sheet('02_TYPE_POOL'); _write_title(ws,'Characteristic Type Pool','Representative characteristic preference types available to the voter sampler.',len(type_df.columns)); _write_df(ws,type_df,start_row=4,freeze='A5')
    ws=wb.create_sheet('03_VOTER_INPUTS'); _write_title(ws,'Actual Voter Inputs','These are the final voter preference vectors used by the mechanisms.',len(input_df.columns)); _write_df(ws,input_df,start_row=4,freeze='A5')
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)

def export_audit(audit_df,path):
    """Write input-validation and numerical audit outcomes to Excel."""
    wb=Workbook(); wb.remove(wb.active); ws=wb.create_sheet('00_AUDIT')
    _write_title(ws,'Automated Input / Output Audit','PASS means the corresponding structural or numerical consistency check succeeded.',max(5,len(audit_df.columns)))
    _write_df(ws,audit_df,start_row=4,freeze='A5')
    # highlight passed cells softly
    if 'passed' in audit_df.columns:
        col=list(audit_df.columns).index('passed')+1
        for r in range(5,5+len(audit_df)):
            if bool(ws.cell(r,col).value): ws.cell(r,col).fill=PatternFill('solid',fgColor=GREEN)
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)

def export_manual(df,path):
    """Export allocations and metrics from a manual single-profile run."""
    exact=df.copy(); wb=_new_book('Divisible Budget – Manual Results',df,exact,[c for c in ['avg_utility','social_welfare','sw_ratio','l1_loss'] if c in df]); ws=wb.create_sheet('02_RESULTS'); _write_df(ws,df,start_row=1,freeze='A2'); wb.save(path); return Path(path)

def export_robustness(df,summary,by_culture,path,figures_dir=None):
    """Export perturbation experiment data and linked figure references."""
    exact=summary.sort_values(['culture','n','m','epsilon','mechanism']).reset_index(drop=True)
    wb=_new_book('Robustness / Sensitivity Analysis',df,exact,[c for c in ['mean_voter_profile_l1_shift','allocation_l1_shift','social_welfare_abs_change','l1_loss_abs_change','avg_utility_abs_change','gini_abs_change'] if c in exact])
    ws=wb.create_sheet('02_EXACT_RESULTS'); _write_title(ws,'Exact Robustness Results','Culture, n, m, epsilon and mechanism are fixed; only repetitions are averaged.',len(exact.columns)); _write_df(ws,exact,start_row=4,freeze='A5')
    ws=wb.create_sheet('03_BY_CULTURE'); _write_title(ws,'Robustness by Preference Culture','Supplementary aggregate by culture and epsilon; exact results remain authoritative.',len(by_culture.columns)); _write_df(ws,by_culture.sort_values(['culture','epsilon','mechanism']),start_row=4,freeze='A5')
    ws=wb.create_sheet('04_DETAILED_RUNS'); _write_title(ws,'Detailed Robustness Runs','One row per repetition, epsilon and mechanism. Mean Input L1 Shift should equal epsilon.',len(df.columns)); _write_df(ws,df.sort_values(['culture','n','m','repetition','epsilon','mechanism']),start_row=4,freeze='A5')
    audit=pd.DataFrame([
        {'check':'Mean per-voter input L1 shift equals epsilon', 'max_error':float((df['mean_voter_profile_l1_shift']-df['epsilon']).abs().max()), 'passed':bool((df['mean_voter_profile_l1_shift']-df['epsilon']).abs().max() <= 1e-10)},
        {'check':'Allocation L1 shift is finite', 'max_error':0.0, 'passed':bool(pd.Series(df['allocation_l1_shift']).notna().all())},
    ])
    ws=wb.create_sheet('05_AUDIT'); _write_title(ws,'Robustness Audit','Automated consistency checks for the controlled perturbation and output metrics.',len(audit.columns)); _write_df(ws,audit,start_row=4,freeze='A5')
    if 'passed' in audit.columns:
        col=list(audit.columns).index('passed')+1
        for r in range(5,5+len(audit)):
            if bool(ws.cell(r,col).value): ws.cell(r,col).fill=PatternFill('solid',fgColor=GREEN)
    if figures_dir is not None:
        figroot=Path(figures_dir)
        ws=wb.create_sheet('06_PLOTS')
        _write_title(ws,'Robustness Plots','Automatically generated from the same robustness summary used in this workbook. Overall plots are supplementary aggregates; controlled plots are stored in the figures folders.',10)
        row=4
        for metric in ['allocation_l1_shift','social_welfare_abs_change','l1_loss_abs_change','avg_utility_abs_change','gini_abs_change']:
            fp=figroot/metric/'OVERALL.png'
            if fp.exists():
                ws.cell(row,1,METRICS.get(metric,(metric,''))[0]).font=Font(bold=True,size=13,color=NAVY)
                img=XLImage(str(fp)); img.width=680; img.height=430
                ws.add_image(img,f'A{row+1}')
                row += 25
        ws.column_dimensions['A'].width=110
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)

def export_benchmark(df,path,raw_df=None):
    """Export runtime measurements and benchmark statistics."""
    wb=Workbook(); wb.remove(wb.active)
    ws=wb.create_sheet('00_READ_ME'); _write_title(ws,'Computational Scalability / Runtime','Controlled timing of mechanism execution only.',8)
    notes=[('What is timed?','Only the mechanism function M(P). Preference generation, metric computation, plots and file export are outside the timed region.'),('Warm-up','Warm-up executions are performed first and discarded, reducing first-call effects.'),('Timed repetitions','Each (n, m, culture, mechanism) setting is executed repeatedly on the same generated profile.'),('Median Runtime','Primary runtime summary. Median is less sensitive to occasional operating-system or background-process delays.'),('Mean / Std. Dev.','Mean runtime and its standard deviation across timed repetitions.'),('Min / Max','Observed range across the timed repetitions.'),('Controlled comparison','The report keeps exact n and m explicit. Runtime-vs-n plots fix m, culture and mechanism; runtime-vs-m plots fix n, culture and mechanism. No averaging is performed across n or m.') ]
    ws['A4']='Item'; ws['B4']='Explanation'
    for c in ws[4]: c.font=Font(bold=True,color=WHITE); c.fill=PatternFill('solid',fgColor=NAVY)
    for i,(a,b) in enumerate(notes,5): ws.cell(i,1,a).font=Font(bold=True); ws.cell(i,1).fill=PatternFill('solid',fgColor=BLUE); ws.cell(i,2,b).alignment=Alignment(wrap_text=True,vertical='top')
    ws.column_dimensions['A'].width=26; ws.column_dimensions['B'].width=105
    ws=wb.create_sheet('01_RUNTIME_SUMMARY'); _write_title(ws,'Runtime Summary','One row per exact n, m, culture and mechanism. Times are in seconds.',len(df.columns)); _write_df(ws,df.sort_values(['n','m','mechanism']),start_row=4,freeze='A5')
    if raw_df is not None and len(raw_df):
        ws=wb.create_sheet('02_RAW_TIMINGS'); _write_title(ws,'Raw Timed Repetitions','Individual timed executions used to compute the summary. Warm-ups are not included.',len(raw_df.columns)); _write_df(ws,raw_df.sort_values(['n','m','mechanism','repetition']),start_row=4,freeze='A5')
    Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path); return Path(path)
