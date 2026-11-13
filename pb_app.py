"""Interactive and batch command-line front end for budget aggregation experiments."""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pb_core import validate_profile, evaluate, MECHANISMS
from pb_plots import plot_allocations, plot_metric_table, plot_controlled_experiment, plot_runtime_controlled, plot_overall_comparisons, plot_robustness
from pb_cultures import generate
from pb_io import load_profile_file
from pb_excel import export_manual, export_experiment, export_fairness, export_robustness, export_benchmark, export_inputs, export_audit

D4_VERSION = 'D4.1-2026-10-11'

ALL = list(MECHANISMS)
YES = {'y', 'yes', 'ν', 'ναι', 'nai'}


def parse_mechs(s):
    """Parse mechanism names from a comma-separated CLI selection."""
    if not s or s.lower() == 'all':
        return ALL
    ans = [x.strip().lower() for x in s.split(',') if x.strip()]
    bad = [x for x in ans if x not in MECHANISMS]
    if bad:
        raise ValueError(f"Unknown mechanism(s): {bad}. Choices: {ALL}")
    return ans


def load_profile(path, n=None, m=None, sheet=0):
    """Read a profile file and verify the requested voter/project dimensions."""
    P = load_profile_file(path, sheet=sheet)
    file_n, file_m = P.shape
    if n is not None and n != file_n:
        raise ValueError(f"--n={n} but input file contains {file_n} voters.")
    if m is not None and m != file_m:
        raise ValueError(f"--m={m} but input file contains {file_m} projects.")
    return validate_profile(file_n if n is None else n, file_m if m is None else m, P)


def ask_positive_int(prompt):
    """Prompt repeatedly until the user enters a positive integer."""
    while True:
        try:
            x = int(input(prompt).strip())
            if x > 0:
                return x
        except ValueError:
            pass
        print('Please enter a positive integer.')


def ask_profile(n, m):
    """Collect and validate one allocation vector per voter interactively."""
    print(f"\nEnter {n} voter rows. Each row must contain {m} non-negative numbers summing to 1.")
    rows = []
    for i in range(n):
        while True:
            try:
                vals = [float(x) for x in input(f"Voter {i+1}: ").replace(',', ' ').split()]
                row = validate_profile(1, m, [vals])[0]
                rows.append(row)
                break
            except ValueError as e:
                print(f"Invalid row: {e}")
    return np.asarray(rows, dtype=float)


def ask_mechanisms():
    """Display available mechanisms and collect an interactive selection."""
    print('\nAvailable mechanisms:')
    for i, key in enumerate(ALL, 1):
        print(f"  {i}. {MECHANISMS[key][0]}")
    print(f"  {len(ALL)+1}. All mechanisms")
    while True:
        raw = input('Select numbers separated by spaces/commas (or type all): ').strip().lower()
        if raw == 'all' or raw == str(len(ALL)+1):
            return ALL
        try:
            nums = [int(x) for x in raw.replace(',', ' ').split()]
            if nums and all(1 <= x <= len(ALL) for x in nums):
                # preserve order and remove duplicates
                return list(dict.fromkeys(ALL[x-1] for x in nums))
        except ValueError:
            pass
        print('Invalid selection. Try again.')


def save_outputs(rows, allocations, outdir, make_table=False, make_plots=False, show_plots=False):
    """Write manual-run allocations, score tables and requested figures."""
    out = Path(outdir)
    df = pd.DataFrame(rows)
    if make_table or make_plots:
        out.mkdir(parents=True, exist_ok=True)
    if make_table:
        df.to_csv(out / 'results.csv', index=False)
        try:
            export_manual(df, out / 'results.xlsx')
        except ImportError:
            print('XLSX skipped (install openpyxl if you want Excel output). CSV was saved.')
    if make_plots:
        plot_allocations(allocations, out / 'plots', show=show_plots)
        if 'sw_ratio' in df.columns:
            plot_metric_table(rows, 'sw_ratio', out / 'plots', zoom=True, show=show_plots)
        plot_metric_table(rows, 'l1_loss', out / 'plots', zoom=True, show=show_plots)
    return df


def run_manual(n=None, m=None, input_path=None, mechanisms=None, output='results_manual',
               make_table=False, make_plots=False, show_plots=False,
               compute_optimum=True, ask_outputs=False, sheet=0):
    """Execute the selected mechanisms for a supplied or interactive profile."""
    if input_path:
        P = load_profile(input_path, n, m, sheet=sheet)
        n, m = P.shape
    else:
        if n is None or m is None:
            raise ValueError('--n and --m are required when --input is not used.')
        P = ask_profile(n, m)
    mechs = parse_mechs(mechanisms) if mechanisms else ask_mechanisms()
    rows, allocs, _, _ = evaluate(P, mechs, compute_optimum=compute_optimum)

    if ask_outputs:
        make_table = input('Save numerical tables (CSV/XLSX)? [y/N]: ').strip().lower() in YES
        make_plots = input('Generate plots? [y/N]: ').strip().lower() in YES
        if make_plots:
            show_plots = input('Show the plots on screen now? [Y/n]: ').strip().lower() not in {'n', 'no', 'ο', 'όχι', 'oxi'}

    df = save_outputs(rows, allocs, output, make_table, make_plots, show_plots)
    print('\nRESULTS')
    with pd.option_context('display.max_colwidth', 120, 'display.width', 220):
        print(df.to_string(index=False))
    if m > 3 and 'piecewise' in mechs:
        print("\nNOTE: Piecewise Uniform is defined and computed for m>3. The specific L1-loss guarantee cited in the thesis is the published m=3 guarantee and is not claimed here for m>3.")


def interactive():
    """Route to the interactive command-line workflow."""
    print('=== Divisible Budget Aggregation ===')
    n = ask_positive_int('Number of voters n: ')
    m = ask_positive_int('Number of projects m: ')
    run_manual(n, m, mechanisms=None, ask_outputs=True)


def manual(args):
    """Run a single configured experiment from CLI arguments."""
    run_manual(args.n, args.m, args.input, args.mechanisms, args.output,
               args.table, args.plots, args.show_plots,
               not args.no_optimum, args.ask_outputs, args.sheet)


def experiment(args):
    """Generate and evaluate controlled synthetic preference profiles."""
    ns = [int(x) for x in args.n_grid.split(',')]
    ms = [int(x) for x in args.m_grid.split(',')]
    cultures = [x.strip() for x in args.cultures.split(',')]
    mechs = parse_mechs(args.mechanisms)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    all_rows, input_rows, type_rows = [], [], []
    for culture in cultures:
        for n in ns:
            for m in ms:
                for rep in range(1, args.repetitions + 1):
                    if culture.lower().replace('-', '_') in {'characteristic','archetype','archetypes'}:
                        P, voter_meta, pool_meta = generate(culture, n, m, rng, args.max_types, args.archetype_jitter, return_metadata=True)
                        for rec in voter_meta:
                            rr = dict(rec); rr.update(culture=culture,n=n,m=m,repetition=rep,seed=args.seed)
                            input_rows.append(rr)
                        for rec in pool_meta:
                            rr = dict(rec); rr.update(culture=culture,n=n,m=m,repetition=rep,seed=args.seed)
                            type_rows.append(rr)
                    else:
                        P = generate(culture, n, m, rng, args.max_types, args.archetype_jitter)
                    rows, _, _, _ = evaluate(P, mechs, compute_optimum=not args.no_optimum)
                    for r in rows:
                        r.update(culture=culture, n=n, m=m, repetition=rep)
                        all_rows.append(r)
    df = pd.DataFrame(all_rows)
    raw = out / 'raw_data'; raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / 'detailed_results.csv', index=False)
    numeric = [c for c in ['l1_loss','social_welfare','avg_utility','validity_error','utility_std','min_utility','p10_utility','utility_gini','optimal_sw','sw_ratio','welfare_gap_pct'] if c in df]
    # IMPORTANT: averages only repetitions; n and m remain explicit grouping keys.
    summary = df.groupby(['culture','n','m','mechanism'], as_index=False)[numeric].mean()
    summary.to_csv(raw / 'summary_by_culture_n_m.csv', index=False)
    l1 = df.groupby(['culture','mechanism'], as_index=False)['l1_loss'].mean()
    l1.to_csv(raw / 'mean_l1_by_preference_culture.csv', index=False)
    input_df = pd.DataFrame(input_rows) if input_rows else pd.DataFrame()
    if input_rows:
        type_df = pd.DataFrame(type_rows)
        input_df.to_csv(raw / 'characteristic_inputs.csv', index=False)
        type_df.to_csv(raw / 'characteristic_type_pool.csv', index=False)
        export_inputs(input_df, type_df, out / '02_INPUTS.xlsx')
    # Automated audit: input simplex + output identities and structural checks.
    audit=[]
    if len(input_df):
        parsed = input_df['final_type_preferences'].map(lambda s: np.fromstring(str(s), sep=','))
        sum_err = max(abs(float(x.sum())-1.0) for x in parsed)
        min_pref = min(float(x.min()) for x in parsed)
        audit.append({'check':'characteristic input rows sum to 1', 'max_error':sum_err, 'passed':bool(sum_err<=1e-10)})
        audit.append({'check':'characteristic input preferences nonnegative', 'max_error':max(0.0,-min_pref), 'passed':bool(min_pref>=-1e-12)})
    audit.append({'check':'avg_utility == social_welfare / n', 'max_error':float((df['avg_utility']-df['social_welfare']/df['n']).abs().max()), 'passed':bool(np.allclose(df['avg_utility'],df['social_welfare']/df['n'],atol=1e-12,rtol=0))})
    counts=df.groupby(['culture','n','m','repetition']).size()
    audit.append({'check':'mechanism rows per profile', 'max_error':float((counts-len(mechs)).abs().max()), 'passed':bool((counts==len(mechs)).all())})
    audit.append({'check':'allocation validity error', 'max_error':float(df['validity_error'].max()), 'passed':bool(df['validity_error'].max()<=1e-8)})
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(raw/'audit_checks.csv',index=False)
    export_audit(audit_df, out / '03_AUDIT.xlsx')
    export_experiment(df, summary, l1, out / '01_RESULTS.xlsx')
    plot_controlled_experiment(summary, out/'figures'/'controlled', metrics=numeric, show=False)
    plot_overall_comparisons(df, out/'figures'/'overview', metrics=numeric, show=False)
    print(f"Saved {len(df):,} mechanism-runs to {out}")



def perturb_profile(P, epsilon, rng, atol=1e-10):
    """Return a controlled perturbation with exact per-voter L1 size.

    For every voter i, the returned profile Q satisfies
        ||Q[i] - P[i]||_1 = epsilon
    up to floating-point tolerance.  A random simplex target is used whenever
    it is far enough from the original report; otherwise a randomly selected
    sufficiently distant simplex vertex is used.  The move is always along a
    convex segment, so non-negativity and row-sum-one are preserved.

    The thesis robustness experiments use epsilon <= 1.  For m >= 2 an L1
    displacement of at most 1 is always feasible from every simplex point.
    """
    P = np.asarray(P, dtype=float)
    n, m = P.shape
    if m < 2:
        if abs(epsilon) <= atol:
            return P.copy()
        raise ValueError('A positive L1 perturbation requires at least 2 projects.')
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError('epsilon must be in [0,1].')
    if epsilon == 0.0:
        return P.copy()

    Q = np.empty_like(P)
    eye = np.eye(m)
    for i, p in enumerate(P):
        target = None
        distance = 0.0
        # Keep the perturbation stochastic while guaranteeing the requested size.
        for _ in range(32):
            candidate = rng.dirichlet(np.ones(m))
            d = float(np.abs(candidate - p).sum())
            if d + atol >= epsilon:
                target, distance = candidate, d
                break
        if target is None:
            # A vertex e_j has distance 2(1-p_j).  Randomize among all vertices
            # that can support the requested displacement.
            feasible = np.flatnonzero(2.0 * (1.0 - p) + atol >= epsilon)
            if feasible.size == 0:
                raise RuntimeError('Could not construct the requested L1 perturbation.')
            j = int(rng.choice(feasible))
            target = eye[j]
            distance = float(np.abs(target - p).sum())

        lam = epsilon / distance
        q = (1.0 - lam) * p + lam * target
        # Numerical cleanup only; this does not change the intended geometry.
        q = np.clip(q, 0.0, 1.0)
        q /= q.sum()
        Q[i] = q

    Q = validate_profile(n, m, Q)
    shifts = np.abs(Q - P).sum(axis=1)
    if not np.allclose(shifts, epsilon, atol=atol, rtol=0.0):
        raise RuntimeError(
            f'Controlled perturbation failed: max |L1-epsilon| = '
            f'{float(np.max(np.abs(shifts-epsilon))):.3e}'
        )
    return Q


def fairness_experiment(args):
    """Compute fairness metrics across independent profiles and cultures."""
    ns = [int(x) for x in args.n_grid.split(',')]
    ms = [int(x) for x in args.m_grid.split(',')]
    cultures = [x.strip() for x in args.cultures.split(',')]
    mechs = parse_mechs(args.mechanisms)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for culture in cultures:
        for n in ns:
            for m in ms:
                for rep in range(1, args.repetitions + 1):
                    P = generate(culture, n, m, rng, args.max_types, args.archetype_jitter)
                    rows, _, _, _ = evaluate(P, mechs, compute_optimum=False)
                    for r in rows:
                        r.update(culture=culture, n=n, m=m, repetition=rep)
                        all_rows.append(r)
    df = pd.DataFrame(all_rows)
    cols = ['culture','n','m','repetition','mechanism','avg_utility','utility_std',
            'min_utility','p10_utility','utility_gini','l1_loss','validity_error']
    raw = out / 'raw_data'; raw.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(raw / 'fairness_detailed.csv', index=False)
    # Primary fairness indicators; Gini is retained as a secondary inequality diagnostic.
    primary = ['avg_utility','utility_std','min_utility','p10_utility','l1_loss']
    secondary = ['utility_gini']
    metrics = primary + secondary
    fair_summary = df.groupby(['culture','n','m','mechanism'], as_index=False)[metrics].mean()
    fair_summary.to_csv(raw / 'fairness_summary.csv', index=False)
    fair_culture = df.groupby(['culture','mechanism'], as_index=False)[metrics].mean()
    fair_culture.to_csv(raw / 'fairness_by_culture.csv', index=False)
    fair_primary = df.groupby(['culture','n','m','mechanism'], as_index=False)[primary].mean()
    fair_primary.to_csv(raw / 'fairness_primary_summary.csv', index=False)
    fair_gini = df.groupby(['culture','n','m','mechanism'], as_index=False)[secondary].mean()
    fair_gini.to_csv(raw / 'fairness_secondary_gini.csv', index=False)
    export_fairness(df[cols], fair_summary, fair_culture, fair_primary, fair_gini, out / '01_FAIRNESS_RESULTS.xlsx')
    plot_controlled_experiment(fair_summary, out/'figures'/'controlled', metrics=['avg_utility','min_utility','p10_utility','utility_std','utility_gini','l1_loss'], show=False)
    plot_overall_comparisons(df, out/'figures'/'overview', metrics=['avg_utility','min_utility','p10_utility','utility_std','utility_gini','l1_loss'], show=False)
    print(f"Saved {len(df):,} fairness mechanism-runs to {out}")


def robustness_experiment(args):
    """Measure allocation and utility changes after controlled perturbations."""
    ns = [int(x) for x in args.n_grid.split(',')]
    ms = [int(x) for x in args.m_grid.split(',')]
    cultures = [x.strip() for x in args.cultures.split(',')]
    epsilons = [float(x) for x in args.epsilons.split(',')]
    if any(e < 0 or e > 1 for e in epsilons):
        raise ValueError('All epsilons must be in [0,1].')
    mechs = parse_mechs(args.mechanisms)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for culture in cultures:
        for n in ns:
            for m in ms:
                for rep in range(1, args.repetitions + 1):
                    P = generate(culture, n, m, rng, args.max_types, args.archetype_jitter)
                    base_rows, base_allocs, _, _ = evaluate(P, mechs, compute_optimum=False)
                    base_by_key = {k: r for k, r in zip(mechs, base_rows)}
                    for epsilon in epsilons:
                        Q = perturb_profile(P, epsilon, rng)
                        pert_rows, pert_allocs, _, _ = evaluate(Q, mechs, compute_optimum=False)
                        pert_by_key = {k: r for k, r in zip(mechs, pert_rows)}
                        profile_shift = float(np.mean(np.abs(Q - P).sum(axis=1)))
                        for key in mechs:
                            a0, a1 = base_allocs[key], pert_allocs[key]
                            r0, r1 = base_by_key[key], pert_by_key[key]
                            all_rows.append({
                                'culture': culture, 'n': n, 'm': m, 'repetition': rep,
                                'epsilon': epsilon, 'mechanism': r0['mechanism'],
                                'mean_voter_profile_l1_shift': profile_shift,
                                'allocation_l1_shift': float(np.abs(a1-a0).sum()),
                                'avg_utility_base': r0['avg_utility'],
                                'avg_utility_perturbed': r1['avg_utility'],
                                'avg_utility_abs_change': abs(r1['avg_utility']-r0['avg_utility']),
                                'social_welfare_base': r0['social_welfare'],
                                'social_welfare_perturbed': r1['social_welfare'],
                                'social_welfare_abs_change': abs(r1['social_welfare']-r0['social_welfare']),
                                'l1_loss_base': r0['l1_loss'],
                                'l1_loss_perturbed': r1['l1_loss'],
                                'l1_loss_abs_change': abs(r1['l1_loss']-r0['l1_loss']),
                                'gini_base': r0['utility_gini'],
                                'gini_perturbed': r1['utility_gini'],
                                'gini_abs_change': abs(r1['utility_gini']-r0['utility_gini']),
                            })
    df = pd.DataFrame(all_rows)
    raw = out / 'raw_data'; raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / 'robustness_detailed.csv', index=False)
    metrics = ['mean_voter_profile_l1_shift','allocation_l1_shift','social_welfare_abs_change','l1_loss_abs_change','avg_utility_abs_change','gini_abs_change']
    rob_summary = df.groupby(['culture','n','m','epsilon','mechanism'], as_index=False)[metrics].mean()
    rob_summary.to_csv(raw / 'robustness_summary.csv', index=False)
    rob_culture = df.groupby(['culture','epsilon','mechanism'], as_index=False)[metrics].mean()
    rob_culture.to_csv(raw / 'robustness_by_culture.csv', index=False)

    # Generate figures BEFORE the workbook so the key overview figures can be
    # embedded into the Excel report. All plots use the just-computed summary.
    figures = out / 'figures'
    plot_robustness(rob_summary, figures, show=False)
    export_robustness(df, rob_summary, rob_culture, out / '01_ROBUSTNESS_RESULTS.xlsx', figures_dir=figures)

    expected = len(cultures)*len(ns)*len(ms)*len(epsilons)*args.repetitions*len(mechs)
    if len(df) != expected:
        raise RuntimeError(f'Robustness row-count audit failed: got {len(df)}, expected {expected}.')
    eps_err = float((df['mean_voter_profile_l1_shift']-df['epsilon']).abs().max())
    if eps_err > 1e-10:
        raise RuntimeError(f'Exact-L1 audit failed: max error={eps_err:.3e}.')
    print(f"[{D4_VERSION}] Saved {len(df):,} robustness mechanism-runs to {out}")
    print(f"[{D4_VERSION}] Figures: {figures} | Excel: {out/'01_ROBUSTNESS_RESULTS.xlsx'}")

def benchmark(args):
    """Controlled mechanism-runtime benchmark.

    Profile generation and file export are outside the timed region. Warm-up runs
    are discarded. Each recorded repetition times only the mechanism function,
    not evaluate(), welfare/fairness metrics, plots, or Excel/CSV writing.

    The benchmark accepts both n-grid and m-grid. Every summary row therefore
    corresponds to one exact (n, m, culture, mechanism) setting. Controlled plots
    vary only n with m fixed, or only m with n fixed.
    """
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError('--warmups must be >= 0 and --repetitions must be >= 1.')
    ns = [int(x) for x in args.n_grid.split(',') if x.strip()]
    if args.m_grid:
        ms = [int(x) for x in args.m_grid.split(',') if x.strip()]
    else:
        ms = [int(args.m)]
    if not ns or not ms or any(n < 1 for n in ns) or any(m < 1 for m in ms):
        raise ValueError('All n-grid and m-grid values must be positive integers.')

    rng = np.random.default_rng(args.seed)
    mechs = parse_mechs(args.mechanisms)
    raw_rows, summary_rows = [], []

    for n in ns:
        for m in ms:
            # One common profile per exact (n,m) setting; generation is untimed.
            P = generate(args.culture, n, m, rng)
            for key in mechs:
                name, fn = MECHANISMS[key]
                for _ in range(args.warmups):
                    fn(P)
                timings = []
                for rep in range(1, args.repetitions + 1):
                    t0 = time.perf_counter()
                    fn(P)
                    elapsed = time.perf_counter() - t0
                    timings.append(elapsed)
                    raw_rows.append({'n': n, 'm': m, 'culture': args.culture,
                                     'mechanism': name, 'repetition': rep,
                                     'seconds': elapsed})
                x = np.asarray(timings, dtype=float)
                summary_rows.append({'n': n, 'm': m, 'culture': args.culture,
                                     'mechanism': name, 'warmups': args.warmups,
                                     'timed_repetitions': args.repetitions,
                                     'median_seconds': float(np.median(x)),
                                     'mean_seconds': float(np.mean(x)),
                                     'std_seconds': float(np.std(x, ddof=0)),
                                     'min_seconds': float(np.min(x)),
                                     'max_seconds': float(np.max(x))})
                print(f"n={n:,} m={m:,} {name}: median={np.median(x):.6f}s "
                      f"mean={np.mean(x):.6f}s std={np.std(x, ddof=0):.6f}s "
                      f"({args.repetitions} timed runs)")

    d = Path(args.output); d.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)
    raw_df.to_csv(d / 'runtime_raw.csv', index=False)
    summary_df.to_csv(d / 'runtime_summary.csv', index=False)
    export_benchmark(summary_df, d / 'benchmark_report.xlsx', raw_df=raw_df)
    plot_runtime_controlled(summary_df, d / 'figures', show=False)
    print(f"Saved {len(raw_df):,} timed mechanism-runs to {d}")


def build_parser():
    """Define the command-line interface and all supported subcommands."""
    p = argparse.ArgumentParser(description='Unified Divisible Budget Aggregation framework')
    sp = p.add_subparsers(dest='mode')
    q = sp.add_parser('manual', help='Run user-supplied voting results')
    q.add_argument('--n', type=int); q.add_argument('--m', type=int)
    q.add_argument('--input', help='Input profile: CSV/XLSX/JSON/XML. n and m are inferred when omitted.')
    q.add_argument('--sheet', default=0, help='Excel sheet index/name (default: 0)')
    q.add_argument('--mechanisms', default='all', help='all or comma list: mean,util,im,ladder,utilprop,piecewise')
    q.add_argument('--output', default='results_manual')
    q.add_argument('--table', action='store_true', help='Save CSV/XLSX table')
    q.add_argument('--plots', action='store_true', help='Generate plot files')
    q.add_argument('--show-plots', action='store_true', help='Display generated plots on screen')
    q.add_argument('--ask-outputs', action='store_true', help='Ask interactively whether to create tables and plots')
    q.add_argument('--no-optimum', action='store_true')
    q.set_defaults(func=manual)

    e = sp.add_parser('experiment', help='Run thesis experiments through the same engine')
    e.add_argument('--n-grid', default='10,25,50,100,250,500,750,1000'); e.add_argument('--m-grid', default='3,4,5,6,8,10,20,30,50')
    e.add_argument('--cultures', default='random,balanced,single_minded,polarized,majority_minority,mixed,sparse,concentrated,characteristic')
    e.add_argument('--repetitions', type=int, default=10); e.add_argument('--mechanisms', default='all'); e.add_argument('--seed', type=int, default=20260918)
    e.add_argument('--max-types', type=int, default=100); e.add_argument('--archetype-jitter', type=float, default=0.03)
    e.add_argument('--output', default='results_experiment'); e.add_argument('--no-optimum', action='store_true'); e.add_argument('--controlled-plots', action='store_true', help='Generate one-factor-at-a-time plots with all other experimental parameters fixed'); e.set_defaults(func=experiment)

    f = sp.add_parser('fairness', help='Voter-level fairness experiment')
    f.add_argument('--n-grid', default='10,25,50,100,250,500,750,1000'); f.add_argument('--m-grid', default='3,4,5,6,8,10,20,30,50')
    f.add_argument('--cultures', default='random,balanced,single_minded,polarized,majority_minority,mixed,sparse,concentrated,characteristic')
    f.add_argument('--repetitions', type=int, default=10); f.add_argument('--mechanisms', default='all')
    f.add_argument('--max-types', type=int, default=100); f.add_argument('--archetype-jitter', type=float, default=0.03)
    f.add_argument('--seed', type=int, default=42); f.add_argument('--output', default='results_fairness'); f.set_defaults(func=fairness_experiment)

    r = sp.add_parser('robustness', help='Sensitivity/robustness under controlled preference perturbations')
    r.add_argument('--n-grid', default='50,100,500'); r.add_argument('--m-grid', default='3,10,50')
    r.add_argument('--cultures', default='random,balanced,single_minded,polarized,majority_minority,mixed,sparse,concentrated,characteristic')
    r.add_argument('--epsilons', default='0.01,0.05,0.10'); r.add_argument('--repetitions', type=int, default=20)
    r.add_argument('--mechanisms', default='all'); r.add_argument('--seed', type=int, default=4242)
    r.add_argument('--max-types', type=int, default=100); r.add_argument('--archetype-jitter', type=float, default=0.03)
    r.add_argument('--output', default='results_robustness'); r.set_defaults(func=robustness_experiment)

    b = sp.add_parser('benchmark', help='Scaling/runtime experiment using the same engine')
    b.add_argument('--n-grid', default='1000,10000,100000,1000000', help='Comma-separated voter counts')
    b.add_argument('--m', type=int, default=3, help='Single project count (backward compatible; ignored when --m-grid is supplied)')
    b.add_argument('--m-grid', default=None, help='Comma-separated project counts; enables controlled scaling in m')
    b.add_argument('--culture', default='random')
    b.add_argument('--mechanisms', default='mean')
    b.add_argument('--warmups', type=int, default=2, help='Untimed warm-up executions per exact n, m and mechanism')
    b.add_argument('--repetitions', type=int, default=10, help='Timed executions per exact n, m and mechanism')
    b.add_argument('--seed', type=int, default=20260918)
    b.add_argument('--output', default='results_benchmark')
    b.set_defaults(func=benchmark)
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    print(f'Participatory Budgeting Framework {D4_VERSION}')
    if args.mode is None:
        interactive()
    else:
        args.func(args)
