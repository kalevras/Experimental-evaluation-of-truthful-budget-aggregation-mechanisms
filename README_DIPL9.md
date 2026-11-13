# dipl9 — human-readable thesis outputs

The computational mechanisms are unchanged. This version redesigns only reporting/export.

## Experiment output
- `01_RESULTS.xlsx`: human-readable results with Read Me, overview, exact n-m results, detailed runs.
- `02_INPUTS.xlsx`: explanation, profile index, characteristic type pool, actual voter inputs.
- `03_AUDIT.xlsx`: automated consistency checks.
- `figures/`: one-factor-at-a-time figures. Every figure explicitly states what changes and what remains fixed.
- `raw_data/`: machine-readable CSV files for reproducibility.

## Fairness output
- `01_FAIRNESS_RESULTS.xlsx`: explanations + exact fairness results + detailed runs + Gini secondary sheet.
- `figures/`: controlled fairness figures with captions.
- `raw_data/`: raw fairness CSV files.

## Important aggregation rule
Exact comparisons keep `(culture, n, m, mechanism)` fixed and average only repetitions. The code does not average n or m into artificial values.


## Runtime benchmark (D1 update)
The benchmark times only the mechanism function. Preference generation, metrics, plots, and file export are outside the timed region. It supports `--n-grid` and `--m-grid`, warm-ups, repeated timed runs, exact summaries (median/mean/std/min/max), raw timings, Excel documentation, and controlled plots where only n or only m changes.

## Correctness audit (final corrected build)

This build includes an automated regression test (`test_d2.py`) and a corrected
robustness perturbation. For every voter in robustness experiments, the generated
profile satisfies `||p_i^epsilon - p_i||_1 = epsilon` up to floating-point tolerance.
The robustness Excel workbook includes a dedicated `05_AUDIT` sheet.

Excel reports are intended to be human-readable as well as auditable: they contain
README/overview sheets, exact-setting summaries, detailed runs, filters, frozen
headers, appropriate numerical formats, controlled aggregates, and automated audit
information where applicable. Exact `(culture,n,m,mechanism)` results remain the
authoritative values for controlled comparisons.

Before a final thesis run, execute:

    python test_d2.py

A successful run prints `PASS` and checks allocation validity, Mean Rule L1=0,
Util welfare against the independent exact optimum, and exact robustness shifts.
