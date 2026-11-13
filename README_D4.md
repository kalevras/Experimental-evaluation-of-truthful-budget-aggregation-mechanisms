# D4 — Divisible Participatory Budget Aggregation

D4 extends the audited D2 core without changing the mathematical mechanism implementations.

## What is new
- Plots are generated automatically by `experiment`, `fairness`, `robustness`, and `benchmark`; no second plotting command is required.
- Controlled plots vary one experimental dimension at a time and state the fixed parameters in the title/caption.
- Overview plots are explicitly labelled as aggregates and do not replace exact controlled comparisons.
- Robustness now reports and plots allocation L1 shift, |ΔSW|, |ΔL1-loss|, |ΔAverage Utility| and |ΔGini| against exact per-voter L1 epsilon.
- External profile import supports CSV, XLSX, JSON and XML through the same validation/evaluation engine.

## Import examples
`python pb_app.py manual --input profile.csv --mechanisms all --table --plots --output MY_DATA`

`python pb_app.py manual --input profile.xlsx --sheet 0 --mechanisms all --table --plots --output MY_DATA`

`python pb_app.py manual --input profile.xml --mechanisms all --table --plots --output MY_DATA`

`python pb_app.py manual --input profile.json --mechanisms all --table --plots --output MY_DATA`

Every imported profile must be a numeric n×m matrix, nonnegative, finite, with each voter row summing to 1. Invalid input is rejected rather than silently normalized.

## Experiment outputs
Each experiment writes raw CSV data, formatted Excel reports, audits where applicable, and a `figures/` directory. The plotting functions consume the results produced by that run; thesis values are not hard-coded in plot code.

## D4.1 robustness fix (2026-10-11)
Robustness now always writes `raw_data/` CSVs, `01_ROBUSTNESS_RESULTS.xlsx`, and automatic PNG figures for allocation L1 shift, |ΔSW|, |ΔL1-loss|, |ΔAverage Utility|, and |ΔGini|. The workbook embeds the five overall robustness figures in sheet `06_PLOTS`. Controlled figures fix culture, n, and m and vary epsilon only. Runtime audits enforce the expected row count and exact per-voter L1 perturbation.

The program prints `Participatory Budgeting Framework D4.1-2026-10-11` at startup so an older extracted copy can be detected immediately.
