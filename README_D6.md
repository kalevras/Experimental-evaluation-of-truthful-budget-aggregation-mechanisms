# D6 — Complete D5 + hill climbing Independent Markets

All original D5 source code, data files and results are preserved under this folder. The new entrypoint is `D6.py`.

## Requirements
Python 3 with numpy, pandas, matplotlib, scipy and openpyxl (as required by D5).

## Hill climbing (m=4)
Run from this folder:

```bash
python D6.py hill-im --projects 4 --restarts 100 --steps 300 --seed 8675309
```

For a longer search:

```bash
python D6.py hill-im --projects 4 --restarts 1000 --steps 1000 --seed 8675309
```

Files are saved into `D6_results/hill_im/`: `witness.json` (full profile, allocation, mean and L1-loss), `voter_profile.csv`, `best_witness.npz`, and `best_trace.csv`. Check an output with:

```bash
python D6.py verify-im D6_results/hill_im/best_witness.npz
```

**Important:** Stochastic hill climbing is a heuristic. The result is a reproducible numerical witness, not a mathematical proof of the global worst case. It may or may not recover the earlier reported 0.803458.

## Existing D5 capabilities

Original `pb_app.py`, `pb_core.py`, `pb_cultures.py`, `pb_excel.py`, `pb_io.py`, `pb_plots.py`, `rebuild_excel_reports.py` and tests remain unchanged. Refer to `README_DIPL9.md` / `README_D4.md` for original commands.

Run the original regression tests:

```bash
python test_d2.py
```
