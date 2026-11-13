"""Audit raw-data provenance, source hashes and independent numerical summaries.

Run from the D7/ root with:
    python analysis/audit_provenance.py

This does not infer missing historical seeds or reconstruct unarchived runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "provenance"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "main": "data/main/detailed_results.csv",
    "n_growth": "data/n_growth/detailed_results_full.csv",
    "n_growth_intervals": "data/n_growth/student_t_95_ci_full.csv",
    "fairness": "data/fairness/fairness_detailed.csv",
    "robustness": "data/robustness/robustness_detailed.csv",
    "bonferroni_fairness": "data/supplementary/bonferroni_fairness_R50.csv",
    "bonferroni_robustness": "data/supplementary/bonferroni_robustness_eps_010_R50.csv",
    "agreement": "data/supplementary/agreement_n100_R50.csv",
    "hill_climbing": "hill_climbing/results/full_restarts.csv",
}

rows = []
for label, rel in DATASETS.items():
    fp = ROOT / rel
    df = pd.read_csv(fp)
    describe = []
    for col in ("n", "m", "epsilon", "culture", "seed", "R", "repetition"):
        if col in df.columns:
            values = df[col].dropna().unique()
            if len(values) < 12:
                describe.append(f"{col}={sorted(str(v) for v in values)}")
            else:
                describe.append(f"{col}: {len(values)} values")
    rows.append({
        "experiment": label,
        "relative_path": rel,
        "sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
        "rows": len(df),
        "columns": len(df.columns),
        "parameters_observable_in_csv": "; ".join(describe),
    })
with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# The number quoted in §4.15 is computed strictly from the final R=50 dataset.
r = pd.read_csv(ROOT / DATASETS["robustness"])
sel = r[
    r["mechanism"].eq("UtilProp")
    & np.isclose(r["epsilon"], 0.1)
    & r["n"].eq(100)
    & r["m"].eq(10)
]
assert len(sel) == 400 and sel["culture"].nunique() == 8
assert all(sel.groupby("culture")["repetition"].nunique() == 50)
utilprop_mean = float(sel["l1_loss_abs_change"].mean())
assert np.isclose(utilprop_mean, 0.03167117093389139, rtol=0, atol=1e-12)
per_culture = sel.groupby("culture")["l1_loss_abs_change"].agg(["mean", "size"])
per_culture.to_csv(OUT / "utilprop_eps010_by_culture.csv", float_format="%.12f")

# An independent numeric check for all n-growth Student-t intervals.
d = pd.read_csv(ROOT / DATASETS["n_growth"])
ci_file = pd.read_csv(ROOT / DATASETS["n_growth_intervals"])
keys = ["m", "n", "mechanism"]
computed = []
for group, frame in d.groupby(keys):
    for metric in ("sw_ratio", "l1_loss", "welfare_gap_pct"):
        x = frame[metric].to_numpy(dtype=float)
        n = len(x)
        avg = float(x.mean())
        sd = float(x.std(ddof=1))
        delta = float(t.ppf(0.975, n-1)*sd/np.sqrt(n))
        computed.append(dict(zip(keys, group), metric=metric, R=n, mean=avg,
                             ci_low=avg-delta, ci_high=avg+delta))
recalc = pd.DataFrame(computed)
check = ci_file.merge(recalc, on=keys+["metric"], suffixes=("_archive", "_recalc"), validate="one_to_one")
assert len(check)==len(ci_file)==720
maxdiff={}
for name in ("mean", "ci_low", "ci_high"):
    diff = (check[f"{name}_archive"]-check[f"{name}_recalc"]).abs().max()
    maxdiff[name]=float(diff)
assert max(maxdiff.values()) < 1e-8, maxdiff

report = {
    "source": "D7 delivered experimental CSVs; hashes recorded in manifest.csv",
    "utilprop_l1_loss_abs_change": {
        "selection": "mechanism=UtilProp, n=100, m=10, epsilon=0.10",
        "observations": len(sel),
        "cultures": 8,
        "repetitions_per_culture": 50,
        "mean": utilprop_mean,
        "rounded_in_thesis": 0.031671,
    },
    "historical_value_0_04097": {
        "status": "unattributed historical claim",
        "reason": "No matched raw observations, seed, and version log were supplied for this historical aggregate.",
        "is_causal_explanation_justified": False,
    },
    "n_growth_interval_recalculation": {
        "rows": len(check),
        "max_absolute_difference": maxdiff,
    },
}
(OUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("Provenance validated; UtilProp mean:", utilprop_mean)
print("Recomputed n-growth Student-t intervals:", len(check), maxdiff)
