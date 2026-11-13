#!/usr/bin/env python3
"""Verify every archived q=240 refined witness with the current pb_core.

The 3,680-run CSV is checked for integrity and its reported summaries;
full 11 million candidate evaluations are NOT rerun by this fast check.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pb_core import MECHANISMS, l1_loss


def main():
    """Recompute archived witnesses and compare with stored L1-loss values."""
    files = Path(__file__).resolve().parent / 'results'
    raw = pd.read_csv(files / 'full_restarts.csv')
    verified = pd.read_csv(files / 'q240_refinement.csv')
    assert len(raw) == 3680, len(raw)
    assert len(verified) == 56, len(verified)
    assert set(verified.mechanism) == {'im', 'ladder', 'utilprop', 'piecewise'}
    failures = []
    max_error = 0
    for i, row in verified.iterrows():
        reports = np.array(json.loads(row.Y), dtype=float)
        weights = np.array(json.loads(row.counts), dtype=int)
        assert len(weights) == len(reports) == int(row.K) and weights.sum() == int(row.n)
        assert reports.shape[1] == int(row.m)
        assert np.allclose(reports.sum(axis=1), 1, atol=1e-10)
        p = np.repeat(reports, weights, axis=0)
        alloc = MECHANISMS[row.mechanism][1](p)
        actual = l1_loss(p, alloc)
        error = abs(actual - float(row.loss_q240))
        max_error = max(error, max_error)
        if error > 1e-7:
            failures.append((i, row.mechanism, row.n, error))
    if failures:
        raise ArithmeticError(f'{len(failures)} witness mismatches: {failures[:5]}')
    outcomes = verified.groupby(['m', 'mechanism']).loss_q240.max().to_dict()
    report = dict(verified_profiles=len(verified), original_runs=len(raw),
                  max_l1_loss_abs_error=max_error,
                  best_m4_im=float(outcomes[4, 'im']),
                  best_m4_utilprop=float(outcomes[4, 'utilprop']),
                  historical_search_rerun=False,
                  meaning='Independent verification of archived witnesses, not fresh search')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not argparse.ArgumentParser().parse_args().__dict__:
        pass


if __name__ == '__main__':
    main()
