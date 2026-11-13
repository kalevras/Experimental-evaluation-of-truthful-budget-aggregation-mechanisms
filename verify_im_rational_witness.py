#!/usr/bin/env python3
"""Self-contained exact certificate for the thesis's IM adverse profile.

Requires Python standard library only. Reproduces exact arithmetic with
fractions.Fraction, avoiding numerical bisection and rounding.
"""
from fractions import Fraction as Q
from statistics import median
import json
from pathlib import Path


def verify_im(m: int = 4):
    """Prove the rational IM example by recomputing medians and loss exactly."""
    if m < 4:
        raise ValueError('At least four projects are needed')
    P = [[Q(0)] * m for _ in range(6)]
    for row in P[:4]:
        row[2] = Q(1)
    for row in P[4:]:
        row[:4] = [Q(1, 5), Q(1, 5), Q(2, 5), Q(1, 5)]
    assert all(sum(row) == 1 for row in P)
    t = Q(1, 10)
    ph = [min(Q(1), k * t) for k in range(7)]
    a = [median([row[j] for row in P] + ph) for j in range(m)]
    mu = [sum(row[j] for row in P) / len(P) for j in range(m)]
    loss = sum(abs(a[j] - mu[j]) for j in range(m))
    assert sum(a) == Q(1)
    assert a[:4] == [Q(1, 5), Q(1, 5), Q(2, 5), Q(1, 5)]
    assert mu[:4] == [Q(1, 15), Q(1, 15), Q(4, 5), Q(1, 15)]
    assert loss == Q(4, 5)
    return dict(m=m,n=6,parameter_t=str(t),phantoms=list(map(str,ph)),
                allocation=list(map(str,a)),mean=list(map(str,mu)),
                l1_loss_exact=str(loss),l1_loss_float=float(loss))

if __name__ == '__main__':
    for m in [4,5,6]:
        print(json.dumps(verify_im(m),ensure_ascii=False,indent=2))
