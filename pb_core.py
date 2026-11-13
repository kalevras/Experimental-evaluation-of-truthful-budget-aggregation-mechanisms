"""Numerical reproducibility script for the participatory budgeting study."""

from __future__ import annotations
import numpy as np

TOL = 1e-9


def validate_profile(n: int, m: int, P, tol: float = 1e-8) -> np.ndarray:
    """Return a validated, independent copy of n simplex reports. Small negative floating error is clipped without mutating the input."""
    P = np.array(P, dtype=float, copy=True)
    if P.shape != (n, m):
        raise ValueError(f"Expected voting matrix shape ({n},{m}), got {P.shape}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("Voting matrix contains NaN/Inf.")
    if np.any(P < -tol):
        raise ValueError("All votes must be non-negative.")
    row_sums = P.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > tol)[0]
    if bad.size:
        sample = ", ".join(str(int(i+1)) for i in bad[:10])
        raise ValueError(f"Each voter row must sum to 1. Invalid row(s): {sample}")
    P[P < 0] = 0.0
    return P


def overlap_utility(P: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Compute each voter utility as coordinatewise overlap with allocation a."""
    return np.minimum(P, a).sum(axis=1)


def social_welfare(P: np.ndarray, a: np.ndarray) -> float:
    """Sum individual overlap utilities for the aggregate allocation."""
    return float(overlap_utility(P, a).sum())


def gini(values: np.ndarray) -> float:
    """Gini coefficient for non-negative individual utilities."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan")
    total = float(x.sum())
    if total <= TOL:
        return 0.0
    x = np.sort(x)
    n = x.size
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(weights, x) / (n * total)) - (n + 1.0) / n)


def fairness_metrics(P: np.ndarray, a: np.ndarray) -> dict:
    """Descriptive voter-level fairness metrics based on overlap utility."""
    u = overlap_utility(P, a)
    return {
        "utility_std": float(np.std(u)),
        "min_utility": float(np.min(u)),
        "p10_utility": float(np.quantile(u, 0.10)),
        "utility_gini": gini(u),
    }


def l1_loss(P: np.ndarray, a: np.ndarray) -> float:
    """Return L1 distance between a and the arithmetic mean of reports."""
    return float(np.abs(a - P.mean(axis=0)).sum())


def optimal_social_welfare(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact SW optimizer for overlap utility.

    For each project, SW is concave piecewise-linear in allocation. We create
    marginal segments and greedily fill the globally highest slopes. This is
    exact up to floating-point tolerance. Intended as an independent validator;
    for very large n*m it can be memory/time intensive.
    """
    n, m = P.shape
    segments = []  # (slope, width, project)
    for j in range(m):
        s = np.sort(P[:, j])
        prev = 0.0
        for r, x in enumerate(s):
            width = float(x - prev)
            if width > TOL:
                segments.append((n-r, width, j))
            prev = float(x)
    segments.sort(key=lambda z: z[0], reverse=True)
    a = np.zeros(m)
    remaining = 1.0
    for slope, width, j in segments:
        if remaining <= TOL:
            break
        take = min(width, remaining)
        a[j] += take
        remaining -= take
    if remaining > 1e-7:
        # Any leftover beyond all reported maxima adds no welfare; place it on
        # the smallest current coordinate to keep a valid distribution.
        j = int(np.argmin(a))
        a[j] += remaining
    return social_welfare(P, a), a


def allocation_from_phantoms(P: np.ndarray, phantoms) -> np.ndarray:
    """Compute the coordinatewise median of reports plus n+1 phantom values."""
    ph = np.asarray(phantoms, dtype=float)
    return np.array([np.median(np.concatenate((P[:, j], ph)))
                     for j in range(P.shape[1])], dtype=float)


def moving_phantom(P: np.ndarray, phantom_function, tol=1e-10, max_iter=100):
    """Bisect t so coordinatewise medians total one. Sum of medians is nondecreasing; tol controls stopping and 1e-7 controls final feasibility. Raise rather than return a bad allocation [1]."""
    n = P.shape[0]
    lo, hi = 0.0, 1.0
    best = None
    best_err = float("inf")
    for _ in range(max_iter):
        t = (lo + hi) / 2.0
        ph = np.asarray(phantom_function(t, n), dtype=float)
        a = allocation_from_phantoms(P, ph)
        s = float(a.sum())
        err = abs(s - 1.0)
        if err < best_err:
            best_err, best = err, (a.copy(), t)
        if err <= tol:
            break
        if s < 1.0:
            lo = t
        else:
            hi = t
    if best is None or best_err > 1e-7:
        raise ArithmeticError(
            f"Moving phantom failed to reach simplex tolerance: "
            f"|sum(a)-1|={best_err:.3g}, max_iter={max_iter}."
        )
    a, t = best
    # Only normalize a numerically negligible residual.
    return a / float(a.sum()), t


def mean_rule(P):
    """Arithmetic coordinatewise mean baseline; not generally truthful."""
    return P.mean(axis=0)


def ph_util(t, n):
    """Util phantoms max(0,min(1,(n+1)t-k)), k=0..n [1]."""
    return [max(0.0, min(1.0, t*(n+1)-k)) for k in range(n+1)]


def util(P):
    """Welfare-maximizing Util moving phantom under L1 preferences [1]."""
    return moving_phantom(P, ph_util)[0]


def ph_independent_markets(t, n):
    # Freeman et al. use {min(k*t,1): k=0,...,n}.  Reversing k gives the
    # identical multiset of phantom values, so medians and the outcome are
    # unchanged. We use the paper's indexing directly for transparency.
    """IM phantoms min(k*t,1), k=0..n; truthful and single-minded proportional [1]."""
    return [min(k * t, 1.0) for k in range(n + 1)]


def independent_markets(P):
    """Truthful single-minded proportional Independent Markets rule [1]."""
    return moving_phantom(P, ph_independent_markets)[0]


def ph_ladder(t, n):
    """Ladder phantoms max(t-k/n,0), k=0..n [20, Corollary 10]."""
    return [max(t-k/n, 0.0) for k in range(n+1)]


def ladder(P):
    """Truthful single-minded proportional Ladder moving phantom [20]."""
    return moving_phantom(P, ph_ladder)[0]


def ph_utilprop(t, n):
    """UtilProp phantoms max(0,min((n-k)/n,(n+1)t-k)), k=0..n [3]."""
    return [max(0.0, min((n-k)/n, t*(n+1)-k)) for k in range(n+1)]


def utilprop(P):
    """Truthful single-minded proportional UtilProp rule [3]."""
    return moving_phantom(P, ph_utilprop)[0]


def ph_piecewise_uniform(t, n):
    """Piecewise-Uniform phantom system (Caragiannis et al., 2024).

    Index convention: k=0,...,n, exactly as in the paper.
    For t < 1/2:
      y_k(t)=0                              if k/n < 1/2
      y_k(t)=4*t*k/n - 2*t                 if k/n >= 1/2
    For t >= 1/2:
      y_k(t)=k*(2*t-1)/n                   if k/n < 1/2
      y_k(t)=k*(3-2*t)/n - 2 + 2*t         if k/n >= 1/2

    The mechanism can be evaluated computationally for any m; the specific
    L1-loss guarantee used in the thesis is the published m=3 result.
    """
    vals = []
    for k in range(n + 1):
        ratio = k / n
        if t < 0.5:
            v = 0.0 if ratio < 0.5 else 4.0 * t * ratio - 2.0 * t
        else:
            v = (ratio * (2.0 * t - 1.0) if ratio < 0.5
                 else ratio * (3.0 - 2.0 * t) - 2.0 + 2.0 * t)
        vals.append(max(0.0, min(1.0, v)))
    return vals


def piecewise_uniform(P):
    # Defined for any m. The specific L1 theoretical guarantee used in the
    # thesis is the published result for m=3.
    """Evaluate Piecewise Uniform; known tight L1 bounds here only for m=3 [2]."""
    return moving_phantom(P, ph_piecewise_uniform)[0]


MECHANISMS = {
    "mean": ("Mean Rule", mean_rule),
    "util": ("Util", util),
    "im": ("Independent Markets", independent_markets),
    "ladder": ("Ladder", ladder),
    "utilprop": ("UtilProp", utilprop),
    "piecewise": ("Piecewise Uniform", piecewise_uniform),
}


def evaluate(P: np.ndarray, selected, compute_optimum=True):
    """Evaluate requested mechanisms against matched profiles and optional exact SW optimum."""
    selected = list(selected)
    sw_star = None
    a_star = None
    # Util is theoretically welfare-optimal, but we keep an independent exact
    # optimizer for validation whenever requested.
    if compute_optimum:
        sw_star, a_star = optimal_social_welfare(P)
    rows = []
    allocations = {}
    for key in selected:
        if key not in MECHANISMS:
            raise ValueError(f"Unknown mechanism: {key}")
        name, fn = MECHANISMS[key]
        a = fn(P)
        allocations[key] = a
        sw = social_welfare(P, a)
        row = {
            "mechanism": name,
            "allocation": "[" + ", ".join(f"{x:.8f}" for x in a) + "]",
            "l1_loss": l1_loss(P, a),
            "social_welfare": sw,
            "avg_utility": sw / len(P),
            "validity_error": abs(float(a.sum())-1.0),
            **fairness_metrics(P, a),
        }
        if sw_star is not None:
            row["optimal_sw"] = sw_star
            if sw_star:
                gap = sw_star - sw
                # Remove harmless floating-point noise around the theoretical boundary.
                if abs(gap) <= 1e-9:
                    gap = 0.0
                ratio = sw / sw_star
                if abs(ratio - 1.0) <= 1e-9:
                    ratio = 1.0
                row["sw_ratio"] = ratio
                row["welfare_gap_pct"] = 100 * gap / sw_star
            else:
                row["sw_ratio"] = np.nan
                row["welfare_gap_pct"] = np.nan
        rows.append(row)
    return rows, allocations, sw_star, a_star
