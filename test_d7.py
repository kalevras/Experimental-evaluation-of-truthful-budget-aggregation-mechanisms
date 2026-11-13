"""Deterministic D7 regression suite: mechanisms, external witnesses, invariants."""

from fractions import Fraction

import numpy as np
import pytest

from pb_core import (
    MECHANISMS, independent_markets, ladder, l1_loss, moving_phantom,
    overlap_utility, ph_independent_markets, piecewise_uniform,
    utilprop, validate_profile,
)


def utilprop_witness():
    """The explicit n=10, m=4 witness in thesis §7.4."""
    return np.array([[0, 0, 0, 1]] * 2 + [[.5, 0, .5, 0]] * 5
                    + [[0, 1, 0, 0]] * 3, dtype=float)


def im_witness():
    """Six-voter rational IM witness with exact L1-loss 4/5."""
    return np.array([[0, 0, 1, 0]] * 4 + [[.2, .2, .4, .2]] * 2,
                    dtype=float)


def test_original_profile_unchanged_after_clipping():
    """Assert that profile validation does not mutate the caller input."""
    p = np.array([[-5e-10, 1 + 5e-10], [.2, .8]])
    untouched = p.copy()
    cleaned = validate_profile(2, 2, p)
    assert np.array_equal(p, untouched)
    assert cleaned[0, 0] == 0
    assert not np.shares_memory(p, cleaned)


def test_nonconvergence_raises():
    """Assert that infeasible non-converged phantom normalization raises."""
    p = np.array([[1., 0.], [0., 1.]])
    with pytest.raises(ArithmeticError):
        moving_phantom(p, lambda t, n: [0.] * (n+1), max_iter=10)


def test_exact_witness_utilprop():
    """Verify the exact worst-case UtilProp witness for four projects."""
    p = utilprop_witness()
    a = utilprop(p)
    assert np.allclose(a, [.5, 0, .5, 0], atol=1e-9)
    assert abs(l1_loss(p, a) - 1) < 1e-9


def test_exact_witness_im():
    """Verify the rational Independent Markets witness with loss four fifths."""
    p = im_witness()
    a = independent_markets(p)
    assert np.allclose(a, [.2, .2, .4, .2], atol=1e-9)
    assert abs(l1_loss(p, a) - .8) < 1e-9


def test_phantom_medians_rational_im():
    """Check the IM phantom medians at the rational witness parameter."""
    reports = [[Fraction(0)] * 4 for _ in range(6)]
    for row in reports[:4]:
        row[2] = Fraction(1)
    for row in reports[4:]:
        row[:] = [Fraction(1, 5), Fraction(1, 5), Fraction(2, 5), Fraction(1, 5)]
    ph = [min(Fraction(1), k * Fraction(1, 10)) for k in range(7)]
    medians = []
    for j in range(4):
        vals = sorted([x[j] for x in reports] + ph)
        medians.append(vals[6])
    assert medians == [Fraction(1, 5), Fraction(1, 5),
                       Fraction(2, 5), Fraction(1, 5)]
    assert sum(medians) == 1


@pytest.mark.parametrize('m', [2, 3, 4, 7])
def test_single_minded_proportionality(m):
    """Check proportional allocations on single-minded voter profiles."""
    rng = np.random.default_rng(101 + m)
    for n in [3, 8, 21]:
        for _ in range(12):
            p = np.eye(m)[rng.integers(0, m, size=n)]
            expected = p.mean(axis=0)
            for mechanism in [independent_markets, ladder, utilprop, piecewise_uniform]:
                assert np.allclose(mechanism(p), expected, atol=1e-7), (n, m, mechanism)


def test_all_four_mpm_agree_for_two_projects():
    """Check the four proportional mechanisms agree for two projects."""
    rng = np.random.default_rng(617)
    for n in [3, 10, 21, 50]:
        for _ in range(25):
            p = rng.dirichlet([.8, 1.2], size=n)
            allocations = [fn(p) for fn in
                           (independent_markets, ladder, utilprop, piecewise_uniform)]
            assert all(np.allclose(a, allocations[0], atol=1e-7) for a in allocations)


def test_truthfulness_random_deviations():
    """Empirical manipulation check, not a mathematical proof of strategyproofness."""
    rng = np.random.default_rng(934)
    keys = ['im', 'ladder', 'utilprop', 'piecewise']
    for m in (2, 3, 4):
        for n in (3, 6, 9):
            for _ in range(25):
                p = rng.dirichlet(np.ones(m), size=n)
                true_p = p[0].copy()
                for key in keys:
                    mechanism = MECHANISMS[key][1]
                    truthful_allocation = mechanism(p)
                    truthful_utility = float(overlap_utility(p, truthful_allocation)[0])
                    q = p.copy()
                    q[0] = rng.dirichlet(np.ones(m))
                    deviation_allocation = mechanism(q)
                    deviating_utility = float(overlap_utility(p, deviation_allocation)[0])
                    assert deviating_utility <= truthful_utility + 1e-7, (key, n, m)
                    assert np.array_equal(p[0], true_p)


def test_regression_example_3_9_8_is_valid():
    """Reproduce the actual displayed n=4, m=3 example (rounded reports)."""
    p = np.array([[.5078, .4894, .0028], [.0112, .1364, .8523],
                  [.2437, .0026, .7537], [.0845, .1564, .7592]])
    p = p / p.sum(axis=1, keepdims=True)  # thesis prints only 4 decimals
    expected = {
        'mean': [.2118, .1962, .5920],
        'util': [.1099, .1364, .7537],
        'im': [.2437, .1891, .5673],
        'ladder': [.2437, .1564, .6000],
        'utilprop': [.1136, .1364, .7500],
        'piecewise': [.2291, .1564, .6145],
    }
    for key, (_, fn) in MECHANISMS.items():
        outcome = fn(p)
        assert abs(outcome.sum() - 1) < 1e-7
        assert np.max(abs(outcome - np.array(expected[key]))) < 0.0015, key
