"""Fast regression tests for the thesis implementation."""
import numpy as np
from pb_core import (MECHANISMS, evaluate, optimal_social_welfare, social_welfare,
                     validate_profile)
from pb_app import perturb_profile
from pb_cultures import generate


def run_tests():
    """Exercise original numerical validity and welfare regression assertions."""
    rng = np.random.default_rng(20261004)
    cultures = ['random','balanced','single_minded','polarized','majority_minority','mixed','sparse','concentrated','characteristic']
    cases = 0
    for culture in cultures:
        for n in (5, 17, 40):
            for m in (3, 5, 10):
                P = generate(culture,n,m,rng)
                validate_profile(n,m,P)
                rows, allocs, sw_star, _ = evaluate(P, MECHANISMS.keys(), compute_optimum=True)
                for key,a in allocs.items():
                    assert np.all(np.isfinite(a)), (culture,n,m,key)
                    assert np.min(a) >= -1e-9, (culture,n,m,key,a)
                    assert abs(float(a.sum())-1.0) <= 1e-7, (culture,n,m,key,a.sum())
                # Independent optimum and Util must agree in welfare.
                util_sw = social_welfare(P, allocs['util'])
                assert abs(util_sw-sw_star) <= 1e-7*max(1.0,sw_star), (culture,n,m,util_sw,sw_star)
                # Mean Rule is exactly the mean benchmark.
                mean_row = next(r for r in rows if r['mechanism']=='Mean Rule')
                assert abs(mean_row['l1_loss']) <= 1e-12
                for eps in (0.01,0.05,0.10,1.0):
                    Q=perturb_profile(P,eps,rng)
                    shifts=np.abs(Q-P).sum(axis=1)
                    assert np.allclose(shifts,eps,atol=1e-10,rtol=0), (culture,n,m,eps,shifts)
                cases += 1
    print(f'PASS: {cases} profile settings; all mechanisms valid; Util=SW*; exact robustness perturbations verified.')

if __name__ == '__main__':
    run_tests()
