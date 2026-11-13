"""Independent reproducible small-dimensional UtilProp adverse-profile search.

Uses exactly one voter mutation per proposal and accepts improvements only;
these are new, finite exploratory runs, not a global worst-case computation.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pb_core import utilprop, l1_loss
from D6 import candidate


def run(projects, seed, restarts=40, steps=150):
    """Perform a bounded stochastic UtilProp profile search in a chosen dimension."""
    rng = np.random.default_rng(seed)
    best_loss = -1
    best = None
    for restart in range(restarts):
        n = int(rng.choice([6, 10, 12, 20]))
        if restart % 2:
            p = np.eye(projects)[rng.integers(projects, size=n)]
        else:
            p = rng.dirichlet(np.full(projects, float(rng.choice([0.05, .2, 1]))), size=n)
        loss = l1_loss(p, utilprop(p))
        for step in range(steps):
            q = candidate(p, rng)
            qloss = l1_loss(q, utilprop(q))
            if qloss > loss + 1e-12:
                p, loss = q, qloss
            if loss > best_loss + 1e-12:
                best_loss, best = loss, p.copy()
    return dict(method='greedy single-voter hill climbing',
                projects=projects, seed=seed, restarts=restarts, steps=steps,
                best_loss=best_loss, voters=len(best),
                profile=best.tolist(),
                allocation=utilprop(best).tolist(),
                note='Exploratory observed lower bound, not global maximum')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--restarts', type=int, default=40)
    p.add_argument('--steps', type=int, default=150)
    args = p.parse_args()
    data = [run(m, seed=20261109 + m, restarts=args.restarts, steps=args.steps) for m in (5,6)]
    out = ROOT / 'data' / 'supplementary' / 'utilprop_hill_m5_m6.json'
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([dict(projects=x['projects'], best_loss=x['best_loss'],
                           voters=x['voters']) for x in data], indent=2))
