"""Supplementary stochastic search for an Independent Markets adverse profile.

This exploratory procedure changes exactly one voter per proposal and uses a
simulated-annealing acceptance rule: it is NOT the weighted K-type hill-climber
responsible for thesis Figures 7.1–7.2. The original K-type source and saved
runs are separately included in hill_climbing/original_code and results.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from pb_core import independent_markets, l1_loss

# Candidate proposal distribution: new Dirichlet row / vertex / mass transfer.
DIRICHLET_PROBABILITY = 0.18
VERTEX_PROBABILITY = 0.14
MASS_TRANSFER_PROBABILITY = 0.68
DIRICHLET_ALPHAS = (0.03, 0.1, 0.3, 1.0)
INITIAL_ALPHAS = (0.05, 0.2, 1.0)
MAX_TRANSFER = 0.6
DEFAULT_TEMPERATURE = 0.002


def evaluate(profile: np.ndarray):
    """Return loss, allocation and mean for an IM voting profile."""
    allocation = np.asarray(independent_markets(profile), dtype=float)
    mean = np.mean(profile, axis=0)
    loss = float(l1_loss(profile, allocation))
    return loss, allocation, mean


def candidate(profile: np.ndarray, rng: np.random.Generator):
    """Propose one changed voter: Dirichlet (18%), vertex (14%), transfer (68%).

    All moves preserve the simplex and modify one randomly selected voter.
    Dirichlet draws use alpha in {0.03,0.1,0.3,1}; vertex chooses one project;
    mass transfer moves a feasible amount between two distinct projects.
    """
    proposal = profile.copy()
    i = int(rng.integers(proposal.shape[0]))
    r = rng.random()
    if r < DIRICHLET_PROBABILITY:
        alpha = float(rng.choice(DIRICHLET_ALPHAS))
        proposal[i] = rng.dirichlet(np.full(proposal.shape[1], alpha))
    elif r < DIRICHLET_PROBABILITY + VERTEX_PROBABILITY:
        proposal[i] = np.eye(proposal.shape[1])[int(rng.integers(proposal.shape[1]))]
    else:
        j, k = rng.choice(proposal.shape[1], size=2, replace=False)
        shift = float(rng.uniform(-MAX_TRANSFER, MAX_TRANSFER))
        shift = float(np.clip(shift, -proposal[i, j], proposal[i, k]))
        proposal[i, j] += shift
        proposal[i, k] -= shift
    return proposal


def search(args):
    """Stochastic multi-start search; record best feasible observed witness only.

    An inferior candidate may be accepted with probability exp((new-old)/T).
    Linearly cooling T starts at args.temperature (default 0.002).
    This heuristic establishes no worst-case upper bound.
    """
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best_loss, best = -1.0, None
    trace, evaluations = [], 0
    start = time.monotonic()
    for restart in range(args.restarts):
        n = int(rng.choice(args.voters))
        if rng.random() < 0.5:
            profile = np.eye(args.projects)[rng.integers(args.projects, size=n)].astype(float)
        else:
            alpha = float(rng.choice(INITIAL_ALPHAS))
            profile = rng.dirichlet(np.full(args.projects, alpha), size=n)
        loss, allocation, mean = evaluate(profile)
        evaluations += 1
        for step in range(args.steps):
            proposed = candidate(profile, rng)
            proposed_loss, proposed_allocation, proposed_mean = evaluate(proposed)
            evaluations += 1
            temperature = args.temperature * (1 - step / max(args.steps, 1))
            if proposed_loss > loss + 1e-12 or (
                temperature > 0
                and rng.random() < np.exp(min(0, (proposed_loss - loss) / temperature))
            ):
                profile, loss, allocation, mean = (
                    proposed, proposed_loss, proposed_allocation, proposed_mean
                )
            if loss > best_loss + 1e-12:
                best_loss, best = loss, profile.copy()
                np.savez(out / 'best_witness.npz', profile=best, allocation=allocation,
                         mean=mean, loss=best_loss)
                trace.append(dict(restart=restart, step=step,
                                  evaluations=evaluations, l1_loss=best_loss))
        print(f'restart {restart + 1}/{args.restarts} | best L1={best_loss:.9f}',
              flush=True)
    if best is None:
        raise ValueError('No candidate evaluated: increase restarts/steps.')
    verified, allocation, mean = evaluate(best)
    assert np.all(best >= -1e-10) and np.allclose(best.sum(axis=1), 1, atol=1e-8)
    assert np.isclose(verified, best_loss, atol=1e-9)
    record = dict(
        mechanism='Independent Markets', method='exploratory simulated annealing',
        seed=args.seed, m=args.projects, n=len(best), restarts=args.restarts,
        steps=args.steps, evaluations=evaluations,
        elapsed_seconds=round(time.monotonic() - start, 3), l1_loss=verified,
        allocation=allocation.tolist(), mean=mean.tolist(), profile=best.tolist(),
        verified_against_pb_core=True,
        note='Observed numerical witness, NOT a proof of a global worst case.'
    )
    (out / 'witness.json').write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    with (out / 'voter_profile.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['voter'] + [f'project_{j + 1}' for j in range(args.projects)])
        for i, row in enumerate(best, 1):
            writer.writerow([i] + [f'{v:.16g}' for v in row])
    with (out / 'best_trace.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['restart', 'step', 'evaluations', 'l1_loss'])
        writer.writeheader()
        writer.writerows(trace)
    print('\nVERIFIED WITNESS:', json.dumps({
        'l1_loss': verified, 'n': len(best), 'm': args.projects,
        'allocation': allocation.tolist(), 'mean': mean.tolist(),
        'output': str(out)}, indent=2))


def verify(path):
    """Reevaluate a previously saved NPZ witness against the current core code."""
    data = np.load(path)
    profile = np.asarray(data['profile'] if 'profile' in data else data['P'], dtype=float)
    loss, allocation, mean = evaluate(profile)
    print(json.dumps(dict(l1_loss=loss, allocation=allocation.tolist(), mean=mean.tolist(),
                          n=len(profile), m=profile.shape[1],
                          rows_sum_to_one=bool(np.allclose(profile.sum(axis=1), 1))), indent=2))


def main():
    """Parse CLI for supplemental search or saved-witness verification."""
    parser = argparse.ArgumentParser(description='D7: optional annealing-based IM search')
    commands = parser.add_subparsers(dest='command', required=True)
    hill = commands.add_parser('hill-im', help='Search for high L1-loss IM profiles')
    hill.add_argument('--projects', type=int, default=4)
    hill.add_argument('--voters', nargs='+', type=int, default=[4, 5, 6, 8, 10, 12, 16, 20])
    hill.add_argument('--restarts', type=int, default=80)
    hill.add_argument('--steps', type=int, default=220)
    hill.add_argument('--seed', type=int, default=8675309)
    hill.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    hill.add_argument('--output', default='D7_results/hill_im')
    check = commands.add_parser('verify-im', help='Verify a saved NPZ witness')
    check.add_argument('npz')
    args = parser.parse_args()
    if args.command == 'hill-im':
        search(args)
    else:
        verify(args.npz)


if __name__ == '__main__':
    main()
