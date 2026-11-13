"""Synthetic preference profile generators and characteristic archetypes."""

import numpy as np


def _normalize(x):
    """Convert nonnegative weights to a unit-sum budget allocation."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = x.sum()
    if s <= 0:
        return np.full(x.size, 1.0 / x.size)
    return x / s


def characteristic_archetypes(m):
    """Deterministic library of interpretable voter archetypes for any m.

    For m>5 the library deliberately includes concentrated, two/three-project,
    block, uniform, head-heavy, tail-heavy and smoothly decreasing profiles.
    Project labels are not privileged in experiments because sampled archetypes
    are randomly permuted before use.
    """
    if m < 2:
        return np.ones((1, 1))
    A = []
    A.append(np.full(m, 1.0 / m))                         # uniform

    # Concentrated profiles, initially anchored to the first coordinates.
    for weights in ([1.0], [0.8, 0.2], [0.6, 0.4], [0.7, 0.2, 0.1],
                    [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1]):
        k = min(len(weights), m)
        x = np.zeros(m); x[:k] = np.asarray(weights[:k], float)
        A.append(_normalize(x))

    # Equal support on k projects: useful especially when m>5.
    ks = sorted(set([2, 3, min(5, m), max(2, m // 4), max(2, m // 2), m]))
    for k in ks:
        k = min(max(1, k), m)
        x = np.zeros(m); x[:k] = 1.0 / k
        A.append(x)

    # Smoothly decreasing / increasing preference shapes.
    A.append(_normalize(np.arange(m, 0, -1, dtype=float)))
    A.append(_normalize(np.arange(1, m + 1, dtype=float)))
    A.append(_normalize(np.geomspace(1.0, 0.05, m)))

    # Block preferences. These are particularly informative for m>5.
    if m > 5:
        for frac in (0.2, 0.33, 0.5):
            k = max(2, min(m, int(round(frac * m))))
            x = np.zeros(m); x[:k] = 1.0 / k; A.append(x)
        # Two separated blocks with unequal mass.
        k = max(1, m // 4)
        x = np.zeros(m); x[:k] = 0.7 / k; x[-k:] += 0.3 / k; A.append(_normalize(x))

    # Remove numerical duplicates.
    uniq = []
    for x in A:
        x = _normalize(x)
        if not any(np.allclose(x, y, atol=1e-12) for y in uniq):
            uniq.append(x)
    return np.asarray(uniq)


def generate_characteristic(n, m, rng, max_types=100, jitter=0.03, return_metadata=False):
    """Sample a population from interpretable characteristic voter types.

    When ``return_metadata`` is True, also return auditable records describing
    the base archetype, project permutation, jitter strength and final voter type.
    This does not consume any extra random numbers, so the generated profile is
    identical for the same seed and parameters.
    """
    if max_types < 1:
        raise ValueError('max_types must be positive.')
    if not (0.0 <= jitter <= 1.0):
        raise ValueError('jitter must be in [0,1].')
    base = characteristic_archetypes(m)
    pool_size = min(int(max_types), max(n, 1))
    pool, pool_meta = [], []
    for i in range(pool_size):
        base_id = i % len(base)
        base_x = base[base_id].copy()
        perm = rng.permutation(m)
        shuffled = base_x[perm]
        x = shuffled.copy()
        eps = 0.0
        if jitter > 0:
            noise = rng.dirichlet(np.ones(m))
            eps = float(rng.uniform(0.0, jitter))
            x = (1.0 - eps) * x + eps * noise
        x = _normalize(x)
        pool.append(x)
        meta = {
            'type_id': i,
            'base_archetype_id': base_id,
            'permutation': ','.join(map(str, perm.tolist())),
            'jitter_epsilon': eps,
            'base_preferences': ','.join(f'{v:.12g}' for v in base_x),
            'shuffled_preferences': ','.join(f'{v:.12g}' for v in shuffled),
            'final_type_preferences': ','.join(f'{v:.12g}' for v in x),
        }
        meta.update({f'p{j+1}': float(v) for j, v in enumerate(x)})
        pool_meta.append(meta)
    pool = np.asarray(pool)
    idx = rng.integers(0, len(pool), size=n)
    P = pool[idx].copy()
    if not return_metadata:
        return P
    voters = []
    for voter_id, type_id in enumerate(idx, 1):
        rec = dict(pool_meta[int(type_id)])
        rec['voter_id'] = voter_id
        voters.append(rec)
    return P, voters, pool_meta


def generate(culture, n, m, rng, max_types=100, jitter=0.03, return_metadata=False):
    """Sample a validated preference culture with the requested dimensions."""
    culture = culture.lower().replace('-', '_')
    if culture == "random":
        return rng.dirichlet(np.ones(m), size=n)
    if culture == "balanced":
        alpha = np.full(m, 8.0)
        return rng.dirichlet(alpha, size=n)
    if culture == "single_minded":
        P = np.zeros((n,m)); choices = rng.integers(0,m,size=n); P[np.arange(n),choices]=1
        return P
    if culture == "polarized":
        P = np.zeros((n,m)); a,b=0,max(0,m-1)
        choices = rng.integers(0,2,size=n)
        P[np.arange(n), np.where(choices==0,a,b)] = 1
        return P
    if culture == "majority_minority":
        P = np.zeros((n,m)); cut=int(round(.7*n)); P[:cut,0]=1; P[cut:,min(1,m-1)]=1
        rng.shuffle(P, axis=0); return P
    if culture == "mixed":
        half=n//2
        return np.vstack([rng.dirichlet(np.ones(m), size=half),
                          generate("single_minded", n-half, m, rng)])
    if culture == "sparse":
        P=np.zeros((n,m)); k=min(3,m)
        for i in range(n):
            idx=rng.choice(m,k,replace=False); P[i,idx]=rng.dirichlet(np.ones(k))
        return P
    if culture == "concentrated":
        return rng.dirichlet(np.full(m,0.25), size=n)
    if culture in {"characteristic", "archetype", "archetypes"}:
        return generate_characteristic(n, m, rng, max_types=max_types, jitter=jitter, return_metadata=return_metadata)
    raise ValueError(f"Unknown culture '{culture}'")
