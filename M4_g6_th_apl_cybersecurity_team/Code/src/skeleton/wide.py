"""M4 wide-d training / calibration / test sets for the skeleton posterior, with disjoint seed namespaces.

    training     seeds       0 ..  9_999   (scripts/train_posterior_wide.py)
    calibration  seeds  10_000 .. 19_999   (floor calibration)
    test         seeds  20_000 and above   (posterior_by_d, scalability sweep)

The graph family is the M3 one (simulate_instance: 1.0-1.5 edges per node, 0-15 % bidirected, n log-uniform 300-2000,
AR rho in {0, .3, .6}); only the number of nodes changes.  No PetShop data or label is used anywhere here.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .simulator import Instance, simulate_instance

SPLITS = {"train": (0, 10_000), "calib": (10_000, 20_000), "test": (20_000, 10 ** 9)}
AR_RHOS = (0.0, 0.3, 0.6)
N_RANGE = (300, 2000)

# bucket name -> (d_min, d_max, number of TRAINING graphs).  "m3" is the old range, so the wide model sees it too.
TRAIN_BUCKETS = {"m3": (25, 40, 100), "d60": (60, 60, 60), "d100": (100, 100, 60), "d150": (150, 150, 40), "d200": (200, 200, 40)}
_BUCKET_IDX = {k: i for i, k in enumerate(["m3", "d60", "d100", "d150", "d200", "d25", "d40", "t"])}


def seed_for(split: str, bucket: str, idx: int) -> int:
    """Unique seed of graph ``idx`` of ``bucket`` in ``split``; asserts the namespace is respected."""
    lo, hi = SPLITS[split]
    s = lo + 1000 * _BUCKET_IDX[bucket] + idx
    assert idx < 1000, "at most 999 graphs per bucket"
    assert lo <= s < hi, f"seed {s} outside the {split} namespace [{lo}, {hi})"
    return s


def one_instance(split: str, bucket: str, idx: int, d_min: int, d_max: int) -> Instance:
    rng = np.random.default_rng(seed_for(split, bucket, idx))
    d = int(rng.integers(d_min, d_max + 1))
    n = int(np.exp(rng.uniform(np.log(N_RANGE[0]), np.log(N_RANGE[1]))))
    rho = float(AR_RHOS[rng.integers(len(AR_RHOS))])
    return simulate_instance(d, n, rng, ar_rho=rho)


def wide_instances(split: str, buckets=None, scale: float = 1.0) -> List[Tuple[str, Instance]]:
    """(bucket, instance) for every graph of the split.  ``scale`` shrinks the counts (tests, smoke runs)."""
    buckets = buckets or TRAIN_BUCKETS
    out = []
    for name, (lo, hi, cnt) in buckets.items():
        for i in range(max(1, int(round(cnt * scale)))):
            out.append((name, one_instance(split, name, i, lo, hi)))
    return out
