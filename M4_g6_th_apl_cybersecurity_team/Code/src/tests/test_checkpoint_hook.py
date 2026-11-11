"""M4 WP0: the anytime hook observes the optimiser without changing it."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import dataclasses
import numpy as np
from pgm.dcd import DCDConfig, discover_admg
from skeleton.simulator import make_instances


def _data():
    ins = make_instances(1, d_range=(8, 8), n_range=(300, 300), seed=11)[0]
    X = (ins.X - ins.X.mean(0)) / ins.X.std(0)
    return X, [f"v{i}" for i in range(X.shape[1])]


def test_hook_does_not_change_the_result_and_fires_once_per_outer_iteration():
    X, names = _data()
    base = DCDConfig(seed=3, max_outer=6, lbfgs_maxiter=80)
    r0 = discover_admg(X, names, base)
    calls = []
    cb = lambda t, W1, W2, h, outer, restart: calls.append((t, W1, W2, h, outer, restart))
    r1 = discover_admg(X, names, dataclasses.replace(base, checkpoint_cb=cb))
    assert np.array_equal(r0.W1_raw, r1.W1_raw) and np.array_equal(r0.W2_raw, r1.W2_raw) and r0.h == r1.h and r0.n_fev == r1.n_fev
    assert len(calls) == r1.n_outer and [c[4] for c in calls] == list(range(1, r1.n_outer + 1))
    ts = [c[0] for c in calls]
    assert all(b >= a for a, b in zip(ts, ts[1:])) and ts[0] >= 0 and ts[-1] <= r1.elapsed + 1e-6
    assert np.array_equal(calls[-1][1], r1.W1_raw)                         # the last checkpoint is the final iterate


def test_hook_receives_copies():
    X, names = _data()
    seen = []
    def cb(t, W1, W2, h, outer, restart):
        seen.append(W1)
        W1[:] = 0.0                                                         # a careless callback must not corrupt the run
    base = DCDConfig(seed=3, max_outer=4, lbfgs_maxiter=60)
    r0 = discover_admg(X, names, base)
    r1 = discover_admg(X, names, dataclasses.replace(base, checkpoint_cb=cb))
    assert np.array_equal(r0.W1_raw, r1.W1_raw) and len(seen) == r1.n_outer
