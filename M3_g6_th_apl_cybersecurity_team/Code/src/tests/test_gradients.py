"""Analytic gradients of pgm/dcd.py against finite differences (see scripts/gradcheck.py)."""
import numpy as np

from pgm.dcd import PENALTIES
from scripts.gradcheck import check_primal_loss, check_structure_penalty

TOL = 1e-5


def test_structure_penalty_gradients():
    for name in PENALTIES:
        e1, e2 = check_structure_penalty(name)
        assert e1 < TOL and e2 < TOL, (name, e1, e2)


def test_primal_loss_gradients_and_gram_path():
    allowed = np.ones((6, 6), dtype=bool)
    allowed[0, 3] = allowed[3, 0] = False
    for name in PENALTIES:
        err, pinned, gap = check_primal_loss(name, allowed=allowed)
        assert err < TOL, (name, err)
        assert pinned == 0.0
        assert gap < 1e-9          # Gram-matrix loss == direct pseudo-variable loss


def test_penalty_zero_iff_ancestral():
    from pgm.dcd import PENALTIES
    d = 5
    for name, fn in PENALTIES.items():
        W1 = np.zeros((d, d)); W1[0, 1] = W1[1, 2] = 1.0         # chain 0 -> 1 -> 2
        W2 = np.zeros((d, d)); W2[3, 4] = W2[4, 3] = 0.7          # 3 <-> 4 : fine
        assert abs(fn(W1, W2)[0]) < 1e-12, name
        W2b = W2.copy(); W2b[0, 2] = W2b[2, 0] = 0.7              # 0 <-> 2 with 0 ->..-> 2 : not ancestral
        assert fn(W1, W2b)[0] > 1e-2, name
        W1c = W1.copy(); W1c[2, 0] = 1.0                          # directed cycle
        assert fn(W1c, W2)[0] > 1e-2, name
