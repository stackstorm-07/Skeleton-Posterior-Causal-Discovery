"""ABIC (Bhattacharya et al., AISTATS 2021): the shared optimizer in dcd.py with no gradient mask."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .dcd import DCDConfig, DCDResult, discover_admg


def run_abic(X: np.ndarray, names: Sequence[str], cfg: Optional[DCDConfig] = None,
             allowed: Optional[np.ndarray] = None) -> DCDResult:
    """Plain ABIC.  ``allowed`` (optional) is a hard skeleton constraint, used only by ablations."""
    return discover_admg(X, names, cfg, mask=None, allowed=allowed)
