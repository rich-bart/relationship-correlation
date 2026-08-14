"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Grid construction helpers for the MIC characteristic matrix.
"""

from __future__ import annotations

import numpy as np


def _grid_budget(n: int, alpha: float, max_bins: int | None) -> int:
    """Return the largest allowed product of grid dimensions."""
    budget = max(4, int(n**alpha))
    return min(budget, max_bins) if max_bins is not None else budget


def _candidate_grid_shapes(budget: int):
    """Yield all two-dimensional grids satisfying ``x_bins * y_bins <= B``."""
    for x_bins in range(2, budget // 2 + 1):
        for y_bins in range(2, budget // x_bins + 1):
            yield x_bins, y_bins


def _equipartition(values: np.ndarray, bins: int) -> np.ndarray:
    """Assign observations to near-equal-frequency bins without splitting ties."""
    if len(values) == 0 or np.all(values == values[0]):
        return np.zeros(len(values), dtype=np.int64)
    _, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse)
    mid_ranks = np.cumsum(counts) - counts / 2
    labels = np.floor(mid_ranks[inverse] * bins / len(values)).astype(np.int64)
    return np.minimum(labels, bins - 1)
