"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Characteristic-matrix calculation for the MIC estimator.
"""

from __future__ import annotations

import numpy as np

from models.maximal_information_coefficient.grid_search import (
    _candidate_grid_shapes,
    _equipartition,
)
from models.maximal_information_coefficient.normalization import (
    _normalized_grid_mi,
)


def characteristic_matrix(
    x: np.ndarray, y: np.ndarray, budget: int
) -> np.ndarray:
    """Return normalized scores indexed by their x/y grid dimensions.

    Entries for inadmissible grids are ``nan``. Quantile grids provide a
    deterministic, dependency-free approximation to the exhaustive MIC grid
    optimization.
    """
    matrix = np.full((budget + 1, budget + 1), np.nan, dtype=float)
    x_partitions: dict[int, np.ndarray] = {}
    y_partitions: dict[int, np.ndarray] = {}
    for x_bins, y_bins in _candidate_grid_shapes(budget):
        x_labels = x_partitions.setdefault(x_bins, _equipartition(x, x_bins))
        y_labels = y_partitions.setdefault(y_bins, _equipartition(y, y_bins))
        matrix[x_bins, y_bins] = _normalized_grid_mi(
            x_labels, y_labels, x_bins, y_bins
        )
    return matrix
