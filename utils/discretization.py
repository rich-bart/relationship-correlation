"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Continuous-variable discretization helpers.
"""

from collections.abc import Sequence

import numpy as np


def _discretize(
    values: np.ndarray, bins: str | int | Sequence[float]
) -> np.ndarray:
    """Convert continuous values to histogram bin labels."""
    numeric = np.asarray(values, dtype=float)
    if len(numeric) == 0 or np.all(numeric == numeric[0]):
        return np.zeros(len(numeric), dtype=np.int64)
    edges = np.histogram_bin_edges(numeric, bins=bins)
    # Internal edges are enough; this also avoids a special overflow category.
    return np.searchsorted(edges[1:-1], numeric, side="right")
