"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Normalized grid mutual-information calculations for MIC.
"""

from __future__ import annotations

import numpy as np


def _normalized_grid_mi(
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    x_bins: int,
    y_bins: int,
) -> float:
    """Compute I(X;Y) / log(min(x_bins, y_bins)) for one grid."""
    if len(x_labels) == 0:
        return 0.0
    joint = np.bincount(
        x_labels * y_bins + y_labels, minlength=x_bins * y_bins
    ).reshape(x_bins, y_bins)
    joint = joint.astype(float) / len(x_labels)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    occupied = joint > 0
    information = np.sum(
        joint[occupied] * np.log(joint[occupied] / expected[occupied])
    )
    denominator = np.log(min(x_bins, y_bins))
    if denominator == 0:
        return 0.0
    return float(np.clip(information / denominator, 0.0, 1.0))
