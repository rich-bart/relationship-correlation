"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Entropy and discrete mutual-information calculations.
"""

import numpy as np

from models.mutual_information.encoding import _factorize


def _entropy(labels: np.ndarray, base: float) -> float:
    """Calculate entropy for factorized labels."""
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    probabilities = counts[counts > 0] / len(labels)
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(base))


def _discrete_mi(
    x: np.ndarray, y: np.ndarray, base: float
) -> tuple[float, float, float]:
    """Calculate MI and individual entropies for two discrete variables."""
    x_labels = _factorize(x)
    y_labels = _factorize(y)
    nx = int(x_labels.max(initial=-1)) + 1
    ny = int(y_labels.max(initial=-1)) + 1
    joint = np.bincount(x_labels * ny + y_labels, minlength=nx * ny)
    joint = joint.reshape(nx, ny).astype(float)
    joint /= len(x_labels)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    nonzero = joint > 0
    mi = float(
        (joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])).sum()
        / np.log(base)
    )
    # Clamp tiny negative round-off errors to zero.
    return max(0.0, mi), _entropy(x_labels, base), _entropy(y_labels, base)
