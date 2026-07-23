"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Private helpers used by :mod:`mutual_information`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _is_missing(value: Any) -> bool:
    """Return whether *value* is a scalar missing value."""
    if value is None:
        return True
    try:
        result = value != value
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _factorize(values: np.ndarray) -> np.ndarray:
    """Factorize values without requiring mutually orderable objects."""
    labels: dict[Any, int] = {}
    result = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        try:
            result[i] = labels.setdefault(value, len(labels))
        except TypeError:
            # Unhashable category values are unusual, but can still be handled.
            key = (type(value).__qualname__, repr(value))
            result[i] = labels.setdefault(key, len(labels))
    return result


def _discretize(
    values: np.ndarray, bins: str | int | Sequence[float]
) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if len(numeric) == 0 or np.all(numeric == numeric[0]):
        return np.zeros(len(numeric), dtype=np.int64)
    edges = np.histogram_bin_edges(numeric, bins=bins)
    # Internal edges are enough; this also avoids a special overflow category.
    return np.searchsorted(edges[1:-1], numeric, side="right")


def _entropy(labels: np.ndarray, base: float) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    probabilities = counts[counts > 0] / len(labels)
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(base))


def _discrete_mi(
    x: np.ndarray, y: np.ndarray, base: float
) -> tuple[float, float, float]:
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


def _infer_discrete(column: np.ndarray) -> bool:
    present = [value for value in column if not _is_missing(value)]
    if not present:
        return True
    if any(isinstance(value, (str, bytes, bool)) for value in present):
        return True
    try:
        numeric = np.asarray(present, dtype=float)
    except (TypeError, ValueError):
        return True
    unique = len(np.unique(numeric))
    # Small integer domains are usually encoded categorical variables.
    return bool(
        np.all(np.equal(numeric, np.floor(numeric)))
        and unique <= min(20, max(2, int(np.sqrt(len(numeric)))))
    )
