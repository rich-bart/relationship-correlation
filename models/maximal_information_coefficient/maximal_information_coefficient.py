"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Public maximal-information-coefficient APIs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from models.maximal_information_coefficient.characteristic_matrix import (
    characteristic_matrix,
)
from models.maximal_information_coefficient.grid_search import _grid_budget
from models.maximal_information_coefficient.validation import (
    _prepare_pair,
    _validate_parameters,
)
from models.mutual_information.missing_values import _is_missing

__all__ = [
    "maximal_information_coefficient",
    "maximal_information_coefficient_matrix",
    "mic_matrix",
]


def maximal_information_coefficient(
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    alpha: float = 0.6,
    max_bins: int | None = None,
    missing: Literal["drop", "raise"] = "drop",
) -> float:
    """Estimate the maximal information coefficient between two variables.

    The estimator searches equal-frequency grids whose dimension product is
    at most ``n ** alpha`` and returns the largest normalized mutual
    information. The result is symmetric and lies between zero and one.
    ``max_bins`` can cap the search budget for large datasets.
    """
    _validate_parameters(alpha, max_bins)
    xa, ya = _prepare_pair(x, y, missing)
    if len(xa) < 2:
        return float("nan")
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return 0.0
    budget = _grid_budget(len(xa), alpha, max_bins)
    # Search in both orientations so the finite approximation is symmetric.
    xy = characteristic_matrix(xa, ya, budget)
    yx = characteristic_matrix(ya, xa, budget)
    return float(max(np.nanmax(xy), np.nanmax(yx)))


def mic_matrix(
    data: Any,
    *,
    alpha: float = 0.6,
    max_bins: int | None = None,
    missing: Literal["pairwise", "listwise", "raise"] = "pairwise",
):
    """Compute MIC for every pair of columns in a two-dimensional dataset."""
    _validate_parameters(alpha, max_bins)
    if missing not in {"pairwise", "listwise", "raise"}:
        raise ValueError("missing must be 'pairwise', 'listwise', or 'raise'")
    is_dataframe = hasattr(data, "columns") and hasattr(data, "to_numpy")
    names = list(data.columns) if is_dataframe else None
    values = np.asarray(data.to_numpy() if is_dataframe else data, dtype=object)
    if values.ndim != 2:
        raise ValueError("data must be two-dimensional")
    missing_mask = np.vectorize(_is_missing, otypes=[bool])(values)
    if missing == "raise" and missing_mask.any():
        raise ValueError("data contains missing values")
    if missing == "listwise":
        values = values[~missing_mask.any(axis=1)]

    columns = values.shape[1]
    result = np.empty((columns, columns), dtype=float)
    pair_missing = "raise" if missing == "raise" else "drop"
    for i in range(columns):
        for j in range(i, columns):
            score = maximal_information_coefficient(
                values[:, i], values[:, j], alpha=alpha,
                max_bins=max_bins, missing=pair_missing,
            )
            result[i, j] = result[j, i] = score
    if is_dataframe:
        import pandas as pd

        return pd.DataFrame(result, index=names, columns=names)
    return result


maximal_information_coefficient_matrix = mic_matrix
