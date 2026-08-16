"""Spearman rank-order correlation calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from models.mutual_information.missing_values import _is_missing
from models.pearson import pearson_correlation

__all__ = ["spearman_correlation", "spearman_correlation_matrix"]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, assigning tied values equal ranks."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman_correlation(
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    missing: Literal["drop", "raise"] = "drop",
) -> float:
    """Calculate Spearman's signed correlation between two numeric variables."""
    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'")
    xa = np.asarray(x, dtype=object).reshape(-1)
    ya = np.asarray(y, dtype=object).reshape(-1)
    if len(xa) != len(ya):
        raise ValueError("x and y must have the same length")
    valid = np.fromiter(
        (not (_is_missing(a) or _is_missing(b)) for a, b in zip(xa, ya)),
        dtype=bool,
        count=len(xa),
    )
    if missing == "raise" and not valid.all():
        raise ValueError("x or y contains missing values")
    try:
        xa = xa[valid].astype(float)
        ya = ya[valid].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Spearman inputs must contain numeric values") from error
    if len(xa) < 2:
        return float("nan")
    return pearson_correlation(
        _average_ranks(xa), _average_ranks(ya), missing="raise"
    )


def spearman_correlation_matrix(
    data: Any,
    *,
    missing: Literal["pairwise", "listwise", "raise"] = "pairwise",
):
    """Calculate a Spearman correlation matrix for numeric columns."""
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
            score = spearman_correlation(
                values[:, i], values[:, j], missing=pair_missing
            )
            result[i, j] = result[j, i] = score
    if is_dataframe:
        import pandas as pd

        return pd.DataFrame(result, index=names, columns=names)
    return result
