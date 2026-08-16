"""Pearson product-moment correlation calculations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from models.mutual_information.missing_values import _is_missing

__all__ = ["pearson_correlation", "pearson_correlation_matrix"]


def pearson_correlation(
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    missing: Literal["drop", "raise"] = "drop",
) -> float:
    """Calculate Pearson's signed linear correlation for two numeric variables."""
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
        raise ValueError("Pearson inputs must contain numeric values") from error
    if len(xa) < 2 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return float("nan")
    centered_x = xa - xa.mean()
    centered_y = ya - ya.mean()
    denominator = np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    return float(np.clip(np.dot(centered_x, centered_y) / denominator, -1.0, 1.0))


def pearson_correlation_matrix(
    data: Any,
    *,
    missing: Literal["pairwise", "listwise", "raise"] = "pairwise",
):
    """Calculate a Pearson correlation matrix for numeric columns."""
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
            score = pearson_correlation(
                values[:, i], values[:, j], missing=pair_missing
            )
            result[i, j] = result[j, i] = score
    if is_dataframe:
        import pandas as pd

        return pd.DataFrame(result, index=names, columns=names)
    return result
