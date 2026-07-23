"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026

Purpose: This module provides functions to compute mutual information between all variables in a dataset.
    Which is a measure of the amount of information obtained about one variable through the other variable. 
    It includes methods for both discrete and continuous variables as well as options for normalization and handling missing data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np

from utils.discretization import _discretize
from utils.information_theory import _discrete_mi
from utils.missing_values import _is_missing
from utils.type_inference import _infer_discrete

__all__ = ["mutual_information", "mutual_information_matrix"]

Normalization = Literal["sqrt", "min", "max"]
MissingPolicy = Literal["pairwise", "listwise", "raise"]


def mutual_information(
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    discrete_x: bool = True,
    discrete_y: bool = True,
    bins: str | int | Sequence[float] = "auto",
    normalize: bool | Normalization = False,
    missing: Literal["drop", "raise"] = "drop",
    base: float = 2.0,
) -> float:
    """Compute the mutual information between two variables.

    Continuous variables are histogram-discretized before calculation.  Set
    ``discrete_x`` or ``discrete_y`` to ``False`` for continuous inputs.
    ``normalize=True`` uses the geometric-mean (``"sqrt"``) normalization.
    Other supported normalizations are ``"min"`` and ``"max"``.
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from 1")
    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'")
    if normalize is True:
        normalize = "sqrt"
    if normalize not in {False, "sqrt", "min", "max"}:
        raise ValueError("normalize must be False, True, 'sqrt', 'min', or 'max'")

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
    xa, ya = xa[valid], ya[valid]
    if len(xa) == 0:
        return float("nan")
    if not discrete_x:
        xa = _discretize(xa, bins)
    if not discrete_y:
        ya = _discretize(ya, bins)

    mi, hx, hy = _discrete_mi(xa, ya, base)
    if not normalize:
        return mi
    denominator = {
        "sqrt": np.sqrt(hx * hy),
        "min": min(hx, hy),
        "max": max(hx, hy),
    }[normalize]
    return 0.0 if denominator == 0 else min(1.0, mi / denominator)


def mutual_information_matrix(
    data: Any,
    *,
    discrete: Literal["auto"] | bool | Iterable[Any] | Sequence[bool] = "auto",
    bins: str | int | Sequence[float] = "auto",
    normalize: bool | Normalization = False,
    missing: MissingPolicy = "pairwise",
    base: float = 2.0,
):
    """Compute pairwise mutual information for every column in a dataset.

    Parameters
    ----------
    data:
        A two-dimensional array-like object or pandas ``DataFrame``. Variables
        are columns and observations are rows.
    discrete:
        ``"auto"`` to infer types, one boolean for all columns, a boolean mask,
        or (for a DataFrame) an iterable of discrete column names.
    bins:
        Histogram bin rule, count, or edges accepted by
        :func:`numpy.histogram_bin_edges`.
    normalize:
        ``False`` for MI in units determined by ``base``; ``True``/``"sqrt"``,
        ``"min"``, or ``"max"`` for normalized MI.
    missing:
        ``"pairwise"`` drops missing rows separately for each pair,
        ``"listwise"`` drops any row containing a missing value, and
        ``"raise"`` rejects missing data.
    base:
        Logarithm base. The default, 2, reports mutual information in bits.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        A symmetric square matrix. A DataFrame input produces a labeled
        DataFrame output.
    """
    if missing not in {"pairwise", "listwise", "raise"}:
        raise ValueError("missing must be 'pairwise', 'listwise', or 'raise'")

    is_dataframe = hasattr(data, "columns") and hasattr(data, "to_numpy")
    names = list(data.columns) if is_dataframe else None
    values = np.asarray(data.to_numpy() if is_dataframe else data, dtype=object)
    if values.ndim != 2:
        raise ValueError("data must be two-dimensional")
    _, n_columns = values.shape

    missing_mask = np.vectorize(_is_missing, otypes=[bool])(values)
    if missing == "raise" and missing_mask.any():
        raise ValueError("data contains missing values")
    if missing == "listwise":
        values = values[~missing_mask.any(axis=1)]

    if isinstance(discrete, str) and discrete == "auto":
        kinds = [_infer_discrete(values[:, i]) for i in range(n_columns)]
    elif isinstance(discrete, (bool, np.bool_)):
        kinds = [bool(discrete)] * n_columns
    else:
        supplied = list(discrete)
        if is_dataframe and not all(isinstance(item, (bool, np.bool_)) for item in supplied):
            selected = set(supplied)
            unknown = selected.difference(names)
            if unknown:
                raise ValueError(f"unknown columns in discrete: {sorted(unknown)!r}")
            kinds = [name in selected for name in names]
        else:
            if len(supplied) != n_columns:
                raise ValueError("discrete mask must have one item per column")
            kinds = [bool(item) for item in supplied]

    result = np.empty((n_columns, n_columns), dtype=float)
    for i in range(n_columns):
        for j in range(i, n_columns):
            value = mutual_information(
                values[:, i],
                values[:, j],
                discrete_x=kinds[i],
                discrete_y=kinds[j],
                bins=bins,
                normalize=normalize,
                missing="drop",
                base=base,
            )
            result[i, j] = result[j, i] = value

    if is_dataframe:
        import pandas as pd

        return pd.DataFrame(result, index=names, columns=names)
    return result

