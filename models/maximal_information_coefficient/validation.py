"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Input validation shared by the MIC public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from models.mutual_information.missing_values import _is_missing
from models.mutual_information.encoding import _factorize
from models.mutual_information.type_inference import _infer_discrete


def _validate_parameters(alpha: float, max_bins: int | None) -> None:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be greater than 0 and at most 1")
    if max_bins is not None and max_bins < 4:
        raise ValueError("max_bins must be at least 4")


def _prepare_pair(
    x: Sequence[Any],
    y: Sequence[Any],
    missing: Literal["drop", "raise"],
    discrete_x: Literal["auto"] | bool,
    discrete_y: Literal["auto"] | bool,
) -> tuple[np.ndarray, np.ndarray]:
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
    xa, ya = xa[valid], ya[valid]
    if discrete_x == "auto":
        discrete_x = _infer_discrete(xa)
    if discrete_y == "auto":
        discrete_y = _infer_discrete(ya)
    if not isinstance(discrete_x, (bool, np.bool_)) or not isinstance(
        discrete_y, (bool, np.bool_)
    ):
        raise ValueError("discrete_x and discrete_y must be true, false, or 'auto'")
    try:
        prepared_x = _factorize(xa).astype(float) if discrete_x else xa.astype(float)
        prepared_y = _factorize(ya).astype(float) if discrete_y else ya.astype(float)
        return prepared_x, prepared_y
    except (TypeError, ValueError) as error:
        raise ValueError(
            "continuous MIC inputs must contain numeric values; mark categorical "
            "inputs as discrete or use 'auto'"
        ) from error
