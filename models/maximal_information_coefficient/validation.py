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


def _validate_parameters(alpha: float, max_bins: int | None) -> None:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be greater than 0 and at most 1")
    if max_bins is not None and max_bins < 4:
        raise ValueError("max_bins must be at least 4")


def _prepare_pair(
    x: Sequence[Any],
    y: Sequence[Any],
    missing: Literal["drop", "raise"],
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
    try:
        return xa[valid].astype(float), ya[valid].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("MIC inputs must contain numeric values") from error
