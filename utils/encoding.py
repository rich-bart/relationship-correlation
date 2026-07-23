"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Categorical encoding helpers.
"""

from typing import Any

import numpy as np


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
