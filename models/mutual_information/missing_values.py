"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Missing-value helpers for mutual information.
"""

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
