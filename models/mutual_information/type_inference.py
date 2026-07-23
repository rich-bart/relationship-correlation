"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Dataset variable-type inference helpers for mutual information.
"""

import numpy as np

from models.mutual_information.missing_values import _is_missing


def _infer_discrete(column: np.ndarray) -> bool:
    """Infer whether a dataset column should be treated as discrete."""
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
