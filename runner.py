"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual Information analysis
"""

from pathlib import Path

import pandas as pd

from mutual_information import mutual_information_matrix


# ---------------------------------------------------------------------------
# Analysis settings
# Edit these values before running: python runner.py
# ---------------------------------------------------------------------------

INPUT_CSV = Path("sample_dataset.csv")

# "auto" infers each column's type. You may instead use:
#   True                         -> treat every column as discrete
#   False                        -> treat every column as continuous
#   ["col5", "col6", "col8"]     -> named discrete columns
#   [False, False, ..., True]    -> one boolean per column
DISCRETE = "auto"

# A NumPy histogram rule ("auto", "fd", "sturges", etc.), an integer number
# of bins, or a sequence of bin edges. Used only for continuous variables.
BINS = "auto"

# False returns MI in bits. True is equivalent to "sqrt" normalization.
# Other normalized options are "min" and "max".
NORMALIZE = True

# "pairwise", "listwise", or "raise"
MISSING = "pairwise"

# 2.0 reports non-normalized mutual information in bits.
BASE = 2.0

# Number of decimal places displayed in the terminal.
ROUND_DIGITS = 3

# Set this to a path such as Path("mi_results.csv") to save the matrix.
# Leave it as None to only print the result.
OUTPUT_CSV = None


def main() -> None:
    """Load the configured CSV, calculate MI, and print/save the result."""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {INPUT_CSV.resolve()}")

    data = pd.read_csv(INPUT_CSV)
    result = mutual_information_matrix(
        data,
        discrete=DISCRETE,
        bins=BINS,
        normalize=NORMALIZE,
        missing=MISSING,
        base=BASE,
    )

    print(f"Dataset: {INPUT_CSV} ({len(data)} rows, {len(data.columns)} columns)")
    print(result.round(ROUND_DIGITS).to_string())

    if OUTPUT_CSV is not None:
        output_path = Path(OUTPUT_CSV)
        result.to_csv(output_path)
        print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
