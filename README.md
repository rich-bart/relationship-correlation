# Mutual Information and MIC

Compute pairwise mutual information between every column in a CSV dataset.
The project supports discrete and continuous variables, normalized results,
and configurable missing-data handling.

The Python API also provides a dependency-free maximal information coefficient
(MIC) estimator for numeric variables. It searches equal-frequency grids under
the standard `B(n) = n ** alpha` grid budget and returns the largest normalized
grid mutual information. This is a deterministic approximation of the
exhaustive MIC grid optimization.

Users configure an analysis in `config.yaml`; no Python files need to be
edited.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- PyYAML

## Installation

Open PowerShell in the project directory:

```powershell
cd C:\Users\rbart\Documents\Coding_Projects\relationship-correlation
```

Creating an isolated environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

To also install the development and testing dependencies:

```powershell
python -m pip install -e ".[dev]"
```

## Configuration

Edit `config.yaml` before running the analysis:

```yaml
input_csv: datasets/sample_dataset.csv
analysis: mic
target_column: null
exclude_columns: []
discrete: auto
bins: auto
normalize: true
missing: pairwise
base: 2.0
round_digits: 3
color_output: true
spectrogram: false
output_csv: null
alpha: 0.6
max_bins: null
permutations: 100
random_seed: 42
```

### Settings

| Setting | Description |
| --- | --- |
| `analysis` | Selects `mic` or `mutual_information`. |
| `input_csv` | Path to the input CSV file. Relative paths start from the directory containing `config.yaml`. |
| `target_column` | Optional outcome column. When set, features are ranked against it and saved to `target_features.csv`. Use `null` to disable. |
| `exclude_columns` | Column names to remove before calculating the relationship matrix. Names absent from the dataset are skipped. Use `[]` to keep every column. |
| `discrete` | Controls which variables are treated as discrete. See the options below. |
| `bins` | Histogram rule, bin count, or bin-edge list used for continuous variables. |
| `normalize` | Controls whether and how the mutual information is normalized. |
| `missing` | Selects the missing-data policy: `pairwise`, `listwise`, or `raise`. |
| `base` | Logarithm base. Use `2.0` to report non-normalized mutual information in bits. |
| `round_digits` | Number of decimal places printed in the terminal. |
| `color_output` | Set to `true` for a colored terminal matrix or `false` for plain text. |
| `spectrogram` | Set to `true` to open a graphical color map of the result matrix. |
| `output_csv` | Output filename for the matrix, or `null` to only print it. |
| `alpha` | MIC grid-budget exponent; the default is `0.6`. |
| `max_bins` | Optional MIC grid-budget cap, or `null` for no cap. |
| `permutations` | Number of label shuffles used for empirical p-values. Use `0` to disable; larger values increase runtime and p-value resolution. |
| `random_seed` | Integer seed that makes permutation results reproducible. |

The `discrete` setting accepts:

```yaml
# Infer each column's type
discrete: auto

# Treat every column as discrete
discrete: true

# Treat every column as continuous
discrete: false

# Treat only these named columns as discrete
discrete: [col5, col6, col8]

# Provide one setting for every column
discrete: [false, false, false, false, true, true, false, true]
```

The `normalize` setting accepts:

- `false`: return unnormalized mutual information
- `true` or `sqrt`: divide by the geometric mean of the two entropies
- `min`: divide by the smaller entropy
- `max`: divide by the larger entropy

The `missing` setting accepts:

- `pairwise`: remove missing observations separately for each column pair
- `listwise`: remove any row containing a missing value
- `raise`: stop with an error if any value is missing

To save the result as another CSV file:

```yaml
output_csv: mi_results.csv
```

## Running an analysis

After editing `config.yaml`, run:

```powershell
python runner.py
```

With `analysis: mic`, `discrete: auto` detects text and categorical columns and
encodes their values as discrete category labels. The named-column and boolean
mask forms of `discrete` work for MIC as well. Settings such as `bins`,
`normalize`, and `base` apply only when `analysis` is `mutual_information`.

The included sample configuration analyzes `datasets/sample_dataset.csv`,
which has 1,000 rows and eight columns named `col1` through `col8`.

## Plotting the track dataset

The included track dataset can be visualized with:

```powershell
python plotter.py
```

This opens a scatter plot of `TIME` versus `RANGE`, with a different color and
legend entry for each `TRACK_ID`.

## Output

The runner prints a symmetric matrix with one row and column for every
variable. It also prints a ranked list containing every unique feature pair
and its coefficient, from strongest to weakest. Diagonal self-correlations and
duplicate reversed pairs are omitted. The same table is automatically saved as
`feature_pairs.csv` beside `config.yaml`:

When `target_column` is set, the runner additionally prints a target-focused
ranking and saves it as `target_features.csv`. The target itself is omitted
from this ranking; the complete pairwise outputs are still produced.

When `permutations` is greater than zero, both ranked CSV files include an
empirical `P-value`. For each pair, one variable is shuffled repeatedly and the
p-value is `(null scores >= observed score + 1) / (permutations + 1)`. These are
raw p-values; multiple-comparison correction is not yet applied.

```text
       col1   col2   col3   col4   col5   col6   col7   col8
col1  1.000  0.017  0.021  0.021  0.000  0.004  0.029  0.006
col2  0.017  1.000  0.589  0.022  0.007  0.490  0.032  0.006
col3  0.021  0.589  1.000  0.020  0.008  0.465  0.027  0.008
col4  0.021  0.022  0.020  1.000  0.008  0.005  0.439  0.005
col5  0.000  0.007  0.008  0.008  1.000  0.001  0.014  0.502
col6  0.004  0.490  0.465  0.005  0.001  1.000  0.009  0.000
col7  0.029  0.032  0.027  0.439  0.014  0.009  1.000  0.013
col8  0.006  0.006  0.008  0.005  0.502  0.000  0.013  1.000
```

For normalized output:

- A value near `0` indicates little shared information.
- A value near `1` indicates a strong relationship.
- The diagonal is `1` for a nonconstant variable because each variable fully
  describes itself.
- Mutual information can detect nonlinear relationships, unlike ordinary
  linear correlation.

Histogram-based continuous estimates depend on the selected `bins` setting
and the amount of available data.

## Python API

The YAML runner is the standard user interface, but the functions can also be
called directly from Python.

Analyze every column:

```python
import pandas as pd

from models.mutual_information import mutual_information_matrix

data = pd.read_csv("datasets/sample_dataset.csv")
matrix = mutual_information_matrix(data, normalize=True)
print(matrix)
```

Analyze two individual variables:

```python
from models.mutual_information import mutual_information

score = mutual_information(
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    discrete_x=True,
    discrete_y=True,
)
print(score)
```

Calculate MIC for two numeric variables or every DataFrame column:

```python
from models.maximal_information_coefficient import (
    maximal_information_coefficient,
    maximal_information_coefficient_matrix,
)

score = maximal_information_coefficient(x, y)
matrix = maximal_information_coefficient_matrix(data)
```

MIC values range from 0 to 1. The optional `alpha` argument controls the grid
budget (default `0.6`), while `max_bins` can cap that budget for large inputs.
Missing values may be dropped or rejected for a pair; matrix calculations also
support pairwise and listwise deletion.

For categorical MIC inputs, category values are converted to integer labels
before grid searching; the original strings do not need to be numeric. If a
graphical spectrogram window is unavailable, the runner saves
`spectrogram.png` beside `config.yaml` instead.

## Project layout

```text
relationship-correlation/
├── config.yaml
├── datasets/
│   ├── sample_dataset.csv
│   └── sample_track_data.csv
├── models/
│   ├── __init__.py
│   ├── maximal_information_coefficient/
│   │   ├── __init__.py
│   │   ├── characteristic_matrix.py
│   │   ├── grid_search.py
│   │   ├── maximal_information_coefficient.py
│   │   ├── normalization.py
│   │   └── validation.py
│   └── mutual_information/
│       ├── __init__.py
│       ├── mutual_information.py
│       ├── discretization.py
│       ├── encoding.py
│       ├── information_theory.py
│       ├── missing_values.py
│       └── type_inference.py
├── pyproject.toml
├── README.md
├── plotter.py
└── runner.py
```

- `config.yaml`: user-editable analysis settings
- `runner.py`: loads the configuration and runs the analysis
- `plotter.py`: plots time versus range for the sample track data
- `models/mutual_information/`: mutual-information API and private helpers
- `models/maximal_information_coefficient/`: MIC estimator, characteristic
  matrix, grid search, normalization, and validation helpers
- `datasets/sample_dataset.csv`: example input dataset
- `datasets/sample_track_data.csv`: example three-track position dataset
