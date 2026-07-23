# Mutual Information

Compute pairwise mutual information between every column in a CSV dataset.
The project supports discrete and continuous variables, normalized results,
and configurable missing-data handling.

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
discrete: auto
bins: auto
normalize: true
missing: pairwise
base: 2.0
round_digits: 3
color_output: true
output_csv: null
```

### Settings

| Setting | Description |
| --- | --- |
| `input_csv` | Path to the input CSV file. Relative paths start from the directory containing `config.yaml`. |
| `discrete` | Controls which variables are treated as discrete. See the options below. |
| `bins` | Histogram rule, bin count, or bin-edge list used for continuous variables. |
| `normalize` | Controls whether and how the mutual information is normalized. |
| `missing` | Selects the missing-data policy: `pairwise`, `listwise`, or `raise`. |
| `base` | Logarithm base. Use `2.0` to report non-normalized mutual information in bits. |
| `round_digits` | Number of decimal places printed in the terminal. |
| `color_output` | Set to `true` for a colored terminal matrix or `false` for plain text. |
| `output_csv` | Output filename for the matrix, or `null` to only print it. |

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
variable:

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
│   │   └── __init__.py
│   └── mutual_information/
│       ├── __init__.py
│       ├── calculator.py
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
- `models/maximal_information_coefficient/`: reserved for the future MIC model
- `datasets/sample_dataset.csv`: example input dataset
- `datasets/sample_track_data.csv`: example three-track position dataset
