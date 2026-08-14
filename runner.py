"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual information and MIC analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from models.maximal_information_coefficient import (
    maximal_information_coefficient_matrix,
)
from models.mutual_information import mutual_information_matrix


CONFIG_PATH = Path(__file__).with_name("config.yaml")
EXPECTED_SETTINGS = {
    "analysis",
    "input_csv",
    "discrete",
    "bins",
    "normalize",
    "missing",
    "base",
    "round_digits",
    "color_output",
    "output_csv",
    "alpha",
    "max_bins",
    "spectrogram",
    "exclude_columns",
}


def load_config() -> dict:
    """Load and validate the user-editable YAML configuration."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping of settings")

    missing_settings = EXPECTED_SETTINGS.difference(config)
    unknown_settings = set(config).difference(EXPECTED_SETTINGS)
    if missing_settings:
        raise ValueError(
            f"config.yaml is missing settings: {sorted(missing_settings)}"
        )
    if unknown_settings:
        raise ValueError(
            f"config.yaml contains unknown settings: {sorted(unknown_settings)}"
        )
    if config["analysis"] not in {"mic", "mutual_information"}:
        raise ValueError("analysis must be 'mic' or 'mutual_information'")
    if not isinstance(config["spectrogram"], bool):
        raise ValueError("spectrogram must be true or false")
    if not isinstance(config["exclude_columns"], list) or not all(
        isinstance(column, str) for column in config["exclude_columns"]
    ):
        raise ValueError("exclude_columns must be a list of column names")
    return config


def print_matrix(matrix: pd.DataFrame, digits: int, color_output: bool) -> None:
    """Print a plain or heatmap-colored mutual-information matrix."""
    if not color_output:
        print(matrix.round(digits).to_string())
        return

    values = matrix.to_numpy(dtype=float)
    finite_values = values[pd.notna(values)]
    scale = max((abs(value) for value in finite_values), default=1.0) or 1.0

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("", style="bold cyan")
    for column in matrix.columns:
        table.add_column(str(column), justify="right")

    for row_name, row in matrix.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("[magenta]nan[/magenta]")
                continue
            intensity = abs(float(value)) / scale
            if intensity >= 0.75:
                color = "bright_green"
            elif intensity >= 0.40:
                color = "yellow"
            elif intensity >= 0.15:
                color = "cyan"
            else:
                color = "bright_black"
            cells.append(f"[{color}]{value:.{digits}f}[/{color}]")
        table.add_row(str(row_name), *cells)

    Console().print(table)


def print_feature_pairs(
    matrix: pd.DataFrame, digits: int, color_output: bool
) -> pd.DataFrame:
    """Print each unique feature pair ranked by its coefficient."""
    pairs = [
        (str(matrix.index[i]), str(matrix.columns[j]), float(matrix.iloc[i, j]))
        for i in range(len(matrix.index))
        for j in range(i + 1, len(matrix.columns))
        if pd.notna(matrix.iloc[i, j])
    ]
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    output = pd.DataFrame(
        pairs, columns=["Feature 1", "Feature 2", "Coefficient"]
    )

    if not color_output:
        print("\nRanked feature pairs:")
        print(output.round({"Coefficient": digits}).to_string(index=False))
        return output

    table = Table(
        title="Ranked feature pairs",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Feature 1", style="cyan")
    table.add_column("Feature 2", style="cyan")
    table.add_column("Coefficient", justify="right")
    for feature_1, feature_2, coefficient in pairs:
        table.add_row(feature_1, feature_2, f"{coefficient:.{digits}f}")
    Console().print(table)
    return output


def show_spectrogram(
    matrix: pd.DataFrame, title: str, fallback_path: Path
) -> None:
    """Display the result matrix, saving it if no GUI backend is available."""
    size = max(7.0, min(16.0, 0.65 * len(matrix.columns) + 3.0))
    try:
        figure, axis = plt.subplots(figsize=(size, size))
    except Exception as error:
        plt.switch_backend("Agg")
        figure, axis = plt.subplots(figsize=(size, size))
        display_error = error
    else:
        display_error = None
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    positions = range(len(matrix.columns))
    axis.set_xticks(positions, [str(column) for column in matrix.columns])
    axis.set_yticks(positions, [str(index) for index in matrix.index])
    axis.tick_params(axis="x", labelrotation=90)
    axis.set_xlabel("Variable")
    axis.set_ylabel("Variable")
    axis.set_title(f"{title} spectrogram")
    figure.colorbar(image, ax=axis, label="Coefficient")
    figure.tight_layout()
    if display_error is None:
        plt.show()
    else:
        figure.savefig(fallback_path, dpi=150)
        plt.close(figure)
        print(
            "\nA graphical window was unavailable; saved the spectrogram to: "
            f"{fallback_path.resolve()}"
        )


def main() -> None:
    """Load the configured CSV, calculate the selected model, and output it."""
    config = load_config()
    config_directory = CONFIG_PATH.parent
    input_csv = config_directory / Path(config["input_csv"])
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {input_csv.resolve()}")

    data = pd.read_csv(input_csv)
    configured_exclusions = config["exclude_columns"]
    excluded = [
        column for column in configured_exclusions if column in data.columns
    ]
    skipped_exclusions = [
        column for column in configured_exclusions if column not in data.columns
    ]
    if excluded:
        data = data.drop(columns=excluded)
    if data.shape[1] < 2:
        raise ValueError("at least two columns must remain after exclusions")
    if config["analysis"] == "mic":
        result = maximal_information_coefficient_matrix(
            data,
            alpha=config["alpha"],
            max_bins=config["max_bins"],
            missing=config["missing"],
            discrete=config["discrete"],
        )
        analysis_name = "Maximal information coefficient"
    else:
        result = mutual_information_matrix(
            data,
            discrete=config["discrete"],
            bins=config["bins"],
            normalize=config["normalize"],
            missing=config["missing"],
            base=config["base"],
        )
        analysis_name = "Mutual information"

    print(f"Analysis: {analysis_name}")
    print(f"Dataset: {input_csv} ({len(data)} rows, {len(data.columns)} columns)")
    print_matrix(result, config["round_digits"], config["color_output"])
    feature_pairs = print_feature_pairs(
        result, config["round_digits"], config["color_output"]
    )
    feature_pairs_path = config_directory / "feature_pairs.csv"
    feature_pairs.to_csv(feature_pairs_path, index=False)
    if excluded:
        print(f"\nExcluded columns: {', '.join(excluded)}")
    if skipped_exclusions:
        print(
            "Skipped exclusions not found in dataset: "
            f"{', '.join(skipped_exclusions)}"
        )
    print(f"\nSaved feature pairs to: {feature_pairs_path.resolve()}")

    if config["output_csv"] is not None:
        output_path = config_directory / Path(config["output_csv"])
        result.to_csv(output_path)
        print(f"\nSaved results to: {output_path.resolve()}")

    if config["spectrogram"]:
        show_spectrogram(
            result,
            analysis_name,
            config_directory / "spectrogram.png",
        )

if __name__ == "__main__":
    main()
