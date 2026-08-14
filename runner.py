"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual information and MIC analysis
"""

from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt
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


def show_spectrogram(matrix: pd.DataFrame, title: str) -> None:
    """Display the result matrix as a color spectrogram."""
    size = max(7.0, min(16.0, 0.65 * len(matrix.columns) + 3.0))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    positions = range(len(matrix.columns))
    axis.set_xticks(positions, [str(column) for column in matrix.columns])
    axis.set_yticks(positions, [str(index) for index in matrix.index])
    axis.tick_params(axis="x", labelrotation=45)
    axis.set_xlabel("Variable")
    axis.set_ylabel("Variable")
    axis.set_title(f"{title} spectrogram")
    figure.colorbar(image, ax=axis, label="Coefficient")
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Load the configured CSV, calculate the selected model, and output it."""
    config = load_config()
    config_directory = CONFIG_PATH.parent
    input_csv = config_directory / Path(config["input_csv"])
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {input_csv.resolve()}")

    data = pd.read_csv(input_csv)
    if config["analysis"] == "mic":
        result = maximal_information_coefficient_matrix(
            data,
            alpha=config["alpha"],
            max_bins=config["max_bins"],
            missing=config["missing"],
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

    if config["output_csv"] is not None:
        output_path = config_directory / Path(config["output_csv"])
        result.to_csv(output_path)
        print(f"\nSaved results to: {output_path.resolve()}")

    if config["spectrogram"]:
        show_spectrogram(result, analysis_name)


if __name__ == "__main__":
    main()
