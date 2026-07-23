"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual Information analysis
"""

from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from models.mutual_information import mutual_information_matrix


CONFIG_PATH = Path(__file__).with_name("config.yaml")
EXPECTED_SETTINGS = {
    "input_csv",
    "discrete",
    "bins",
    "normalize",
    "missing",
    "base",
    "round_digits",
    "color_output",
    "output_csv",
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


def main() -> None:
    """Load the configured CSV, calculate MI, and print/save the result."""
    config = load_config()
    config_directory = CONFIG_PATH.parent
    input_csv = config_directory / Path(config["input_csv"])
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {input_csv.resolve()}")

    data = pd.read_csv(input_csv)
    result = mutual_information_matrix(
        data,
        discrete=config["discrete"],
        bins=config["bins"],
        normalize=config["normalize"],
        missing=config["missing"],
        base=config["base"],
    )

    print(f"Dataset: {input_csv} ({len(data)} rows, {len(data.columns)} columns)")
    print_matrix(result, config["round_digits"], config["color_output"])

    if config["output_csv"] is not None:
        output_path = config_directory / Path(config["output_csv"])
        result.to_csv(output_path)
        print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
