"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual Information analysis
"""

from pathlib import Path

import pandas as pd
import yaml

from mutual_information import mutual_information_matrix


CONFIG_PATH = Path(__file__).with_name("config.yaml")
EXPECTED_SETTINGS = {
    "input_csv",
    "discrete",
    "bins",
    "normalize",
    "missing",
    "base",
    "round_digits",
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
    print(result.round(config["round_digits"]).to_string())

    if config["output_csv"] is not None:
        output_path = config_directory / Path(config["output_csv"])
        result.to_csv(output_path)
        print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
