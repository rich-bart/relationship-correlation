"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Runner for mutual information and MIC analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from models.maximal_information_coefficient import (
    maximal_information_coefficient,
    maximal_information_coefficient_matrix,
)
from models.mutual_information import mutual_information_matrix
from models.mutual_information.discretization import _discretize
from models.mutual_information.encoding import _factorize
from models.mutual_information.missing_values import _is_missing
from models.mutual_information.type_inference import _infer_discrete


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
    "target_column",
    "permutations",
    "random_seed",
    "multiple_comparison",
    "significance_level",
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
    if config["target_column"] is not None and not isinstance(
        config["target_column"], str
    ):
        raise ValueError("target_column must be a column name or null")
    if not isinstance(config["permutations"], int) or config["permutations"] < 0:
        raise ValueError("permutations must be a nonnegative integer")
    if not isinstance(config["random_seed"], int):
        raise ValueError("random_seed must be an integer")
    if config["multiple_comparison"] not in {"benjamini_hochberg", "none"}:
        raise ValueError(
            "multiple_comparison must be 'benjamini_hochberg' or 'none'"
        )
    if not isinstance(config["significance_level"], (int, float)) or not (
        0 < config["significance_level"] < 1
    ):
        raise ValueError("significance_level must be greater than 0 and less than 1")
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


def feature_pair_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return every unique feature pair ranked by its coefficient."""
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
    return output


def print_feature_pairs(
    output: pd.DataFrame, digits: int, color_output: bool
) -> None:
    """Print a ranked table of unique feature pairs."""

    if not color_output:
        print("\nRanked feature pairs:")
        print(
            output.round(
                {
                    "Coefficient": digits,
                    "P-value": digits,
                    "Adjusted P-value": digits,
                }
            )
            .to_string(index=False)
        )
        return

    table = Table(
        title="Ranked feature pairs",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Feature 1", style="cyan")
    table.add_column("Feature 2", style="cyan")
    table.add_column("Coefficient", justify="right")
    if "P-value" in output:
        table.add_column("P-value", justify="right")
    if "Adjusted P-value" in output:
        table.add_column("Adjusted P-value", justify="right")
        table.add_column("Significant", justify="center")
    for row in output.itertuples(index=False, name=None):
        feature_1, feature_2, coefficient, *statistics = row
        cells = [feature_1, feature_2, f"{coefficient:.{digits}f}"]
        cells.extend(
            f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in statistics
        )
        table.add_row(*cells)
    Console().print(table)


def target_feature_table(
    matrix: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Return all non-target features ranked by association with the target."""
    rows = [
        (str(feature), target_column, float(matrix.loc[feature, target_column]))
        for feature in matrix.index
        if feature != target_column and pd.notna(matrix.loc[feature, target_column])
    ]
    rows.sort(key=lambda row: row[2], reverse=True)
    return pd.DataFrame(rows, columns=["Feature", "Target", "Coefficient"])


def print_target_features(
    table_data: pd.DataFrame, digits: int, color_output: bool
) -> None:
    """Print a target-focused feature ranking."""
    if not color_output:
        print("\nFeatures ranked against target:")
        print(
            table_data.round(
                {
                    "Coefficient": digits,
                    "P-value": digits,
                    "Adjusted P-value": digits,
                }
            ).to_string(index=False)
        )
        return

    table = Table(
        title="Features ranked against target",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Feature", style="cyan")
    table.add_column("Target", style="magenta")
    table.add_column("Coefficient", justify="right")
    if "P-value" in table_data:
        table.add_column("P-value", justify="right")
    if "Adjusted P-value" in table_data:
        table.add_column("Adjusted P-value", justify="right")
        table.add_column("Significant", justify="center")
    for row in table_data.itertuples(index=False, name=None):
        feature, target, coefficient, *statistics = row
        cells = [feature, target, f"{coefficient:.{digits}f}"]
        cells.extend(
            f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in statistics
        )
        table.add_row(*cells)
    Console().print(table)


def _discrete_column_map(data: pd.DataFrame, discrete) -> dict[str, bool]:
    """Resolve the YAML discrete setting to one boolean per column."""
    names = list(data.columns)
    if discrete == "auto":
        return {name: _infer_discrete(data[name].to_numpy()) for name in names}
    if isinstance(discrete, bool):
        return dict.fromkeys(names, discrete)
    supplied = list(discrete)
    if all(isinstance(item, bool) for item in supplied):
        if len(supplied) != len(names):
            raise ValueError("discrete mask must have one item per column")
        return dict(zip(names, supplied))
    selected = set(supplied)
    return {name: name in selected for name in names}


def add_permutation_p_values(
    table_data: pd.DataFrame,
    data: pd.DataFrame,
    config: dict,
    feature_columns: tuple[str, str],
) -> pd.DataFrame:
    """Add empirical permutation p-values to a pair-ranking table."""
    count = config["permutations"]
    if count == 0:
        return table_data
    rng = np.random.default_rng(config["random_seed"])
    kinds = _discrete_column_map(data, config["discrete"])
    p_values = []
    for _, row in table_data.iterrows():
        left = row[feature_columns[0]]
        right = row[feature_columns[1]]
        observed = row["Coefficient"]
        x = data[left].to_numpy(dtype=object)
        y = data[right].to_numpy(dtype=object)
        valid = np.fromiter(
            (not (_is_missing(a) or _is_missing(b)) for a, b in zip(x, y)),
            dtype=bool,
            count=len(x),
        )
        x, y = x[valid], y[valid]
        if config["analysis"] == "mutual_information":
            x_labels = (
                _factorize(x)
                if kinds[left]
                else _discretize(x, config["bins"])
            )
            y_labels = (
                _factorize(y)
                if kinds[right]
                else _discretize(y, config["bins"])
            )
        exceedances = 0
        for _ in range(count):
            if config["analysis"] == "mic":
                shuffled = rng.permutation(y)
                score = maximal_information_coefficient(
                    x, shuffled,
                    alpha=config["alpha"], max_bins=config["max_bins"],
                    missing="raise", discrete_x=kinds[left],
                    discrete_y=kinds[right],
                )
            else:
                score = _encoded_mutual_information(
                    x_labels,
                    rng.permutation(y_labels),
                    config["normalize"],
                    config["base"],
                )
            exceedances += score >= observed
        p_values.append((exceedances + 1) / (count + 1))
    output = table_data.copy()
    output["P-value"] = p_values
    return output


def add_multiple_comparison_correction(
    table_data: pd.DataFrame,
    method: str,
    significance_level: float,
) -> pd.DataFrame:
    """Add adjusted p-values and decisions for one family of comparisons."""
    if method == "none" or "P-value" not in table_data:
        return table_data
    p_values = table_data["P-value"].to_numpy(dtype=float)
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    output = table_data.copy()
    output["Adjusted P-value"] = adjusted
    output["Significant"] = adjusted <= significance_level
    return output


def _encoded_mutual_information(
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    normalize,
    base: float,
) -> float:
    """Calculate MI quickly from pre-encoded labels during permutation tests."""
    nx = int(x_labels.max(initial=-1)) + 1
    ny = int(y_labels.max(initial=-1)) + 1
    joint_ids, joint_counts = np.unique(
        x_labels * ny + y_labels, return_counts=True
    )
    joint = joint_counts / len(x_labels)
    px = np.bincount(x_labels, minlength=nx) / len(x_labels)
    py = np.bincount(y_labels, minlength=ny) / len(y_labels)
    joint_x = joint_ids // ny
    joint_y = joint_ids % ny
    log_base = np.log(base)
    mi = float(
        np.sum(joint * np.log(joint / (px[joint_x] * py[joint_y])))
        / log_base
    )
    if not normalize:
        return max(0.0, mi)
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])) / log_base)
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])) / log_base)
    method = "sqrt" if normalize is True else normalize
    denominator = {
        "sqrt": np.sqrt(hx * hy),
        "min": min(hx, hy),
        "max": max(hx, hy),
    }[method]
    return 0.0 if denominator == 0 else min(1.0, max(0.0, mi) / denominator)


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
    target_column = config["target_column"]
    if target_column is not None and target_column not in data.columns:
        if target_column in excluded:
            raise ValueError(
                f"target_column {target_column!r} cannot also be excluded"
            )
        raise ValueError(f"target_column was not found in dataset: {target_column!r}")
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
    if target_column is not None:
        print(f"Target column: {target_column}")
    permutation_data = data
    if config["missing"] == "listwise":
        permutation_data = data.dropna(axis=0, how="any")

    print_matrix(result, config["round_digits"], config["color_output"])
    if target_column is not None:
        target_features = target_feature_table(result, target_column)
        target_features = add_permutation_p_values(
            target_features,
            permutation_data,
            config,
            ("Feature", "Target"),
        )
        target_features = add_multiple_comparison_correction(
            target_features,
            config["multiple_comparison"],
            config["significance_level"],
        )
        print_target_features(
            target_features,
            config["round_digits"],
            config["color_output"],
        )
        target_features_path = config_directory / "target_features.csv"
        target_features.to_csv(target_features_path, index=False)
    feature_pairs = feature_pair_table(result)
    feature_pairs = add_permutation_p_values(
        feature_pairs,
        permutation_data,
        config,
        ("Feature 1", "Feature 2"),
    )
    feature_pairs = add_multiple_comparison_correction(
        feature_pairs,
        config["multiple_comparison"],
        config["significance_level"],
    )
    print_feature_pairs(
        feature_pairs, config["round_digits"], config["color_output"]
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
    if target_column is not None:
        print(f"Saved target features to: {target_features_path.resolve()}")

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
