"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Read in Pokemon dataset from kaggle
Kaggle dataset: https://www.kaggle.com/datasets/darkmatternet/ultimate-pokmon-dataset-2025/data
Kaggle Author:  DARKMATTERNET
"""

from pathlib import Path

import kagglehub

# Use a dedicated directory because KaggleHub may clear the output directory
# when downloading or refreshing a dataset.
datasets_dir = Path(__file__).resolve().parent / "pokemon"
datasets_dir.mkdir(parents=True, exist_ok=True)
path = kagglehub.dataset_download(
    "darkmatternet/ultimate-pokmon-dataset-2025",
    output_dir=str(datasets_dir),
)

print("Path to dataset files:", path)
