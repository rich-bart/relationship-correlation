"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Plot time versus range for the sample track dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
0

DATA_FILE = Path(__file__).parent / "datasets" / "sample_track_data.csv"
X_LABEL = "TIME"
Y_LABEL = "RANGE"
GROUP_BY = "TRACK_ID"


def main() -> None:
    data = pd.read_csv(DATA_FILE)

    for track_id, track in data.groupby(GROUP_BY):
        plt.scatter(
            track[X_LABEL],
            track[Y_LABEL],
            s=14,
            alpha=0.75,
            label=f"Track {track_id}",
        )

    plt.xlabel(f"{X_LABEL}")
    plt.ylabel(f"{Y_LABEL}")
    plt.title(f"{X_LABEL} vs. {Y_LABEL} by {GROUP_BY}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
