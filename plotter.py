"""Plot time versus range for the sample track dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path(__file__).with_name("sample_track_data.csv")


def main() -> None:
    data = pd.read_csv(DATA_FILE)

    for track_id, track in data.groupby("TRACK_ID"):
        plt.scatter(
            track["TIME"],
            track["RANGE"],
            s=14,
            alpha=0.75,
            label=f"Track {track_id}",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Range")
    plt.title("Time vs. Range by Track")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
