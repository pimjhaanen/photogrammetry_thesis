"""This code can be used to visualize UWB distance fluctuations from a CSV log.
It linearly interpolates missing samples (NaNs) and applies a simple low-pass
exponential smoothing filter to highlight slow variations and suppress noise."""

import matplotlib.pyplot as plt
import numpy as np
import csv


def plot_distance_from_csv(csv_path: str) -> None:
    """RELEVANT FUNCTION INPUTS:
    - csv_path: path to a CSV file containing columns
        "Timestamp (s)" and "Distance (m)" (blank distance cells are treated as missing)

    Behavior:
    - Missing distances are filled by linear interpolation over time.
    - A first-order low-pass (exponential moving average) with α=0.95 is applied.
    - The plot shows detrended series (value minus its mean) for clear fluctuation comparison.
    """
    timestamps = []
    distances = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["Timestamp (s)"]))
            distance_str = row["Distance (m)"]
            distances.append(float(distance_str) if distance_str else np.nan)

    timestamps = np.array(timestamps, dtype=float)
    distances = np.array(distances, dtype=float)

    # --- Interpolate missing data points (NaNs) ---
    valid_mask = ~np.isnan(distances)
    if valid_mask.sum() < 2:
        raise ValueError("Not enough valid samples to interpolate.")
    distances_interpolated = np.interp(timestamps, timestamps[valid_mask], distances[valid_mask])

    # --- Low-pass filter (exponential moving average) ---
    alpha = 0.95
    distances_filtered = distances_interpolated.copy()
    for i in range(1, len(distances_filtered)):
        distances_filtered[i] = alpha * distances_filtered[i - 1] + (1 - alpha) * distances_filtered[i]

    # --- Detrend for fluctuation visualization ---
    detrended_measured = distances_interpolated - np.mean(distances_interpolated)
    detrended_filtered = distances_filtered - np.mean(distances_filtered)

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, detrended_measured, label="Measured distance")
    plt.plot(timestamps, detrended_filtered, label="Filtered (Low-pass) distance")
    plt.axhline(0, linestyle="--")  # reference line
    plt.title("UWB Fluctuations")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$d_t - \overline{d_t}$ (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage — replace with your file path
    plot_distance_from_csv("../output/testing_noise_reduction_raw.csv")
