"""This code can be used to visualize pitot tube airspeed fluctuations from a CSV file.
It recenters the airspeed signal around its mean, applies a simple low-pass filter
(exponential moving average), and plots the raw vs. filtered signals for comparison."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_airspeed_fluctuations(csv_path: str, alpha: float = 0.95) -> None:
    """RELEVANT FUNCTION INPUTS:
    - csv_path: path to the CSV file containing at least:
        * 'recording timestamp' (x-axis)
        * 'Airspeed in m/s'    (raw airspeed values)
    - alpha: smoothing factor for the low-pass filter (default = 0.95).
             Higher alpha = stronger smoothing.

    BEHAVIOR:
    - Loads the CSV file.
    - Recenters the airspeed values by subtracting their mean.
    - Applies a first-order low-pass filter to smooth the signal.
    - Plots raw fluctuations (centered) vs. smoothed signal.
    """

    # --- Step 1: Load the data ---
    data = pd.read_csv(csv_path)

    # Ensure required columns exist
    if "recording timestamp" not in data.columns or "Airspeed in m/s" not in data.columns:
        raise ValueError("CSV must contain 'recording timestamp' and 'Airspeed in m/s' columns.")

    # --- Step 2: Center airspeed around mean ---
    airspeed_avg = data["Airspeed in m/s"].mean()
    data["Airspeed_centered"] = data["Airspeed in m/s"] - airspeed_avg

    # --- Step 3: Apply low-pass filter (EMA) ---
    data["Airspeed_filtered"] = data["Airspeed_centered"].copy()
    for i in range(1, len(data)):
        data.loc[i, "Airspeed_filtered"] = (
            alpha * data.loc[i - 1, "Airspeed_filtered"]
            + (1 - alpha) * data.loc[i, "Airspeed_centered"]
        )

    # --- Step 4: Plot results ---
    plt.figure(figsize=(8, 4))
    plt.plot(data["recording timestamp"].values,
             data["Airspeed_centered"].values,
             label="Measured $V_\\infty$")
    plt.plot(data["recording timestamp"].values,
             data["Airspeed_filtered"].values,
             label="Filtered (Low-pass) $V_\\infty$")
    plt.axhline(0, color="k", linestyle="--")

    plt.xlabel("Timestamp")
    plt.ylabel(r"$V_\infty - \overline{V_\infty}$ (m/s)")
    plt.title("Airspeed Fluctuations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example call
    plot_airspeed_fluctuations("Calibration/data_inside_thursday/recording_IV10A0.csv", alpha=0.95)
