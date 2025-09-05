import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_distance_from_csv(csv_path):
    timestamps = []
    distances = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["Timestamp (s)"]))
            distance_str = row["Distance (m)"]
            distances.append(float(distance_str) if distance_str else np.nan)

    timestamps = np.array(timestamps)
    distances = np.array(distances)

    # Interpolate missing data points
    # Interpolation works with NaN values by filling them with the linear interpolation of neighboring values
    valid_mask = ~np.isnan(distances)
    distances_interpolated = np.interp(timestamps, timestamps[valid_mask], distances[valid_mask])

    # Apply a low-pass filter (LPF) with alpha = 0.9
    alpha = 0.95
    distances_filtered = distances_interpolated.copy()
    for i in range(1, len(distances_filtered)):
        distances_filtered[i] = alpha * distances_filtered[i - 1] + (1 - alpha) * distances_filtered[i]

    # Plotting
    plt.figure(figsize=(8, 4))

    # Plot original (unfiltered) and filtered distances
    plt.plot(timestamps, distances_interpolated - np.mean(distances_interpolated), label="Measured distance")
    plt.plot(timestamps, distances_filtered - np.mean(distances_filtered), label="Filtered (Low-pass) distance")

    # Add horizontal line at 0 for better visualization
    plt.axhline(0, color="k", linestyle="--")

    # Title and labels
    plt.title("UWB Fluctuations")
    plt.xlabel("Time (s)")
    plt.ylabel("$d_t-\overline{d_t}$ (m)")

    # Show legend and grid
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

# Call the function with your CSV file
plot_distance_from_csv("output/testing_noise_reduction_raw.csv")
