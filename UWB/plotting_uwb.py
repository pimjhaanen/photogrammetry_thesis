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
    valid_distances = distances[~np.isnan(distances)]
    mean_val = np.mean(valid_distances)

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, distances-mean_val, label="Unfiltered")
    plt.axhline(0, color="k", linestyle="--")
    plt.title("UWB Distance Measurement, fixed distance - mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_distance_from_csv("output/testing_noise_reduction.csv")