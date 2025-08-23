import pickle
import os
import pandas as pd

# List your calibration files (with paths or relative to script)
file_paths = [
    "calibration_charuco_linear_camera_2.pkl",
    "calibration_charuco_wide_camera_2.pkl",
    "calibration_checkerboard_linear_camera_2.pkl",
    "calibration_checkerboard_wide_camera_2.pkl",
    "calibration_circles_wide_camera_2.pkl"
]

# Store extracted data
data = []

for file_path in file_paths:
    with open(file_path, "rb") as f:
        calib = pickle.load(f)

    # Parse matrix and distortion coefficients
    mtx = calib["camera_matrix"]
    dist = calib["dist_coeffs"].flatten()
    error = calib.get("reprojection_error", None)

    # Create a row for the table
    entry = {
        "method": os.path.splitext(os.path.basename(file_path))[0],
        "fx": round(mtx[0, 0], 3),
        "fy": round(mtx[1, 1], 3),
        "cx": round(mtx[0, 2], 3),
        "cy": round(mtx[1, 2], 3),
        "k1": round(dist[0], 3) if len(dist) > 0 else None,
        "k2": round(dist[1], 3) if len(dist) > 1 else None,
        "p1": round(dist[2], 3) if len(dist) > 2 else None,
        "p2": round(dist[3], 3) if len(dist) > 3 else None,
        "k3": round(dist[4], 3) if len(dist) > 4 else None,
        "reprojection_error": round(error, 3) if error is not None else None
    }

    data.append(entry)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv("calibration_summary_camera_2.csv", index=False)
print("Saved to calibration_summary.csv")
