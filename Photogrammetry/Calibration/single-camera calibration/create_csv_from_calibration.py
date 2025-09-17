"""This code can be used to summarize the calibration of the three different calibration boards:
the chequerboard, charuco and circles boards. Different lenses can also be compared."""

import os
import glob
import pickle
import pandas as pd
from typing import Iterable, List, Union

def summarize_calibrations(
    file_paths: Iterable[str],
    output_csv: str = "calibration_summary_camera_2.csv",
    round_ndigits: int = 3
) -> pd.DataFrame:

    """RELEVANT FUNCTION INPUTS:
    -file_paths: name calibration files you want to summarize in a csv
    -output_csv: name of the output file
    -round_ndigits: the number of digits calibration parametres should be rounded to"""
    # Expand any glob patterns and deduplicate while preserving order
    expanded: List[str] = []
    seen = set()
    for p in file_paths:
        matches = glob.glob(p) or [p]  # keep literal if no glob match
        for m in matches:
            if m not in seen:
                expanded.append(m)
                seen.add(m)

    rows = []
    for file_path in expanded:
        try:
            with open(file_path, "rb") as f:
                calib = pickle.load(f)
        except FileNotFoundError:
            print(f"[warn] Not found: {file_path}")
            continue
        except Exception as e:
            print(f"[warn] Failed to load {file_path}: {e}")
            continue

        mtx = calib.get("camera_matrix", None)
        dist = calib.get("dist_coeffs", None)
        error = calib.get("reprojection_error", None)

        if mtx is None or dist is None:
            print(f"[warn] Missing keys in {file_path} (need 'camera_matrix' and 'dist_coeffs'). Skipping.")
            continue

        # Ensure dist is 1D
        try:
            dist_flat = dist.flatten()
        except Exception:
            dist_flat = dist

        def r(x):
            return round(float(x), round_ndigits) if x is not None else None

        row = {
            "method": os.path.splitext(os.path.basename(file_path))[0],
            "fx": r(mtx[0, 0]),
            "fy": r(mtx[1, 1]),
            "cx": r(mtx[0, 2]),
            "cy": r(mtx[1, 2]),
            "k1": r(dist_flat[0]) if len(dist_flat) > 0 else None,
            "k2": r(dist_flat[1]) if len(dist_flat) > 1 else None,
            "p1": r(dist_flat[2]) if len(dist_flat) > 2 else None,
            "p2": r(dist_flat[3]) if len(dist_flat) > 3 else None,
            "k3": r(dist_flat[4]) if len(dist_flat) > 4 else None,
            "reprojection_error": r(error) if error is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {'single_calibration_output/'+output_csv}")
    return df


if __name__ == "__main__":
    # Example usage (your original list):
    FILES = [
        "calibration_charuco_linear_camera_2.pkl",
        "calibration_charuco_wide_camera_2.pkl",
        "calibration_checkerboard_linear_camera_2.pkl",
        "calibration_checkerboard_wide_camera_2.pkl",
        "calibration_circles_wide_camera_2.pkl",
        # You can also use wildcards, e.g. "single_calibration_output/*.pkl"
    ]
    summarize_calibrations(FILES, output_csv="calibration_summary_camera_2.csv", round_ndigits=3)
