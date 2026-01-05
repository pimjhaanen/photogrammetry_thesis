"""This code can be used to calibrate and evaluate a pitot + AoA probe from a folder of CSV files.
It auto-discovers files named like '...V{windspeed}A{angle}...' (e.g. recording_IV10A0.csv), takes
the first N seconds of each file, computes mean wind speed and mean AoA, fits a linear wind-speed
calibration at α=0° (Actual = a * Measured + b), saves it to JSON, and generates accuracy plots/tables.

Assumptions:
- Each CSV contains columns: 'recording timestamp', 'Airspeed in m/s', and 'AOA'.
- 'recording timestamp' is in seconds (monotonic); we rebase so that t=0 at the first sample.
- AoA sensor may wrap; values > 180 are shifted by 6553.5 → adjust this if your device differs.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D
def analyze_pitot_dataset(
    data_dir: str,
    take_first_seconds: float = 5.0,
    max_windspeed_for_scan: int = 30,
    aoa_wrap_threshold_deg: float = 180.0,
    aoa_wrap_subtract: float = 6553.5,
    calib_json_path: str = "pitot_calibration.json",
) -> Dict[str, pd.DataFrame]:
    """RELEVANT FUNCTION INPUTS:
    - data_dir: folder with CSV files. Filenames must contain 'V{ws}A{angle}', e.g. '...V15A10...csv'
    - take_first_seconds: window length per file to compute means (default: 5 s)
    - max_windspeed_for_scan: ignore files with ws > this value in the discovery loop (default: 30)
    - aoa_wrap_threshold_deg: if AOA sample > this threshold, subtract 'aoa_wrap_subtract' (wrap fix)
    - aoa_wrap_subtract: value subtracted when AOA > threshold (device-specific constant)
    - calib_json_path: path to save the linear calibration coefficients for wind speed

    RETURNS:
    - dict of DataFrames:
        {
          "wind_ws_table":  per-windspeed AoA sensitivity table (angles, measured & calibrated V),
          "angle_error":    per-angle average absolute & relative wind-speed errors,
          "aoa_error":      per-windspeed average absolute & relative AoA errors,
        }
    """

    # --- Discover files and aggregate means (first N seconds) ---
    plot_data: Dict[int, Dict[str, List[float]]] = {}     # {ws: {"angle":[], "measured_ws":[]}}
    aoa_plot_data: Dict[int, Dict[str, List[float]]] = {} # {angle: {"windspeed":[], "measured_aoa":[]}}

    for filename in os.listdir(data_dir):
        match = re.search(r"V(\d+)A(\d+)", filename)
        if not match:
            continue

        windspeed = int(match.group(1))
        angle = int(match.group(2))
        if windspeed > max_windspeed_for_scan:
            continue

        filepath = os.path.join(data_dir, filename)
        if not filepath.lower().endswith(".csv"):
            continue

        df = pd.read_csv(filepath)

        required_cols = {"recording timestamp", "Airspeed in m/s", "AOA"}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Skipping {filename}: missing required columns {required_cols - set(df.columns)}")
            continue

        # Rebase time and take first N seconds
        df["recording timestamp"] = df["recording timestamp"] - df["recording timestamp"].iloc[0]
        df_5s = df[df["recording timestamp"] <= take_first_seconds].copy()
        if df_5s.empty:
            print(f"⚠️ Skipping {filename}: no samples in first {take_first_seconds}s.")
            continue

        # AoA wrap fix and mean
        aoa_series = df_5s["AOA"].apply(lambda x: x - aoa_wrap_subtract if x > aoa_wrap_threshold_deg else x)
        aoa_mean = float(aoa_series.mean())

        # Wind-speed mean (measured)
        ws_meas_mean = float(df_5s["Airspeed in m/s"].mean())

        # Accumulate wind-speed sensitivity data
        if windspeed not in plot_data:
            plot_data[windspeed] = {"angle": [], "measured_ws": []}
        plot_data[windspeed]["angle"].append(angle)
        plot_data[windspeed]["measured_ws"].append(ws_meas_mean)

        # Accumulate AoA accuracy data
        if angle not in aoa_plot_data:
            aoa_plot_data[angle] = {"windspeed": [], "measured_aoa": []}
        aoa_plot_data[angle]["windspeed"].append(windspeed)
        aoa_plot_data[angle]["measured_aoa"].append(aoa_mean)

    if not plot_data:
        raise RuntimeError("No valid files found. Ensure filenames contain 'V{ws}A{angle}' and CSV schema matches.")

    # --- Calibration from α = 0° only ---
    actual_ws_alpha0, measured_ws_alpha0 = [], []
    for ws, data in plot_data.items():
        for angle, measured_ws in zip(data["angle"], data["measured_ws"]):
            if angle == 0:
                actual_ws_alpha0.append(ws)
                measured_ws_alpha0.append(measured_ws)

    if len(actual_ws_alpha0) < 2:
        raise RuntimeError("Not enough α=0° samples to fit calibration (need ≥2).")

    ws_slope, ws_intercept, ws_r, _, _ = linregress(measured_ws_alpha0, actual_ws_alpha0)
    print(f"Calibration (α=0°): Actual = {ws_slope:.3f} × Measured + {ws_intercept:.3f}  (R²={ws_r**2:.4f})")

    # Save calibration
    with open(calib_json_path, "w") as f:
        json.dump({"a": float(ws_slope), "b": float(ws_intercept)}, f, indent=4)
    print(f"✅ Calibration saved to: {os.path.abspath(calib_json_path)}")

    # Apply calibration to all measured wind speeds
    for ws, data in plot_data.items():
        data["calibrated_ws"] = [ws_slope * v + ws_intercept for v in data["measured_ws"]]

    # --- Plot 1: wind-speed vs AoA, per true ws (RAW vs CALIBRATED), legend OUTSIDE ---
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    windspeed_handles = []

    for windspeed, data in sorted(plot_data.items()):
        pairs = sorted(zip(data["angle"], data["measured_ws"], data["calibrated_ws"]))
        angs = [p[0] for p in pairs]
        ws_meas = [p[1] for p in pairs]
        ws_cal = [p[2] for p in pairs]

        # Measured defines colour, but only windspeed goes in legend
        h_meas, = ax.plot(angs, ws_meas, "o-", label=f"{windspeed} m/s")
        color = h_meas.get_color()
        windspeed_handles.append(h_meas)

        # Calibrated (same colour) -> NOT in legend
        ax.plot(angs, ws_cal, "--", color=color, label="_nolegend_")

        # Truth crosses (same colour) -> NOT in legend
        ax.plot(
            angs, [windspeed] * len(angs),
            linestyle="None", marker="x",
            markersize=7, markeredgewidth=1.5,
            color=color, label="_nolegend_"
        )

    ax.set_xlabel(r"Inflow Angle $\phi$ (°)")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(True)

    # Style key (black)
    style_handles = [
        Line2D([0], [0], color="k", marker="o", linestyle="-", label="measured"),
        Line2D([0], [0], color="k", linestyle="--", label="calibrated"),
        Line2D([0], [0], color="k", marker="x", linestyle="None", markersize=7, label="truth"),
    ]

    # One combined legend: first the style key, then wind speeds
    handles = style_handles + windspeed_handles
    labels = [h.get_label() for h in handles]

    ax.legend(
        handles, labels,
        title="Legend",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.,
        fontsize=9,
    )

    fig.subplots_adjust(right=0.75)
    plt.show()

    # Build a per-windspeed table (optional, for inspection)
    wind_ws_rows = []
    for ws, data in sorted(plot_data.items()):
        for ang, meas, cal in zip(data["angle"], data["measured_ws"], data["calibrated_ws"]):
            wind_ws_rows.append({"True WS (m/s)": ws, "Angle (°)": ang, "Measured (m/s)": meas, "Calibrated (m/s)": cal})
    wind_ws_df = pd.DataFrame(wind_ws_rows).sort_values(["True WS (m/s)", "Angle (°)"])

    # --- Per-angle wind-speed error summary (using CALIBRATED values) ---
    # Also assemble a flat list of residuals for a simple accuracy plot.
    errors_ws = []  # calibrated - actual
    for ws, data in sorted(plot_data.items()):
        for ang, cal in zip(data["angle"], data["calibrated_ws"]):
            errors_ws.append(cal - ws)
    errors_ws = np.asarray(errors_ws, dtype=float)

    if errors_ws.size:
        mean_ws  = float(np.mean(errors_ws))
        std_ws   = float(np.std(errors_ws, ddof=1)) if errors_ws.size > 1 else 0.0
        # 95% CI for mean
        try:
            from scipy import stats
            t_val = stats.t.ppf(0.975, df=errors_ws.size-1) if errors_ws.size > 1 else float('nan')
        except Exception:
            t_val = 1.96
        sem = std_ws / np.sqrt(errors_ws.size) if errors_ws.size > 0 else float('inf')
        ciL, ciH = mean_ws - t_val*sem, mean_ws + t_val*sem
        qL, qH = np.percentile(errors_ws, [2.5, 97.5])

        print("\nWind-speed accuracy (calibrated vs actual, all angles & ws≤max):")
        print(f"N points             : {errors_ws.size}")
        print(f"Mean error (bias)    : {mean_ws:.3f} m/s")
        print(f"95% CI of mean       : [{ciL:.3f}, {ciH:.3f}] m/s")
        print(f"Std dev of errors    : {std_ws:.3f} m/s")
        print(f"Central 95% of errors: [{qL:.3f}, {qH:.3f}] m/s  (empirical)")

        # Simple histogram with normal fit + shading, legend OUTSIDE
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(errors_ws, bins=30, density=True, alpha=0.6, edgecolor='k', label="Errors (hist)")
        sigma = std_ws if std_ws > 0 else 1e-9
        xs = np.linspace(mean_ws - 4*sigma, mean_ws + 4*sigma, 400)
        pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs - mean_ws)/sigma)**2)
        ax.plot(xs, pdf, linewidth=2, label="Normal fit")

        # Shaded central 95% (light orange)
        mask_cent = (xs >= qL) & (xs <= qH)
        ax.fill_between(xs[mask_cent], 0, pdf[mask_cent], alpha=0.20, color='orange',
                        label="Central 95% (empirical)")
        # 95% CI of mean (blue band)
        ax.axvspan(ciL, ciH, alpha=0.15, color='tab:blue', label="95% CI of mean (bias)")
        ax.axvline(mean_ws, linestyle='-', linewidth=2, label=f"Mean = {mean_ws:.2f} m/s")

        ax.set_title("Wind-speed error distribution (calibrated − actual)")
        ax.set_xlabel("Error [m/s]")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    # --- Per-angle table: average absolute & relative wind-speed errors (CALIBRATED) ---
    angles_unique = sorted({ang for d in plot_data.values() for ang in d["angle"]})
    angle_results = []
    for angle in angles_unique:
        abs_errors, actual_ws_vals = [], []
        for actual_ws, data in plot_data.items():
            if angle in data["angle"]:
                idx = data["angle"].index(angle)
                cal_ws = data["calibrated_ws"][idx]
                abs_errors.append(abs(cal_ws - actual_ws))
                actual_ws_vals.append(actual_ws)
        if abs_errors:
            avg_abs_error = float(np.mean(abs_errors))
            avg_actual_ws = float(np.mean(actual_ws_vals))
            avg_rel_error = (avg_abs_error / avg_actual_ws) * 100 if avg_actual_ws != 0 else 0.0
            angle_results.append(
                {"Angle (°)": angle, "Avg Abs Error (m/s)": avg_abs_error, "Avg Rel Error (%)": avg_rel_error}
            )
    angle_err_df = pd.DataFrame(angle_results).sort_values("Angle (°)")
    if not angle_err_df.empty:
        overall_avg_abs_error = float(angle_err_df["Avg Abs Error (m/s)"].mean())
        overall_avg_rel_error = float(angle_err_df["Avg Rel Error (%)"].mean())
        print("\nWind-speed error summary per inflow angle (calibrated):")
        print(angle_err_df.round(3).to_string(index=False))
        print("\nOverall average errors across all angles:")
        print(f"Average Absolute Error: {overall_avg_abs_error:.3f} m/s")
        print(f"Average Relative Error: {overall_avg_rel_error:.3f} %")

    # --- AoA error per windspeed (10–max), simple tables only (no extra plots) ---
    aoa_error_rows = []
    for windspeed in range(10, max_windspeed_for_scan + 1):
        abs_errors, rel_errors = [], []
        for angle, data in aoa_plot_data.items():
            if windspeed in data["windspeed"]:
                idx = data["windspeed"].index(windspeed)
                measured_aoa = data["measured_aoa"][idx]
                abs_err = abs(measured_aoa - angle)
                rel_err = (abs_err / angle) * 100 if angle != 0 else 0.0
                abs_errors.append(abs_err)
                rel_errors.append(rel_err)
        if abs_errors:
            aoa_error_rows.append(
                {
                    "Windspeed (m/s)": windspeed,
                    "Avg Abs Error (°)": float(np.mean(abs_errors)),
                    "Avg Rel Error (%)": float(np.mean(rel_errors)),
                }
            )
    aoa_err_df = pd.DataFrame(aoa_error_rows).sort_values("Windspeed (m/s)")
    if not aoa_err_df.empty:
        overall_avg_aoa_abs_error = float(aoa_err_df["Avg Abs Error (°)"].mean())
        overall_avg_aoa_rel_error = float(aoa_err_df["Avg Rel Error (%)"].mean())
        print("\nAoA error summary per wind speed (measured vs nominal):")
        print(aoa_err_df.round(3).to_string(index=False))
        print("\nOverall average AoA errors across all wind speeds:")
        print(f"Average Absolute Error: {overall_avg_aoa_abs_error:.3f} °")
        print(f"Average Relative Error: {overall_avg_aoa_rel_error:.3f} %")
    else:
        print("\nNo AoA results (did not find matching files).")

    # --- Plot 2 (unchanged style, legend OUTSIDE): Measured vs Actual AoA per windspeed ---
    if aoa_plot_data:
        plt.figure(figsize=(10, 7))
        for angle, data in sorted(aoa_plot_data.items()):
            sorted_pairs = sorted(zip(data["windspeed"], data["measured_aoa"]))
            ws_sorted, aoa_meas_sorted = zip(*sorted_pairs)
            line_meas, = plt.plot(ws_sorted, np.abs(aoa_meas_sorted), "o-", label=f"{angle}° Measured")
            color = line_meas.get_color()
            plt.hlines(angle, min(ws_sorted), max(ws_sorted), colors=color, linestyles="dotted", label=f"{angle}° Actual")
        plt.xlabel("Wind Tunnel Wind Speed (m/s)")
        plt.ylabel("Angle of Attack (°)")
        plt.title("Measured vs Actual Angle of Attack per Wind Speed")
        plt.grid(True)
        plt.legend(title="Angle of Attack", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    return {
        "wind_ws_table": wind_ws_df,
        "angle_error": angle_err_df,
        "aoa_error": aoa_err_df,
    }

if __name__ == "__main__":
    analyze_pitot_dataset(
        data_dir="data_inside_thursday",
        take_first_seconds=5.0,
        max_windspeed_for_scan=30,
        aoa_wrap_threshold_deg=180.0,
        aoa_wrap_subtract=6553.5,
        calib_json_path="pitot_calibration.json",
    )
