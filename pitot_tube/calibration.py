import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import linregress
import json
# Adjust this path to your directory
data_dir = 'data_inside_thursday'

# Automatically find and process relevant CSV files
plot_data = {}
aoa_plot_data = {}

for filename in os.listdir(data_dir):
    match = re.search(r'V(\d+)A(\d+)', filename)
    if match:
        windspeed = int(match.group(1))
        angle = int(match.group(2))

        # Skip wind speeds above 30
        if windspeed > 30:
            continue

        actual_wind_speed = windspeed

        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        # Take first 5 seconds
        df['recording timestamp'] -= df['recording timestamp'].iloc[0]
        df_5s = df[df['recording timestamp'] <= 5]

        # Correct angle anomalies
        aoa = df_5s['AOA'].apply(lambda x: x - 6553.5 if x > 180 else x).mean()

        # Prepare windspeed sensitivity data
        if actual_wind_speed not in plot_data:
            plot_data[actual_wind_speed] = {'angle': [], 'measured_ws': []}
        plot_data[actual_wind_speed]['angle'].append(angle)
        plot_data[actual_wind_speed]['measured_ws'].append(df_5s['Airspeed in m/s'].mean())

        # Prepare AoA accuracy data
        if angle not in aoa_plot_data:
            aoa_plot_data[angle] = {'windspeed': [], 'measured_aoa': []}
        aoa_plot_data[angle]['windspeed'].append(actual_wind_speed)
        aoa_plot_data[angle]['measured_aoa'].append(aoa)

# Prepare data at alpha=0°
actual_ws_alpha0, measured_ws_alpha0 = [], []

for ws, data in plot_data.items():
    for angle, measured_ws in zip(data['angle'], data['measured_ws']):
        if angle == 0:
            actual_ws_alpha0.append(ws)
            measured_ws_alpha0.append(measured_ws)

# Perform linear regression at alpha=0° only
ws_slope, ws_intercept, ws_r, _, _ = linregress(measured_ws_alpha0, actual_ws_alpha0)
print(f'Calibration (α=0°): Actual = {ws_slope:.3f} × Measured + {ws_intercept:.3f} (R²={ws_r ** 2:.4f})')

# === SAVE TO FILE ===
calibration_file = "pitot_calibration.json"
calibration_data = {"a": ws_slope, "b": ws_intercept}

with open(calibration_file, 'w') as f:
    json.dump(calibration_data, f, indent=4)
print(f"✅ Calibration saved to: {os.path.abspath(calibration_file)}")

# Apply this calibration to all wind speed data
for ws, data in plot_data.items():
    data['calibrated_ws'] = [ws_slope * v + ws_intercept for v in data['measured_ws']]

# First plot: Windspeed sensitivity to AoA
plt.figure(figsize=(10, 7))

colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

for color, (windspeed, data) in zip(colors, sorted(plot_data.items())):
    angles, measured_ws = zip(*sorted(zip(data['angle'], data['calibrated_ws'])))
    plt.plot(angles, measured_ws, 'o-', color=color, label=f'{windspeed} m/s')
    plt.hlines(windspeed, min(angles), max(angles), colors=color, linestyles='dotted')

plt.xlabel('Inflow Angle $\phi$ (°)')
plt.ylabel('Measured Wind Speed (m/s)')
plt.title('Wind Speed Accuracy and Sensitivity')
plt.grid(True)
plt.legend(title='Actual Speed', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# Collect unique inflow angles
angles_unique = sorted(set(angle for data in plot_data.values() for angle in data['angle']))
results = []

# Compute correct average relative errors per angle
for angle in angles_unique:
    abs_errors, actual_ws_values = [], []

    for actual_ws, data in plot_data.items():
        if angle in data['angle']:
            idx = data['angle'].index(angle)
            measured_ws = data['measured_ws'][idx]
            abs_errors.append(abs(measured_ws - actual_ws))
            actual_ws_values.append(actual_ws)

    avg_abs_error = np.mean(abs_errors)
    avg_actual_ws = np.mean(actual_ws_values)

    avg_rel_error = (avg_abs_error / avg_actual_ws) * 100 if avg_actual_ws != 0 else 0

    results.append({
        'Angle (°)': angle,
        'Avg Abs Error (m/s)': avg_abs_error,
        'Avg Rel Error (%)': avg_rel_error
    })

# Results table
results_df = pd.DataFrame(results)

# Overall averages
overall_avg_abs_error = results_df['Avg Abs Error (m/s)'].mean()
overall_avg_rel_error = results_df['Avg Rel Error (%)'].mean()

# Print detailed table
print("Corrected error summary per inflow angle:")
print(results_df.round(3).to_string(index=False))

# Print overall averages
print("\nOverall average errors across all angles:")
print(f"Average Absolute Error: {overall_avg_abs_error:.3f} m/s")
print(f"Average Relative Error: {overall_avg_rel_error:.3f} %")

# AoA error calculation per windspeed (10-30 m/s)
aoa_error_results = []

# Loop through wind speeds 10 m/s to 30 m/s
for windspeed in range(10, 31):
    abs_errors, rel_errors = [], []

    for angle, data in aoa_plot_data.items():
        if windspeed in data['windspeed']:
            # Get corresponding measured AoA
            idx = data['windspeed'].index(windspeed)
            measured_aoa = data['measured_aoa'][idx]

            # Calculate error between measured and actual AoA
            abs_error = abs(measured_aoa - angle)
            rel_error = (abs_error / angle) * 100 if angle != 0 else 0

            abs_errors.append(abs_error)
            rel_errors.append(rel_error)

    if abs_errors:  # Avoid empty lists
        avg_abs_error = np.mean(abs_errors)
        avg_rel_error = np.mean(rel_errors)

        aoa_error_results.append({
            'Windspeed (m/s)': windspeed,
            'Avg Abs Error (°)': avg_abs_error,
            'Avg Rel Error (%)': avg_rel_error
        })

# Results table for AoA error
aoa_results_df = pd.DataFrame(aoa_error_results)

# Overall averages for AoA
overall_avg_aoa_abs_error = aoa_results_df['Avg Abs Error (°)'].mean()
overall_avg_aoa_rel_error = aoa_results_df['Avg Rel Error (%)'].mean()

# Print AoA error table
print("\nCorrected AoA error summary per wind speed:")
print(aoa_results_df.round(3).to_string(index=False))

# Print overall averages for AoA
print("\nOverall average errors across all wind speeds:")
print(f"Average Absolute Error: {overall_avg_aoa_abs_error:.3f} °")
print(f"Average Relative Error: {overall_avg_aoa_rel_error:.3f} %")

# Final Plot: Measured vs Actual AoA per Wind Speed
plt.figure(figsize=(10, 7))

# Create color map based on the number of angle setpoints
colors = plt.cm.plasma(np.linspace(0, 1, len(aoa_plot_data)))

for color, (angle, data) in zip(colors, sorted(aoa_plot_data.items())):
    sorted_data = sorted(zip(data['windspeed'], data['measured_aoa']))
    windspeed_sorted, measured_aoa_sorted = zip(*sorted_data)

    plt.plot(windspeed_sorted, np.abs(measured_aoa_sorted), 'o-', color=color, label=f'{angle}° Measured')
    plt.hlines(angle, min(windspeed_sorted), max(windspeed_sorted), colors=color, linestyles='dotted',
               label=f'{angle}° Actual')

plt.xlabel('Wind Tunnel Wind Speed (m/s)')
plt.ylabel('Angle of Attack (°)')
plt.title('Measured vs Actual Angle of Attack per Wind Speed')
plt.grid(True)
plt.legend(title='Angle of Attack', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

