import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === 1) Load lookup table (CSV) ===
csv_file = 'testing_identifiers.csv'  # change path if needed
df = pd.read_csv(csv_file)

# First column is velocity; set as index
df = df.rename(columns={df.columns[0]: 'Va (m/s)'})
df.set_index('Va (m/s)', inplace=True)

# Make angle headers strings; keep also an int list for plotting/labels
df.columns = df.columns.map(str)
angles = [int(c) for c in df.columns]  # e.g. [0,5,10,15,20,25,30,35]

# === 2) Compute averaged measured angles per (Va, angle) over first 5 s ===
results = {angle: [] for angle in angles}

for va, row in df.iterrows():
    for angle in angles:
        file_name = row.loc[str(angle)]
        if isinstance(file_name, str) and os.path.isfile(file_name):
            data = pd.read_csv(file_name)
            if 'recording timestamp' in data.columns and 'Sideslip Angle' in data.columns:
                subset = data[data['recording timestamp'] <= 5]
                avg_angle = subset['Sideslip Angle'].mean()
                results[angle].append(avg_angle)
            else:
                results[angle].append(np.nan)
                print(f"⚠️ Columns missing in {file_name}")
        else:
            results[angle].append(np.nan)
            print(f"⚠️ File not found: {file_name}")

# Convert to arrays aligned with velocities
x_vel = df.index.to_numpy(dtype=float)  # velocities
Y_meas = {ang: np.array(vals, dtype=float) for ang, vals in results.items()}

# === 3) Build global calibration using only Va >= 10 m/s ===
# We fit: a * measured + b ≈ nominal_angle
x_mask = x_vel >= 10.0
calib_x = []  # measured (raw)
calib_y = []  # target = nominal angle

for ang in angles:
    y_raw = Y_meas[ang]
    if np.any(~np.isnan(y_raw[x_mask])):
        valid = x_mask & ~np.isnan(y_raw)
        calib_x.extend(y_raw[valid].tolist())
        calib_y.extend([ang] * np.sum(valid))

calib_x = np.array(calib_x, dtype=float)
calib_y = np.array(calib_y, dtype=float)

if calib_x.size >= 2:
    # Fit target = a * measured + b
    a, b = np.polyfit(calib_x, calib_y, 1)
    print(f"Global calibration: angle_cal = {a:.6f} * angle_meas + {b:.6f}")
else:
    a, b = 1.0, 0.0
    print("⚠️ Not enough data (Va ≥ 10 m/s) to fit calibration. Using identity (a=1, b=0).")

# === 4) Apply calibration to all points ===
Y_cal = {ang: a * Y_meas[ang] + b for ang in angles}

# === 5) Plot raw and calibrated lines (same color, calibrated dashed), legend OUTSIDE ===
plt.figure(figsize=(10, 6))
for ang in angles:
    y_raw = Y_meas[ang]
    y_cal = Y_cal[ang]
    # Raw
    plt.plot(x_vel, y_raw, marker='o', label=f'{ang}° raw')
    # Calibrated (same color as previous by picking last line color)
    color = plt.gca().lines[-1].get_color()
    plt.plot(x_vel, y_cal, linestyle='--', label=f'{ang}° calibrated', color=color)

plt.xlabel('Velocity (m/s)')
plt.ylabel('Angle')
plt.title('Angle vs velocity: raw and globally calibrated (fit from Va ≥ 10 m/s)')
plt.grid(True)
plt.legend(fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space right for legend
plt.show()

# === 6) Accuracy & plots for Va >= 10 m/s only (simple: mean±95% CI + central 95%) ===
import math

mask_10 = x_vel >= 10.0

# Collect residuals (calibrated - nominal) for Va >= 10 m/s
errors_10 = []
vels_10 = []
labels_10 = []

for ang in angles:
    y_cal = Y_cal[ang]
    for i in range(len(x_vel)):
        if mask_10[i] and np.isfinite(y_cal[i]):
            errors_10.append(y_cal[i] - ang)
            vels_10.append(x_vel[i])
            labels_10.append(ang)

errors_10 = np.asarray(errors_10, dtype=float)
vels_10   = np.asarray(vels_10, dtype=float)
labels_10 = np.asarray(labels_10, dtype=int)

N10 = errors_10.size
if N10 == 0:
    print("⚠️ No valid calibrated points with Va ≥ 10 m/s.")
else:
    mean10  = float(np.mean(errors_10))                  # bias
    std10   = float(np.std(errors_10, ddof=1)) if N10 > 1 else 0.0
    # 95% CI for the mean (bias)
    try:
        from scipy import stats
        t_val10 = stats.t.ppf(0.975, df=N10-1) if N10 > 1 else float('nan')
    except Exception:
        t_val10 = 1.96
    sem10   = std10 / math.sqrt(N10) if N10 > 0 else float('inf')
    ciL10   = mean10 - t_val10 * sem10
    ciH10   = mean10 + t_val10 * sem10

    # Central 95% of individual errors (empirical quantiles)
    qL, qH = np.percentile(errors_10, [2.5, 97.5])

    print("\n=== Calibration accuracy (Va ≥ 10 m/s) ===")
    print(f"N points             : {N10}")
    print(f"Mean error (bias)    : {mean10:.3f}°")
    print(f"95% CI of mean (bias): [{ciL10:.3f}°, {ciH10:.3f}°]")
    print(f"Std dev of errors    : {std10:.3f}°")
    print(f"Central 95% of errors: [{qL:.3f}°, {qH:.3f}°]  (empirical)")

    # --- Error distribution plot (Va >= 10 m/s), legend OUTSIDE ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors_10, bins=30, density=True, alpha=0.6, edgecolor='k', label="Errors (hist)")

    # Normal fit
    mu = mean10
    sigma = std10 if std10 > 0 else 1e-9
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs - mu)/sigma)**2)
    ax.plot(xs, pdf, linewidth=2, label="Normal fit")

    # Fill central 95% under the curve (lighter orange)
    mask_central = (xs >= qL) & (xs <= qH)
    ax.fill_between(xs[mask_central], 0, pdf[mask_central], alpha=0.20, color='orange',
                    label="Central 95% (empirical)")

    # Mean ± 95% CI of MEAN (blue band)
    ax.axvspan(ciL10, ciH10, alpha=0.15, color='tab:blue', label="95% CI of mean (bias)")

    # Mean line
    ax.axvline(mu, linestyle='-', linewidth=2, label=f"Mean = {mu:.2f}°")

    ax.set_title("Calibration error distribution (Va ≥ 10 m/s, calibrated − nominal)")
    ax.set_xlabel("Error [degrees]")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # room for legend
    plt.show()

# --- (Optional) Re-plot curves with 5 m/s hidden; legend OUTSIDE ---
plt.figure(figsize=(10, 6))
for ang in angles:
    y_raw = Y_meas[ang].copy()
    y_cal = Y_cal[ang].copy()
    y_raw[~mask_10] = np.nan
    y_cal[~mask_10] = np.nan

    plt.plot(x_vel, y_raw, marker='o', label=f'{ang}° raw')
    color = plt.gca().lines[-1].get_color()
    plt.plot(x_vel, y_cal, linestyle='--', label=f'{ang}° calibrated', color=color)

plt.xlabel('Velocity (m/s)')
plt.ylabel('Angle α (°)')
plt.title('Angle vs velocity (raw & calibrated)')
plt.grid(True)
plt.legend(fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()
