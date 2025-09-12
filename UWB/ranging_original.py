from pypozyx import (PozyxSerial, POZYX_SUCCESS, get_first_pozyx_serial_port, DeviceRange)
import time
import numpy as np
import csv
import os
import json
import tkinter as tk
import RPi.GPIO as GPIO
from datetime import datetime, timezone

# =========================
# GPIO / Sync flash
# =========================
GPIO.setmode(GPIO.BCM)   # Use BCM numbering
GPIO.setup(18, GPIO.OUT) # LED or sync pin on GPIO18

def flash_sync(duration_ms=100):
    GPIO.output(18, GPIO.HIGH)
    t0_epoch = time.time()  # epoch at the instant the LED is driven HIGH
    print(f"⚡ Sync LED ON @ {datetime.fromtimestamp(t0_epoch, tz=timezone.utc).isoformat()}")
    time.sleep(duration_ms / 1000)
    GPIO.output(18, GPIO.LOW)
    return t0_epoch


# =========================
# Post-processing
# =========================
def apply_postprocessing(
    raw_csv_path,
    calibration_path="calibration/uwb_calibration.json",
    low_pass_filter=True
):
    # Load calibration (y = a*x + b)
    with open(calibration_path, 'r') as f:
        calib = json.load(f)
    a, b = calib["a"], calib["b"]

    # Read raw data (epoch seconds + raw distance)
    epochs = []
    distances_raw = []
    with open(raw_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # "Timestamp (s)" now stores epoch seconds
            t = row.get("Timestamp (s)")
            d = row.get("Distance (m)")
            if t is None:
                continue
            epochs.append(float(t))
            distances_raw.append(float(d) if d not in (None, "",) else np.nan)

    if not epochs:
        raise ValueError("No samples found in raw CSV.")

    # Convert to UTC strings and relative timestamps
    start_epoch = epochs[0]
    elapsed_s = [ts - start_epoch for ts in epochs]
    utc_iso = [datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() for ts in epochs]

    # Calibrate (preserve NaNs)
    calibrated = [a * d + b if not np.isnan(d) else np.nan for d in distances_raw]

    # Interpolate NaNs so the series is complete (optional, same as your previous logic)
    x = np.arange(len(calibrated))
    y = np.array(calibrated, dtype=float)
    mask = ~np.isnan(y)

    if mask.any():
        interp = np.interp(x, x[mask], y[mask])
    else:
        # If everything is NaN, keep NaNs
        interp = np.full_like(y, np.nan, dtype=float)

    sources = ['reality' if not np.isnan(orig) else 'interpolated' for orig in calibrated]

    # Optional simple low-pass filter (EMA)
    if low_pass_filter and mask.any():
        alpha = 0.95
        filtered = []
        last = None
        for val in interp:
            last = val if last is None else alpha * last + (1 - alpha) * val
            filtered.append(last)
        interp = filtered

    # Write processed CSV: UTC + relative timestamp + calibrated distance
    output_path = raw_csv_path.replace("_raw.csv", ".csv")
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["UTC (ISO8601)", "Timestamp (s)", "Distance (m)", "Source"])
        for u, rel, d, s in zip(utc_iso, elapsed_s, interp, sources):
            writer.writerow([u, round(rel, 6), "" if np.isnan(d) else round(float(d), 6), s])

    return output_path

# =========================
# Filenaming
# =========================
now = datetime.now(timezone.utc)
timestamp_str = now.strftime("%Y%m%d_%H%M%S")
filename = f"uwb_flight_{timestamp_str}"
os.makedirs("output", exist_ok=True)
csv_path = os.path.join("output", filename + "_raw.csv")

# =========================
# Pozyx setup
# =========================
serial_port = get_first_pozyx_serial_port()
if serial_port is None:
    print("❌ No Pozyx connected.")
    GPIO.cleanup()
    raise SystemExit

remote_id = 0x6923
destination_id = 0x6931
pozyx = PozyxSerial(serial_port)
device_range = DeviceRange()

pozyx.setTxPower(33, remote_id)
pozyx.setTxPower(33, destination_id)

# =========================
# Measurement loop
# =========================
distances = []
epochs = []

try:
    # Take sync flash; first timestamp is t=0 reference
    t0_epoch = flash_sync()
    epochs.append(t0_epoch)
    distances.append(np.nan)  # sync marker row (no distance)

    while True:
        status = pozyx.doRanging(destination_id, device_range, remote_id)
        now_epoch = time.time()  # timestamp the just-returned ranging sample

        if get_first_pozyx_serial_port() is None:
            print("Pozyx has been disconnected")
            break

        if status == POZYX_SUCCESS:
            dist_m = device_range.distance / 1000.0
            distances.append(dist_m)
            epochs.append(now_epoch)
            print(f"📏 Distance: {round(dist_m, 3)} m")
        else:
            print("⚠️ Sensors out of reach")
            distances.append(np.nan)
            epochs.append(now_epoch)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("⏹️ Script interrupted with Ctrl+C. Saving...")


finally:
    # === SAVE RAW ===
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Store epoch seconds in "Timestamp (s)" as requested
        writer.writerow(["Timestamp (s)", "Distance (m)"])
        for t, d in zip(epochs, distances):
            writer.writerow([round(t, 6), "" if np.isnan(d) else round(float(d), 6)])

    print(f"✅ Data written to: {os.path.abspath(csv_path)}")

    # === POSTPROCESS ===
    try:
        postprocessed_path = apply_postprocessing(os.path.abspath(csv_path))
        print(f"✅ Processed data written to: {os.path.abspath(postprocessed_path)}")
    except Exception as e:
        print(f"❗ Postprocessing failed: {e}")

    GPIO.cleanup()
