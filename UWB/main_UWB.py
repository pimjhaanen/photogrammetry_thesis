""" This code should be transfered and run on the Raspberry Pi, with the UWB shield connected via Micro USB.
This code can be used to log Pozyx UWB distances with a millisecond-accurate UTC timestamp,
emit a GPIO sync flash for cross-device alignment, and post-process the log (calibration + smoothing).
Use it to do simple range logging during a kite flight or a walk test: start the script, the LED flashes
once as t=0, then *walk away with one device* to see link quality and distance over time."""

from pypozyx import (
    PozyxSerial, POZYX_SUCCESS, get_first_pozyx_serial_port, DeviceRange
)
import time
import os
import csv
import json
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt  # optional (not required for logging)
from datetime import datetime, timezone

# GPIO is only available on a Raspberry Pi; guard the import for portability.
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False


# =========================
# GPIO / Sync flash
# =========================
def setup_gpio(pin: int = 18) -> None:
    """RELEVANT FUNCTION INPUTS:
    - pin: BCM pin number used to drive an LED (or sync output). Default: 18

    Sets the GPIO mode and configures the pin as an output (HIGH=ON). Does nothing if GPIO is unavailable.
    """
    if not _HAS_GPIO:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)


def flash_sync(duration_ms: int = 100, pin: int = 18) -> float:
    """RELEVANT FUNCTION INPUTS:
    - duration_ms: how long to keep the LED ON (milliseconds)
    - pin: BCM pin number to toggle

    RETURNS:
    - t0_epoch: epoch seconds (float) captured at the instant the LED is driven HIGH

    Emits a brief LED flash and returns the precise UTC epoch timestamp for sync with other sensors/cameras.
    """
    if _HAS_GPIO:
        GPIO.output(pin, GPIO.HIGH)
    t0_epoch = time.time()  # epoch at the instant the LED is driven HIGH
    print(f"⚡ Sync LED ON @ {datetime.fromtimestamp(t0_epoch, tz=timezone.utc).isoformat()}")
    time.sleep(duration_ms / 1000.0)
    if _HAS_GPIO:
        GPIO.output(pin, GPIO.LOW)
    return t0_epoch


def cleanup_gpio() -> None:
    """RELEVANT FUNCTION INPUTS:
    (none)

    Cleans up GPIO state if available (safe to call on non-RPi)."""
    if _HAS_GPIO:
        GPIO.cleanup()


# =========================
# Post-processing
# =========================
def apply_postprocessing(
    raw_csv_path: str,
    calibration_path: Optional[str] = "calibration/uwb_calibration.json",
    apply_low_pass: bool = True,
    alpha: float = 0.95,
) -> str:
    """RELEVANT FUNCTION INPUTS:
    - raw_csv_path: path to the RAW CSV produced by the logger (columns: 'Timestamp (s)', 'Distance (m)')
                    where 'Timestamp (s)' stores *epoch seconds*
    - calibration_path: JSON file with linear calibration coefficients {"a": <float>, "b": <float>};
                        if None or file missing, identity calibration is used (a=1, b=0)
    - apply_low_pass: if True, apply exponential moving average (EMA) smoothing to the calibrated series
    - alpha: EMA smoothing factor (0..1); closer to 1 = smoother

    RETURNS:
    - output CSV path with processed data (columns: 'UTC (ISO8601)','Timestamp (s)','Distance (m)','Source')
    """
    # Load calibration (y = a*x + b)
    a, b = 1.0, 0.0
    if calibration_path and os.path.isfile(calibration_path):
        with open(calibration_path, "r") as f:
            calib = json.load(f)
        a, b = float(calib.get("a", 1.0)), float(calib.get("b", 0.0))
        print(f"🧪 Using calibration: y = {a:.6f} * x + {b:.6f}")
    else:
        print("ℹ️ No calibration file found — using identity (a=1, b=0).")

    # Read raw data (epoch seconds + raw distance)
    epochs: List[float] = []
    distances_raw: List[float] = []
    with open(raw_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    y = np.array([a * d + b if not np.isnan(d) else np.nan for d in distances_raw], dtype=float)

    # Interpolate NaNs so the series is complete (linear in index space)
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() >= 2:
        y_interp = np.interp(x, x[mask], y[mask])
    else:
        y_interp = y.copy()  # not enough valid samples to interpolate

    sources = ["reality" if not np.isnan(orig) else "interpolated" for orig in y]

    # Optional simple low-pass filter (EMA)
    if apply_low_pass and mask.any():
        ema = []
        last = None
        for val in y_interp:
            last = val if last is None else alpha * last + (1 - alpha) * val
            ema.append(last)
        y_out = np.array(ema, dtype=float)
    else:
        y_out = y_interp

    # Write processed CSV: UTC + relative timestamp + calibrated distance
    output_path = raw_csv_path.replace("_raw.csv", ".csv")
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["UTC (ISO8601)", "Timestamp (s)", "Distance (m)", "Source"])
        for u, rel, d, s in zip(utc_iso, elapsed_s, y_out, sources):
            writer.writerow([u, f"{rel:.6f}", "" if np.isnan(d) else f"{float(d):.6f}", s])

    print(f"✅ Processed data written to: {os.path.abspath(output_path)}")
    return output_path


# =========================
# UWB logger
# =========================
def run_uwb_logger(
    remote_id: int,
    destination_id: int,
    tx_power_remote: int = 33,
    tx_power_destination: int = 33,
    sample_dt_s: float = 0.01,
    output_dir: str = "output",
    calibration_path: Optional[str] = "calibration/uwb_calibration.json",
    gpio_pin: int = 18,
    flash_ms: int = 100,
    do_postprocess: bool = True,
) -> Tuple[str, Optional[str]]:
    """RELEVANT FUNCTION INPUTS:
    - remote_id: Pozyx 16-bit ID initiating the ranging (e.g. 0x6923)
    - destination_id: Pozyx 16-bit ID being ranged (e.g. 0x6931)
    - tx_power_remote: TX power level for the remote device (Pozyx units, e.g. 33)
    - tx_power_destination: TX power level for the destination device (Pozyx units, e.g. 33)
    - sample_dt_s: delay between ranging attempts in seconds (e.g. 0.01 ≈ 100 Hz max loop)
    - output_dir: directory to store CSV outputs
    - calibration_path: path to JSON with {'a','b'} for linear calibration; set None to skip calibration
    - gpio_pin: BCM pin number for LED/sync output (ignored if GPIO unavailable), for synchronisation with gopro
    - flash_ms: duration of sync flash (milliseconds)
    - do_postprocess: if True, immediately run post-processing on the saved RAW CSV, after ranging finished

    BEHAVIOR / INSTRUCTIONS:
    - On start, the code toggles an LED (if available) and records the exact UTC epoch for sync (t=0) with the gopro.
    - RAW CSV saves epoch seconds and raw distances. Post-processing adds UTC ISO time, relative time,
      linear calibration, and optional EMA smoothing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filename stamp
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    base = f"uwb_flight_{timestamp_str}"
    raw_csv_path = os.path.join(output_dir, base + "_raw.csv")

    # Pozyx setup
    serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("❌ No Pozyx connected.")
        return "", None

    pozyx = PozyxSerial(serial_port)
    device_range = DeviceRange()

    # Best-effort TX power config (ignore failures silently)
    try:
        pozyx.setTxPower(tx_power_remote, remote_id)
        pozyx.setTxPower(tx_power_destination, destination_id)
    except Exception as e:
        print(f"ℹ️ Could not set TX power on one or both devices: {e}")

    # GPIO
    setup_gpio(gpio_pin)

    distances: List[float] = []
    epochs: List[float] = []

    try:
        # Sync flash (first row is the marker with NaN distance)
        t0_epoch = flash_sync(duration_ms=flash_ms, pin=gpio_pin)
        epochs.append(t0_epoch)
        distances.append(np.nan)  # marker row

        print("▶️ Starting UWB ranging loop (Ctrl+C to stop)...")
        while True:
            status = pozyx.doRanging(destination_id, device_range, remote_id)
            now_epoch = time.time()  # timestamp the *returned* sample

            # Hot-plug check
            if get_first_pozyx_serial_port() is None:
                print("🔌 Pozyx has been disconnected — stopping.")
                break

            if status == POZYX_SUCCESS:
                dist_m = device_range.distance / 1000.0
                distances.append(dist_m)
                epochs.append(now_epoch)
                print(f"📏 {dist_m:.3f} m")
            else:
                distances.append(np.nan)
                epochs.append(now_epoch)
                print("⚠️ Sensors out of reach")

            time.sleep(sample_dt_s)

    except KeyboardInterrupt:
        print("⏹️ Interrupted by user. Saving...")

    finally:
        # Save RAW CSV
        with open(raw_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp (s)", "Distance (m)"])  # epoch seconds
            for t, d in zip(epochs, distances):
                writer.writerow([f"{t:.6f}", "" if np.isnan(d) else f"{float(d):.6f}"])
        print(f"✅ RAW data written to: {os.path.abspath(raw_csv_path)}")

        # Post-process
        post_path = None
        if do_postprocess:
            try:
                post_path = apply_postprocessing(
                    os.path.abspath(raw_csv_path),
                    calibration_path=calibration_path,
                    apply_low_pass=True,
                    alpha=0.95,
                )
            except Exception as e:
                print(f"❗ Post-processing failed: {e}")

        cleanup_gpio()

    return raw_csv_path, post_path


if __name__ == "__main__":
    # Example device IDs — replace with your own
    run_uwb_logger(
        remote_id=0x6923,
        destination_id=0x6931,
        tx_power_remote=33,
        tx_power_destination=33,
        sample_dt_s=0.01,
        output_dir="output",
        calibration_path="calibration/uwb_calibration.json",
        gpio_pin=18,
        flash_ms=100,
        do_postprocess=True,
    )
