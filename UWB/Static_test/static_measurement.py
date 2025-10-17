"""
Quick Pozyx UWB average distance (3s) with linear calibration.

Calibrated distance (m) = a * raw_distance_m + b
"""

from pypozyx import (
    PozyxSerial, DeviceRange, POZYX_SUCCESS, get_first_pozyx_serial_port
)
import time
import json
from typing import Optional, Dict
import numpy as np


def load_linear_calibration(calibration_json_path: str) -> Dict[str, float]:
    """Load {a, b} from JSON. Falls back to a=1.0, b=0.0 if file/keys missing."""
    a, b = 1.0, 0.0
    try:
        with open(calibration_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        a = float(data.get("a", 1.0))
        b = float(data.get("b", 0.0))
    except Exception as e:
        print(f"⚠️ Calibration load issue ({e}). Using a=1.0, b=0.0.")
    return {"a": a, "b": b}


def uwb_quick_average(
    remote_id: int,
    destination_id: int,
    calibration_json_path: str,
    duration_sec: float = 3.0,
    sample_hz: float = 20.0,
    serial_port: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Measures Pozyx UWB distance for ~duration_sec, applies linear calibration,
    and prints the average calibrated distance. Returns summary stats.
    """
    # --- Serial port selection ---
    if serial_port is None:
        serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("❌ No Pozyx connected. Plug in a device or specify 'serial_port'.")
        return None

    cal = load_linear_calibration(calibration_json_path)
    a, b = cal["a"], cal["b"]

    pozyx = PozyxSerial(serial_port)
    device_range = DeviceRange()

    dt = 1.0 / sample_hz if sample_hz > 0 else 0.05
    raw_meters = []

    start = time.time()
    end_time = start + duration_sec

    print(f"🕒 Measuring for {duration_sec:.1f}s at ~{sample_hz:.1f} Hz "
          f"between {hex(remote_id)} ➝ {hex(destination_id)}")
    while time.time() < end_time:
        status = pozyx.doRanging(destination_id, device_range, remote_id)
        if status == POZYX_SUCCESS:
            # Pozyx returns distance in millimeters
            raw_m = device_range.distance / 1000.0
            raw_meters.append(raw_m)
        # else: consider it a dropped sample; we just skip it from averaging
        time.sleep(dt)

    raw_np = np.array(raw_meters, dtype=float)
    n_total = int(np.ceil(duration_sec * max(sample_hz, 1)))
    n_valid = raw_np.size
    valid_ratio = (n_valid / n_total) if n_total > 0 else 0.0

    if n_valid == 0:
        print("⚠️ No valid samples recorded in the time window.")
        return {
            "mean_cal_m": float("nan"),
            "std_cal_m": float("nan"),
            "n_valid": 0,
            "n_total": n_total,
            "valid_ratio": valid_ratio,
            "a": a,
            "b": b,
        }

    # Apply linear calibration: calibrated = a * raw + b (in meters)
    cal_np = a * raw_np + b

    mean_cal = float(np.mean(cal_np))
    std_cal = float(np.std(cal_np))

    print(f"✅ Average calibrated distance over {n_valid} valid samples "
          f"({valid_ratio*100:.0f}% valid): {mean_cal:.3f} m (±{std_cal:.3f} m, 1σ)")
    print(f"   Using calibration: a={a:.6f}, b={b:.6f} (meters)")

    return {
        "mean_cal_m": mean_cal,
        "std_cal_m": std_cal,
        "n_valid": n_valid,
        "n_total": n_total,
        "valid_ratio": valid_ratio,
        "a": a,
        "b": b,
    }


if __name__ == "__main__":
    # Example usage — replace IDs and calibration path with your own:
    uwb_quick_average(
        remote_id=0x6923,
        destination_id=0x6931,
        calibration_json_path="../calibration/uwb_calibration.json",  # <-- your file with {"a":..., "b":...}
        duration_sec=3.0,
        sample_hz=20.0,
        serial_port=None,  # e.g., "COM6" or "/dev/ttyACM0"
    )
