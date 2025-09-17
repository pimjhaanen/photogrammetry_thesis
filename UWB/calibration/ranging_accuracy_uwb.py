"""This code can be used to collect UWB distance samples for 5 seconds,
then plot the measurements with a horizontal line showing the mean and
a shaded band for ±1 standard deviation. This is useful for quickly
assessing the short-term stability (precision) of a static setup."""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from pypozyx import (
    PozyxSerial, DeviceRange, POZYX_SUCCESS, get_first_pozyx_serial_port
)


def uwb_stability_test(
    remote_id: int,
    destination_id: int,
    duration_s: float = 5.0,
    sample_dt_s: float = 0.01,
    serial_port: Optional[str] = None,
) -> Tuple[float, float]:
    """RELEVANT FUNCTION INPUTS:
    - remote_id: Pozyx 16-bit ID initiating the ranging (e.g. 0x6923)
    - destination_id: Pozyx 16-bit ID being ranged (e.g. 0x6931)
    - duration_s: how long to record data in seconds (default: 5.0)
    - sample_dt_s: delay between ranging attempts in seconds (default: 0.01 ≈ 100 Hz)
    - serial_port: optional explicit serial port (None → auto-detect)

    RETURNS:
    - (mean, std): mean and standard deviation of valid distance samples (meters)
    """
    if serial_port is None:
        serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("❌ No Pozyx connected.")
        return np.nan, np.nan

    pozyx = PozyxSerial(serial_port)
    device_range = DeviceRange()

    distances = []
    start = time.time()
    while time.time() - start < duration_s:
        status = pozyx.doRanging(destination_id, device_range, remote_id)
        if status == POZYX_SUCCESS:
            dist_m = device_range.distance / 1000.0
            distances.append(dist_m)
            print(f"📏 {dist_m:.3f} m")
        else:
            distances.append(np.nan)
            print("⚠️ Failed sample (NaN)")
        time.sleep(sample_dt_s)

    distances = np.array(distances, dtype=float)
    valid = distances[~np.isnan(distances)]

    if valid.size == 0:
        print("❌ No valid distances measured.")
        return np.nan, np.nan

    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))

    print(f"\n✅ Stability results over {duration_s:.1f} s:")
    print(f"   Mean distance: {mean_val:.3f} m")
    print(f"   Std deviation: {std_val:.3f} m")

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(distances, "o-", label="Measured samples")
    plt.axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.3f} m")
    plt.fill_between(
        np.arange(len(distances)),
        mean_val - std_val,
        mean_val + std_val,
        color="red",
        alpha=0.2,
        label=f"±1 std = {std_val:.3f} m",
    )
    plt.xlabel("Sample index")
    plt.ylabel("Distance (m)")
    plt.title("UWB Stability Test (5s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_val, std_val


if __name__ == "__main__":
    # Example IDs — replace with your actual device IDs
    uwb_stability_test(remote_id=0x6923, destination_id=0x6931)
