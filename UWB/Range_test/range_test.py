"""This code can be used to perform a simple UWB range/stability test between two Pozyx devices.
It continuously measures distance for a fixed duration while you physically walk away with ONE device
(either the remote or the destination) to reveal where the link starts dropping packets.
The resulting plot shows how far the signal remained reliable before failures (NaNs) appear."""

from pypozyx import (
    PozyxSerial, DeviceRange, POZYX_SUCCESS, get_first_pozyx_serial_port
)
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


def uwb_range_test(
    remote_id: int,
    destination_id: int,
    duration_sec: float = 60.0,
    sample_hz: float = 20.0,
    serial_port: Optional[str] = None,
    plot: bool = True,
) -> Dict[str, np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - remote_id: 16-bit Pozyx ID of the device that INITIATES ranging (e.g. 0x6923)
    - destination_id: 16-bit Pozyx ID of the device being ranged to (e.g. 0x6931)
    - duration_sec: total measurement time window in seconds (e.g. 60.0)
    - sample_hz: target measurement rate in Hz (e.g. 20.0 → ~0.05 s between queries)
    - serial_port: optional explicit serial port (e.g. 'COM6' on Windows or '/dev/ttyACM0');
                   leave None to auto-detect the first connected Pozyx
    - plot: if True, show a time-series plot of the measured distances (NaN = failed reads)

    INSTRUCTIONS:
    - Start with the two devices a short distance apart.
    - Run this script and *walk away* holding ONE device while keeping the other stationary.
    - Packet losses will appear as NaNs; the plot helps you see at what distance/phase the link degrades.
    """
    # --- Connect ---
    if serial_port is None:
        serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("❌ No Pozyx connected. Plug in a device or specify 'serial_port'.")
        return {"distances_m": np.array([]), "valid_mask": np.array([])}

    pozyx = PozyxSerial(serial_port)
    device_range = DeviceRange()

    # --- Collect ---
    dt = 1.0 / sample_hz if sample_hz > 0 else 0.05
    distances: List[float] = []
    timestamps: List[float] = []

    start_time = time.time()
    print(f"🕒 Collecting UWB distances for {duration_sec:.0f}s between "
          f"{hex(remote_id)} ➝ {hex(destination_id)} at ~{sample_hz:.1f} Hz...")
    while (time.time() - start_time) < duration_sec:
        status = pozyx.doRanging(destination_id, device_range, remote_id)
        now = time.time()
        if status == POZYX_SUCCESS:
            dist_m = device_range.distance / 1000.0  # mm → m
            distances.append(dist_m)
            timestamps.append(now)
            print(f"📏 {dist_m:.3f} m")
        else:
            distances.append(np.nan)
            timestamps.append(now)
        time.sleep(dt)

    distances_np = np.array(distances, dtype=float)
    timestamps_np = np.array(timestamps, dtype=float)
    valid_mask = ~np.isnan(distances_np)
    valid_vals = distances_np[valid_mask]

    # --- Stats ---
    if valid_vals.size > 0:
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        valid_ratio = float(valid_vals.size / distances_np.size)
        print(f"\n✅ Stats on valid samples:"
              f"\n  Mean distance: {mean_val:.3f} m"
              f"\n  Std dev     : {std_val:.3f} m"
              f"\n  Valid reads : {valid_vals.size}/{distances_np.size} ({valid_ratio*100:.1f}%)")
    else:
        print("\n⚠️ No valid distance samples recorded (all reads failed).")

    # --- Plot ---
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(distances_np, label="Measured distance (m)")
        plt.title("Pozyx UWB Range Test (NaN = packet loss)")
        plt.xlabel("Sample index")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "distances_m": distances_np,
        "timestamps_s": timestamps_np,
        "valid_mask": valid_mask,
    }


if __name__ == "__main__":
    # Example IDs — replace with your own device IDs if needed.
    uwb_range_test(
        remote_id=0x6923,
        destination_id=0x6931,
        duration_sec=60.0,
        sample_hz=20.0,
        serial_port=None,
        plot=True,
    )
