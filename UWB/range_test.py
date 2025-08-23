from pypozyx import (PozyxSerial, PozyxConstants, version,
                     SingleRegister, DeviceRange, POZYX_SUCCESS, get_first_pozyx_serial_port)
import time
import numpy as np
import matplotlib.pyplot as plt

# === SETUP ===
serial_port = get_first_pozyx_serial_port()
if serial_port is None:
    print("No Pozyx connected.")
    quit()

remote_id = 0x6923          # Anchor initiating the ranging
destination_id = 0x6931     # Anchor being ranged to
pozyx = PozyxSerial(serial_port)
device_range = DeviceRange()

# === COLLECT DATA FOR 60 SECONDS ===
distances = []
start_time = time.time()
print(f"🕒 Collecting distances for 60 seconds between {hex(remote_id)} ➝ {hex(destination_id)}...")
while time.time() - start_time < 60:
    status = pozyx.doRanging(destination_id, device_range, remote_id)
    if status == POZYX_SUCCESS:
        dist_m = device_range.distance / 1000  # convert mm to m
        distances.append(dist_m)
        print(f"📏 Distance: {round(dist_m, 3)} m")
    else:
        distances.append(np.nan)
    time.sleep(0.05)  # ~20 Hz max rate

# === CALCULATE STATISTICS ===
valid_distances = np.array(distances)
valid_distances = valid_distances[~np.isnan(valid_distances)]
mean_val = np.mean(valid_distances)
std_val = np.std(valid_distances)

# === PLOT ===
plt.figure(figsize=(10, 4))
plt.plot(distances, label="Measured Distance")
plt.title("UWB Range Test")
plt.xlabel("Sample Index")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
