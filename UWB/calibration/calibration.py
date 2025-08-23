import numpy as np
import matplotlib.pyplot as plt
import json
import os

# === INPUT ===
measured = np.array([4.95, 6.91, 8.93, 10.91, 12.90, 14.97])
actual   = np.array([5, 7, 9, 11, 13, 15])

# === CALIBRATION FIT ===
coeffs = np.polyfit(measured, actual, 1)  # degree 1 polynomial
a, b = coeffs
print(f"Calibration equation: corrected = {a:.4f} * measured + {b:.4f}")

# === SAVE TO FILE ===
calibration_file = "uwb_calibration.json"
calibration_data = {"a": a, "b": b}
with open(calibration_file, 'w') as f:
    json.dump(calibration_data, f, indent=4)
print(f"✅ Calibration saved to: {os.path.abspath(calibration_file)}")

# === APPLY CALIBRATION ===
corrected = a * measured + b

# === ERROR ANALYSIS ===
error_before = measured - actual
error_after = corrected - actual

avg_error_before = np.mean(np.abs(error_before))
avg_error_after = np.mean(np.abs(error_after))

print(f"Average error before calibration: {avg_error_before:.4f} m")
print(f"Average error after calibration:  {avg_error_after:.4f} m")

# === PLOT ERROR ===
plt.plot(actual, error_before, 'o-', label="Error before calibration")
plt.plot(actual, error_after, 'x--', label="Error after calibration")
plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
plt.xlabel("Actual distance (m)")
plt.ylabel("Error (measured - actual) (m)")
plt.title("UWB Measurement Error Before and After Calibration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
