import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# ---- Input pairs (UWB time, Flash time) ----
pairs = [
    ("09:50:34.33", "10:50:36.0"),
    ("09:50:41.30", "10:50:43.4"),
    ("09:50:50.30", "10:50:51.9"),
    ("09:51:03.30", "10:51:04.8"),
    ("09:51:10.31", "10:51:11.9"),
    ("09:51:16.31", "10:51:18.0"),
    ("09:51:23.30", "10:51:25.3"),
]

def parse_time(s: str) -> datetime:
    # Pad to microseconds for datetime
    hhmmss, frac = s.split('.') if '.' in s else (s, '0')
    frac = (frac + "000000")[:6]
    return datetime.strptime(f"{hhmmss}.{frac}", "%H:%M:%S.%f")

# ---- Compute lag per pair ----
lags = []
for uwb_s, flash_s in pairs:
    uwb_t = parse_time(uwb_s)
    flash_t = parse_time(flash_s)
    delta = flash_t - uwb_t
    # Remove any full hours (e.g., 1h timezone offset)
    lag = (delta - timedelta(hours=delta.seconds // 3600)).total_seconds()
    lags.append(lag)

# ---- Summary stats ----
avg_lag = sum(lags) / len(lags)
print(f"Average lag: {avg_lag:.3f} s")

# ---- Plot ----
plt.figure(figsize=(7,4))
plt.plot(range(1, len(lags)+1), lags, marker="o", label="Measured lag")
plt.axhline(avg_lag, color="#1f77b4", linestyle="--", label=f"Average = {avg_lag:.3f} s")
plt.ylim(0,3)
plt.xlabel("Sample index")
plt.ylabel("Lag (s)")
plt.title("Lag between first UWB log and flash")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
