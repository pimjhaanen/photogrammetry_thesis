"""Visualize pitot-tube airspeed fluctuations with a ZERO-PHASE EMA smoother.

- Zero-phase = forward + backward EMA (no lag).
- Tune by:
    * alpha (0..1): higher => stronger smoothing, OR
    * fc_hz (cutoff in Hz): alpha derived from timestamps.

Plot shows raw (mean-centered) vs filtered signal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# -------------------- Reusable smoothing utilities --------------------

def zero_phase_ema_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Forward–backward (zero-phase) EMA on a 1D array.

    Parameters
    ----------
    x : np.ndarray
        1D signal to smooth (e.g., centered airspeed).
    alpha : float
        EMA smoothing factor in [0, 1). Higher -> stronger smoothing.

    Returns
    -------
    np.ndarray
        Smoothed signal, same shape as x, with zero phase lag.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    a = float(np.clip(alpha, 0.0, 0.999999))
    one_minus_a = 1.0 - a

    # Forward pass
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, n):
        y[i] = a * y[i - 1] + one_minus_a * x[i]

    # Backward pass
    z = np.empty_like(y)
    z[-1] = y[-1]
    for i in range(n - 2, -1, -1):
        z[i] = a * z[i + 1] + one_minus_a * y[i]

    return z


def alpha_from_fc(fc_hz: float, dt_s: float) -> float:
    """
    Convert desired cutoff (Hz) to EMA alpha via tau = 1/(2*pi*fc), alpha = exp(-dt/tau).

    Returns alpha in [0, 1).
    """
    if fc_hz <= 0.0:
        raise ValueError("fc_hz must be > 0.")
    if dt_s <= 0.0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be a positive finite value.")
    tau = 1.0 / (2.0 * np.pi * fc_hz)
    return float(np.exp(-dt_s / tau))


# -------------------- Plotting script (EMA version) --------------------

def plot_airspeed_fluctuations(
    csv_path: str,
    alpha: float = 0.95,
    fc_hz: Optional[float] = None,
    timestamp_col: str = "recording timestamp",
    airspeed_col: str = "Airspeed in m/s",
    center: bool = True,
) -> None:
    """Inputs
    - csv_path: CSV containing at least:
        * 'recording timestamp' (x-axis)
        * 'Airspeed in m/s'    (raw airspeed values)
    - alpha: EMA smoothing factor (0..1). Ignored if fc_hz is provided.
    - fc_hz: cutoff in Hz; if provided, alpha is derived from timestamps.
    - center: subtract mean before filtering (recommended for fluctuation plots).
    """
    # --- Load ---
    data = pd.read_csv(csv_path)

    if timestamp_col not in data.columns or airspeed_col not in data.columns:
        raise ValueError(f"CSV must contain '{timestamp_col}' and '{airspeed_col}' columns.")

    t = pd.to_numeric(data[timestamp_col], errors="coerce").to_numpy()
    v = pd.to_numeric(data[airspeed_col], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]

    if t.size < 3:
        raise ValueError("Not enough valid samples after cleaning.")

    # --- Center (optional) ---
    v_mean = float(np.mean(v))
    v_plot = v - v_mean if center else v.copy()

    # --- Choose alpha ---
    if fc_hz is not None:
        dt = float(np.median(np.diff(t)))
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Non-positive or invalid time step detected.")
        alpha_eff = float(np.clip(alpha_from_fc(fc_hz, dt), 0.0, 0.999999))
        label_suffix = f"(fc={fc_hz:.3g} Hz)"
    else:
        alpha_eff = float(np.clip(alpha, 0.0, 0.999999))
        label_suffix = f"(alpha={alpha_eff:.3f})"

    # --- Filter (zero-phase EMA) ---
    v_filt = zero_phase_ema_filter(v_plot, alpha_eff)

    # --- Plot ---
    plt.figure(figsize=(9, 4.5))
    plt.plot(t, v_plot, label="Measured $V_\\infty - \\overline{V_\\infty}$")
    plt.plot(t, v_filt, label=f"Filtered zero-phase EMA {label_suffix}")
    plt.axhline(0 if center else v_mean, color="k", linestyle="--", linewidth=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel(r"$V_\infty - \overline{V_\infty}$ (m/s)" if center else r"$V_\infty$ (m/s)")
    plt.title("Airspeed Fluctuations (zero-phase EMA smoothing)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example 1: tune by alpha (no lag, forward–backward EMA)
    plot_airspeed_fluctuations(
        "Calibration/data_inside_thursday/recording_IV10A0.csv",
        alpha=0.9,
        fc_hz=None,
        center=True,
    )

    # Example 2: tune by physical cutoff (Hz); alpha computed from timestamps
    # plot_airspeed_fluctuations(
    #     "Calibration/data_inside_thursday/recording_IV10A0.csv",
    #     fc_hz=1.0,  # try 0.7–1.5 Hz for ~11 Hz sampling
    #     center=True,
    # )
