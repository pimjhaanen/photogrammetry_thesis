#!/usr/bin/env python3
"""
Processes a KCU output CSV and writes a cleaned CSV with derived and calibrated fields.
Steps:
1) read input CSV,
2) compute timestamp = time - time[0],
3) normalize & UTC-shift time_of_day (Ireland -> default −2 h),
4) copy selected KCU signals,
5) calibrate kite_measured_va from a calibration CSV,
6) apply zero-phase exponential LPF (alpha=0.95 default) to calibrated pitot (no lag),
7) scale steering/depower by /100 and round(4),
8) write to output CSV.

User inputs only in __main__: input file, pitot calibration CSV (optional), output file.
"""

import argparse
import sys
from typing import Tuple, Union, Optional, List
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def epoch_to_hms_str(series: pd.Series, tz_offset_hours: int = 0, decimals: int = 1) -> pd.Series:
    """
    Convert UNIX epoch seconds in `series` to 'HH:MM:SS.s' strings.
    - tz_offset_hours: +2 for Ireland local (summer), 0 for UTC
    - decimals: number of fractional digits on seconds (default 1)
    Returns text strings (not timedeltas), safe for Excel display.
    """
    fmt = f"{{:02d}}:{{:02d}}:{{:0{2 + (1 + decimals) if decimals>0 else 2}.{decimals}f}}"
    out = []
    vals = pd.to_numeric(series, errors="coerce")
    for v in vals:
        if pd.isna(v):
            out.append("")
            continue
        t = datetime.utcfromtimestamp(float(v)) + timedelta(hours=tz_offset_hours)
        ss = t.second + t.microsecond / 1e6
        if decimals == 0:
            hms = f"{t.hour:02d}:{t.minute:02d}:{int(round(ss)):02d}"
        else:
            hms = fmt.format(t.hour, t.minute, ss)
        out.append(hms)
    return pd.Series(out, index=series.index, dtype="string")


# ------------------------------ Time utilities --------------------------------

def parse_kcu_time_of_day(raw: Union[str, float, int]) -> Optional[str]:
    """Normalize KCU 'time_of_day' like '16.58:33.1' -> 'HH:MM:SS(.fff)'."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if "." in s and ":" in s:
        dot_idx = s.find(".")
        colon_idx = s.find(":")
        if 0 < dot_idx < colon_idx:
            s = s.replace(".", ":", 1)
    try:
        parts = s.split(":")
        if len(parts) == 1:
            hh, mm, ss = int(parts[0]), 0, 0.0
        elif len(parts) == 2:
            hh, mm, ss = int(parts[0]), int(parts[1]), 0.0
        else:
            hh, mm, ss = int(parts[0]), int(parts[1]), float(parts[2])
        base = datetime(2000, 1, 1, hh, mm, int(ss))
        frac = ss - int(ss)
        base += timedelta(seconds=frac)
        out = base.strftime("%H:%M:%S")
        if frac > 0:
            out += f"{frac:.3f}"[1:]
        return out
    except Exception:
        return s


def shift_hms_utc(hms: Optional[str], hours_delta: int = -2) -> Optional[str]:
    """Shift 'HH:MM:SS(.f)' by hours_delta, wrapping midnight."""
    if hms is None or pd.isna(hms):
        return None
    try:
        main, frac = hms, ""
        if "." in hms:
            main, frac = hms.split(".", 1)
            frac = "." + frac
        t = datetime.strptime(main, "%H:%M:%S") + timedelta(hours=hours_delta)
        return t.strftime("%H:%M:%S") + frac
    except Exception:
        return hms


# ------------------------------ Calibration -----------------------------------

def load_pitot_calibration(cal_path: str):
    """
    Load pitot calibration from a JSON file with fields:
        { "a": <float>, "b": <float> }

    Returns
    -------
    tuple (a, b)
        Linear calibration coefficients for y = a*x + b
    """
    try:
        with open(cal_path, "r") as f:
            data = json.load(f)
        a = float(data["a"])
        b = float(data["b"])
        return a, b
    except Exception as e:
        raise ValueError(f"Invalid calibration JSON '{cal_path}': {e}")


def apply_pitot_calibration(series: pd.Series, cal_spec):
    """
    Apply linear calibration y = a*x + b using coefficients from JSON file.
    """
    if cal_spec is None:
        return pd.to_numeric(series, errors="coerce")

    a, b = cal_spec
    x = pd.to_numeric(series, errors="coerce")
    return a * x + b


# ------------------------------ Filtering -------------------------------------

def exponential_lpf(series: pd.Series, alpha: float = 0.95) -> pd.Series:
    """
    Causal EMA (kept for reference):
      y[t] = alpha*y[t-1] + (1-alpha)*x[t]
    """
    x = pd.to_numeric(series, errors="coerce").astype(float).values
    y = np.empty_like(x)
    y[:] = np.nan
    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0:
        return pd.Series(y, index=series.index, dtype=float)
    y[valid[0]] = x[valid[0]]
    for i in range(valid[0] + 1, len(x)):
        xi = x[i]
        if np.isnan(xi):
            y[i] = y[i - 1]
        else:
            y[i] = alpha * y[i - 1] + (1.0 - alpha) * xi
    return pd.Series(y, index=series.index, dtype=float)

def _ema_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    One directional EMA with NaN handling; starts at first finite value.
    """
    y = np.empty_like(x, dtype=float)
    y[:] = np.nan
    idx = np.where(np.isfinite(x))[0]
    if len(idx) == 0:
        return y
    i0 = idx[0]
    y[i0] = x[i0]
    for i in range(i0 + 1, len(x)):
        xi = x[i]
        y[i] = (alpha * y[i-1] + (1.0 - alpha) * xi) if np.isfinite(xi) else y[i-1]
    return y

def zero_phase_exponential_lpf(series: pd.Series, alpha: float = 0.95) -> pd.Series:
    """
    Zero-phase (acausal) EMA by forward-backward filtering (no lag).
    Steps:
      1) fwd[t] = alpha*fwd[t-1] + (1-alpha)*x[t]
      2) bwd_rev[k] = alpha*bwd_rev[k-1] + (1-alpha)*fwd_rev[k]
         where fwd_rev is fwd reversed in time; final output = reverse(bwd_rev).

    Note: magnitude response is effectively squared (heavier smoothing than one-pass EMA),
    so you may want a slightly smaller alpha than your causal setting.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return pd.Series([], index=series.index, dtype=float)
    fwd = _ema_pass(x, alpha)
    bwd = _ema_pass(fwd[::-1], alpha)[::-1]
    return pd.Series(bwd, index=series.index, dtype=float)


# ------------------------------ Core transform --------------------------------

REQUIRED_PASSTHROUGH_COLS: List[str] = [
    "kite_0_latitude",
    "kite_0_longitude",
    "kite_0_altitude",
    "kite_0_roll",
    "kite_0_pitch",
    "kite_0_yaw",
    "ground_tether_reelout_speed",
    "ground_tether_length",
    "ground_tether_force",
    "airspeed_angle_of_attack",
    "airspeed_sideslip_angle",
]


def compute_timestamp(df: pd.DataFrame, time_col: str = "time") -> pd.Series:
    """Compute relative time: timestamp = time - time.iloc[0], rounded to 0.1 s."""
    if time_col not in df.columns:
        return pd.Series(dtype=float)
    t = pd.to_numeric(df[time_col], errors="coerce")
    if t.empty or t.isna().all():
        return pd.Series(dtype=float)
    return np.round(t - t.iloc[0], 1)


def build_output(df: pd.DataFrame,
                 utc_shift_hours: int,
                 cal_spec: Optional[Tuple[str, Tuple[np.ndarray, np.ndarray]]] = None,
                 pitot_lpf_alpha: float = 0.95,
                 pitot_zero_phase: bool = True) -> pd.DataFrame:
    """
    Build output dataframe:
      - timestamp from 'time'
      - time_of_day_utc (recomputed from epoch 'time')
      - passthrough telemetry
      - kite_measured_va: calibrate -> zero-phase LPF(α)  [no lag]
      - steering/depower: /100, round(4)
    """
    out = pd.DataFrame(index=df.index)

    # 1) timestamp
    out["timestamp"] = compute_timestamp(df, time_col="time")

    # 2) time_of_day_utc (from epoch seconds)
    if "time" in df.columns:
        out["time_of_day_utc"] = epoch_to_hms_str(df["time"], tz_offset_hours=0, decimals=1)

    # 3) passthrough telemetry
    for col in REQUIRED_PASSTHROUGH_COLS:
        if col in df.columns:
            out[col] = df[col]

    # 4) pitot: calibrate then LPF (zero-phase to remove lag)
    if "kite_measured_va" in df.columns:
        va_cal = apply_pitot_calibration(df["kite_measured_va"], cal_spec)
        if pitot_zero_phase:
            va_filt = zero_phase_exponential_lpf(va_cal, alpha=pitot_lpf_alpha)
        else:
            va_filt = exponential_lpf(va_cal, alpha=pitot_lpf_alpha)
        out["kite_measured_va"] = va_filt

    # 5) steering/depower scaling
    for col in ("kite_actual_steering", "kite_actual_depower"):
        if col in df.columns:
            scaled = pd.to_numeric(df[col], errors="coerce") / 100.0
            out[col] = scaled.round(4)

    # drop empty columns
    out = out.loc[:, ~out.isna().all(axis=0)]
    return out.reset_index(drop=True)


def write_output_csv(df_out: pd.DataFrame, out_path: str) -> None:
    """Write dataframe to CSV (no index). Ensures time_of_day_utc stays as text."""
    if "time_of_day_utc" in df_out.columns:
        df_out["time_of_day_utc"] = df_out["time_of_day_utc"].map(lambda x: f"{x}")
    df_out.to_csv(out_path, index=False)



# ----------------------------------- CLI --------------------------------------

def main():
    """Main entry point: read input, apply calibration & filtering, and write output."""

    # === USER CONFIGURATION (edit these paths only) ===
    INPUT_FILE = r"input/2025-10-09_16-58-33_ProtoLogger.csv"
    CALIBRATION_FILE = r"Pitot/Calibration/pitot_calibration.json"
    OUTPUT_FILE = r"output/KCU_output_09_10_no_lpf.csv"
    UTC_SHIFT_HOURS = -2                               # Ireland time → UTC
    PITOT_LPF_ALPHA = 0                            # EMA smoothing factor
    PITOT_ZERO_PHASE = True                            # Use forward-backward (no lag)

    # 1) Read input CSV
    try:
        df_in = pd.read_csv(INPUT_FILE)
        print(f"[→] Loaded input: {INPUT_FILE}  ({len(df_in)} rows)")
    except Exception as e:
        print(f"[✗] Error reading input CSV '{INPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Load calibration (JSON)
    cal_spec = None
    if CALIBRATION_FILE:
        try:
            cal_spec = load_pitot_calibration(CALIBRATION_FILE)
            print(f"[→] Loaded JSON calibration: {CALIBRATION_FILE} (a={cal_spec[0]:.6f}, b={cal_spec[1]:.6f})")
        except Exception as e:
            print(f"[!] Warning: could not load calibration '{CALIBRATION_FILE}' ({e}); continuing uncalibrated.",
                  file=sys.stderr)
            cal_spec = None

    # 3) Build output dataframe
    df_out = build_output(
        df_in,
        utc_shift_hours=UTC_SHIFT_HOURS,
        cal_spec=cal_spec,
        pitot_lpf_alpha=PITOT_LPF_ALPHA,
        pitot_zero_phase=PITOT_ZERO_PHASE
    )

    for col in ("time_of_day_utc", "time_of_day_local"):
        if col in df_out.columns:
            df_out[col] = df_out[col].where(df_out[col].isna(), df_out[col].map(lambda s: f"'{s}"))

    # 4) Write to CSV
    try:
        write_output_csv(df_out, OUTPUT_FILE)
        print(f"[✓] Wrote output to: {OUTPUT_FILE}")
        print(f"[→] Columns: {', '.join(df_out.columns)}")
    except Exception as e:
        print(f"[✗] Error writing output CSV '{OUTPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------- script main --------------------------------
if __name__ == "__main__":
    main()
