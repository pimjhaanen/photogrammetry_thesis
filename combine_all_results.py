#!/usr/bin/env python3
"""
Robust synchronization of Photogrammetry (frame-based), UWB (UTC), and KCU pitot (time-of-day).
- Strict nearest-match in UTC with tight tolerance (default 0.12 s)
- Optional constant KCU UTC offset to correct inter-device clock skew
- Diagnostics printed to help verify alignment quality
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, Tuple, List

# ------------------------- Readers -------------------------

def read_uwb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["uwb_utc"] = pd.to_datetime(df["UTC (ISO8601)"], utc=True, errors="coerce")
    df["uwb_ts_s"] = pd.to_numeric(df["Timestamp (s)"], errors="coerce")
    df["uwb_distance_m"] = pd.to_numeric(df.get("Distance (m)"), errors="coerce")
    df = df.dropna(subset=["uwb_utc"]).sort_values("uwb_utc").reset_index(drop=True)
    return df[["uwb_utc", "uwb_ts_s", "uwb_distance_m"]]


def _parse_tod(t: str) -> Optional[timedelta]:
    """
    Parse 'HH:MM:SS(.fff)' or "'HH:MM:SS(.fff)" -> timedelta since midnight.
    Returns None if malformed.
    """
    if pd.isna(t):
        return None
    s = str(t).strip().lstrip("'")
    parts = s.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0]); mm = int(parts[1]); ss = float(parts[2].replace(",", "."))  # tolerate comma decimals
    except Exception:
        return None
    return timedelta(hours=hh, minutes=mm, seconds=ss)


def read_kcu(path: str, uwb_start_utc: pd.Timestamp) -> pd.DataFrame:
    """
    Build absolute UTC for KCU by combining the UWB date with KCU time_of_day_utc,
    then enforce monotonicity across midnight if needed.
    """
    df = pd.read_csv(path, low_memory=False)
    if "time_of_day_utc" not in df.columns:
        raise ValueError("KCU file missing 'time_of_day_utc'")

    base_date = uwb_start_utc.floor("D")
    tod = df["time_of_day_utc"].apply(_parse_tod)
    df["kcu_utc"] = [pd.NaT if v is None else (base_date + v) for v in tod]
    df["kcu_utc"] = pd.to_datetime(df["kcu_utc"], utc=True, errors="coerce")

    # Enforce monotonicity across midnight (multiple wraps handled)
    df = df.sort_values("kcu_utc").reset_index(drop=True)
    day_add = 0
    for i in range(1, len(df)):
        if df.loc[i, "kcu_utc"] < df.loc[i - 1, "kcu_utc"]:
            day_add += 1
        if day_add:
            df.loc[i, "kcu_utc"] = df.loc[i, "kcu_utc"] + timedelta(days=day_add)

    return df


def read_photogrammetry(path: str, uwb_start_utc: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce").astype("Int64")
    df["photo_utc"] = uwb_start_utc + pd.to_timedelta(df["time_s"], unit="s")
    df = df.dropna(subset=["photo_utc"]).sort_values("photo_utc").reset_index(drop=True)
    return df

# ------------------------- Core Sync -------------------------

def _strict_nearest_asof(left: pd.DataFrame,
                         right: pd.DataFrame,
                         left_on: str,
                         right_on: str,
                         tol_s: float,
                         suffixes=("", "_r")) -> pd.DataFrame:
    """
    merge_asof + hard cutoff: if |Δt| > tol, drop the match (set to NaN).
    """
    merged = pd.merge_asof(
        left.sort_values(left_on),
        right.sort_values(right_on),
        left_on=left_on, right_on=right_on,
        direction="nearest", tolerance=pd.Timedelta(seconds=tol_s),
        suffixes=suffixes
    )
    # Compute abs time error; reject outside tolerance explicitly (in case some slipped through)
    dt = (merged[left_on] - merged[right_on]).abs()
    bad = dt > pd.Timedelta(seconds=tol_s)
    if bad.any():
        right_cols = [c for c in right.columns if c != right_on]
        merged.loc[bad, right_cols] = np.nan
        merged.loc[bad, right_on] = pd.NaT
    return merged


def synchronize(photo_df, uwb_df, kcu_df, uwb_tol=0.5, kcu_tol=1.0):
    """
    Synchronize photogrammetry frames with UWB (for UTC) and then
    attach the nearest *previous* KCU telemetry sample.
    """
    # --- step 1: make sure all UTCs are sorted ---
    p = photo_df.sort_values("photo_utc").copy()
    u = uwb_df.sort_values("uwb_utc").copy()
    k = kcu_df.sort_values("kcu_utc").copy()

    # --- step 2: photo ↔ UWB merge (nearest is fine; same timeline) ---
    merged = pd.merge_asof(
        p, u,
        left_on="photo_utc", right_on="uwb_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=uwb_tol)
    )

    # --- step 3: attach KCU but only backward in time ---
    merged = pd.merge_asof(
        merged.sort_values("photo_utc"), k,
        left_on="photo_utc", right_on="kcu_utc",
        direction="backward",             # ✅ critical fix
        tolerance=pd.Timedelta(seconds=kcu_tol),
        suffixes=("", "_kcu")
    )

    lost = merged["kcu_utc"].isna().sum()
    print(f"[✓] Synced {len(merged)} frames. {lost} frames had no KCU sample within {kcu_tol}s.")
    return merged.reset_index(drop=True)

# ------------------------- Finalize -------------------------

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["utc"] = out["photo_utc"]
    out["timestamp"] = out["time_s"]

    # Order: core columns first; keep all KCU/UWB telemetry afterwards
    core = ["utc", "timestamp", "frame_idx", "marker_id", "x", "y", "z", "uwb_distance_m"]
    rest = [c for c in out.columns if c not in core]
    out = out[core + rest]

    # Drop time helper columns if you prefer a clean table:
    drop_cols = ["photo_utc", "uwb_utc", "uwb_ts_s", "kcu_utc"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out

# ------------------------- Offset helper (optional) -------------------------

def estimate_best_kcu_offset_seconds(photo_df: pd.DataFrame,
                                     kcu_df: pd.DataFrame,
                                     test_range_s: Tuple[float, float] = (-2.0, 2.0),
                                     step_s: float = 0.1) -> float:
    """
    Grid-search a constant offset that minimizes median |Δt(photo_utc - kcu_utc)|.
    Use when you suspect a constant clock skew between systems.
    """
    offsets = np.arange(test_range_s[0], test_range_s[1] + 1e-9, step_s)
    meds = []
    for off in offsets:
        k = kcu_df.copy()
        k["kcu_utc"] = k["kcu_utc"] + pd.to_timedelta(off, unit="s")
        tmp = pd.merge_asof(photo_df.sort_values("photo_utc"),
                            k.sort_values("kcu_utc"),
                            left_on="photo_utc", right_on="kcu_utc",
                            direction="nearest")
        dt = (tmp["photo_utc"] - tmp["kcu_utc"]).abs().dropna()
        med = (dt / pd.Timedelta(milliseconds=1)).median() if not dt.empty else np.inf
        meds.append(med)
    best_i = int(np.nanargmin(meds))
    print(f"[Offset search] Best offset ≈ {offsets[best_i]:+.2f}s (median |Δt|={meds[best_i]:.1f} ms)")
    return float(offsets[best_i])

# ------------------------- Orchestrator -------------------------

def couple_results(kcu_csv, uwb_csv, photo_csv, output_csv=None, uwb_tol=0.5, kcu_tol=1.0):
    """Main wrapper for reading and synchronizing."""
    uwb = read_uwb(uwb_csv)
    uwb_start_utc = uwb["uwb_utc"].iloc[0]

    photo = read_photogrammetry(photo_csv, uwb_start_utc)
    kcu = read_kcu(kcu_csv, uwb_start_utc)

    synced = synchronize(photo, uwb, kcu, uwb_tol, kcu_tol)
    out = finalize(synced)

    if output_csv:
        out.to_csv(output_csv, index=False)
        print(f"[→] Wrote synchronized dataset → {output_csv}")

    return out

# ------------------------- CLI -------------------------

if __name__ == "__main__":
    KCU_PATH   = r"KCU_pitot/output/KCU_output_09_10.csv"
    UWB_PATH   = r"UWB/output/uwb_flight_09_10.csv"
    PHOTO_PATH = r"Photogrammetry/output/left_turn_frame_7182.csv"
    OUT_PATH   = r"output/left_turn_frame_7182_complete_dataset.csv"

    # Start strict (120 ms), no offset; if still wrong, run the offset search helper (see above)
    df = couple_results(
        kcu_csv=KCU_PATH,
        uwb_csv=UWB_PATH,
        photo_csv=PHOTO_PATH,
        output_csv=OUT_PATH,
        uwb_tol=0.5,  # instead of uwb_tol_s
        kcu_tol=1.0  # instead of kcu_tol_s
    )

    print(df.head(8))
