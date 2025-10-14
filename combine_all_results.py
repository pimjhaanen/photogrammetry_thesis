#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# ----------------------------- Readers ----------------------------------------

def read_uwb(path: str) -> pd.DataFrame:
    """
    Read UWB CSV:
      Columns: 'UTC (ISO8601)', 'Timestamp (s)', 'Distance (m)', 'Source'
    Returns df with:
      - uwb_utc  (tz-aware UTC datetime64[ns, UTC])
      - uwb_ts_s (float seconds since UWB start)
      - uwb_distance_m, uwb_source
    """
    df = pd.read_csv(path)
    df["uwb_utc"] = pd.to_datetime(df["UTC (ISO8601)"], utc=True)
    df["uwb_ts_s"] = pd.to_numeric(df["Timestamp (s)"], errors="coerce")
    if "Distance (m)" in df.columns:
        df["uwb_distance_m"] = pd.to_numeric(df["Distance (m)"], errors="coerce")
    else:
        df["uwb_distance_m"] = np.nan
    df["uwb_source"] = df.get("Source", pd.Series(index=df.index, dtype="string"))
    return df[["uwb_utc", "uwb_ts_s", "uwb_distance_m", "uwb_source"]].sort_values("uwb_utc").reset_index(drop=True)


def _parse_kcu_time_of_day_text(s: str) -> Optional[Tuple[int,int,float]]:
    """Parse text like \"16:58:33.1\" (optional leading apostrophe) into (H,M,S.f)."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s.startswith("'"):
        s = s[1:]
    parts = s.split(":")
    try:
        if len(parts) == 3:
            hh = int(parts[0]); mm = int(parts[1]); ss = float(parts[2])
        elif len(parts) == 2:
            hh = 0; mm = int(parts[0]); ss = float(parts[1])
        else:
            return None
    except Exception:
        return None
    return (hh, mm, ss)


def _combine_date_and_hms(anchor_date_utc: datetime, hms: Tuple[int,int,float]) -> datetime:
    """Combine UTC anchor DATE with (H,M,S.f) to a tz-aware UTC datetime."""
    hh, mm, ss = hms
    base = datetime(anchor_date_utc.year, anchor_date_utc.month, anchor_date_utc.day, tzinfo=timezone.utc)
    return base + timedelta(hours=hh, minutes=mm, seconds=ss)


def read_kcu(path: str, uwb_anchor_utc: datetime) -> pd.DataFrame:
    """
    Read KCU CSV created earlier. Build:
      - kcu_utc (tz-aware) by combining UWB anchor DATE with 'time_of_day_utc',
        with day wrap handling to keep monotonicity.
    """
    df = pd.read_csv(path)
    hms = df["time_of_day_utc"].apply(_parse_kcu_time_of_day_text)

    kcu_utc = []
    prev_dt = None
    day_offset = 0
    anchor_date = uwb_anchor_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    for val in hms:
        if val is None:
            kcu_utc.append(pd.NaT)
            continue
        dt = _combine_date_and_hms(anchor_date + timedelta(days=day_offset), val)
        if prev_dt is not None and dt < prev_dt:
            day_offset += 1
            dt = _combine_date_and_hms(anchor_date + timedelta(days=day_offset), val)
        kcu_utc.append(dt)
        prev_dt = dt
    df["kcu_utc"] = pd.to_datetime(kcu_utc, utc=True)

    df["kcu_ts_s"] = pd.to_numeric(df["timestamp"], errors="coerce") if "timestamp" in df.columns else np.nan
    return df.sort_values("kcu_utc").reset_index(drop=True)


def read_photogrammetry(path: str, uwb_start_utc: datetime) -> pd.DataFrame:
    """
    Read photogrammetry CSV (long format: multiple rows per time_s).
      Columns: time_s, frame_idx, marker_id, x, y, z
    Build:
      - photo_utc = uwb_start_utc + time_s (seconds)   [tz-aware]
    """
    df = pd.read_csv(path)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce").astype("Int64")
    df["photo_utc"] = pd.to_datetime(uwb_start_utc, utc=True) + pd.to_timedelta(df["time_s"], unit="s")
    return df.sort_values("photo_utc").reset_index(drop=True)


# ----------------------------- Synchronizer -----------------------------------

def synchronize_on_photo_time(
    photo_df: pd.DataFrame,
    uwb_df: pd.DataFrame,
    kcu_df: pd.DataFrame,
    uwb_tolerance_s: float = 0.25,
    kcu_tolerance_s: float = 0.25,
) -> pd.DataFrame:
    """
    Attach the nearest UWB and KCU rows to each photogrammetry row (by photo_utc),
    within given tolerances. Returns a single long-form dataframe.
    """
    p = photo_df.sort_values("photo_utc").copy()
    u = uwb_df.sort_values("uwb_utc").copy()
    k = kcu_df.sort_values("kcu_utc").copy()

    merged = pd.merge_asof(
        p, u,
        left_on="photo_utc", right_on="uwb_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=uwb_tolerance_s),
    )

    merged = pd.merge_asof(
        merged.sort_values("photo_utc"), k,
        left_on="photo_utc", right_on="kcu_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=kcu_tolerance_s),
        suffixes=("", "_kcu"),
    )
    return merged.reset_index(drop=True)


# ----------------------------- Final shaping ----------------------------------

def finalize_synced_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final view:
      - 'utc' as first column (from photo_utc)
      - 'timestamp' as second column (from time_s)
      - drop redundant time columns:
          ['photo_utc','uwb_utc','uwb_ts_s','uwb_source','time_of_day_utc','kcu_utc','kcu_ts_s']
      - reorder columns: ['utc','timestamp','frame_idx','marker_id','x','y','z','uwb_distance_m', <rest...>]
    """
    out = df.copy()

    # New unified columns
    out["utc"] = out["photo_utc"]
    out["timestamp"] = out["time_s"]

    # Columns to drop if present
    drop_cols = ["photo_utc", "uwb_utc", "uwb_ts_s", "uwb_source", "time_of_day_utc", "kcu_utc", "kcu_ts_s"]
    for c in drop_cols:
        if c in out.columns:
            out = out.drop(columns=c)

    # Reorder: put key columns in front, then the rest
    front = ["utc", "timestamp", "frame_idx", "marker_id", "x", "y", "z", "uwb_distance_m"]
    existing_front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in existing_front]
    out = out[existing_front + rest]

    return out


# ----------------------------- Orchestrator -----------------------------------

def couple_results(
    kcu_csv: str,
    uwb_csv: str,
    photo_csv: str,
    output_csv: Optional[str] = None,
    uwb_tolerance_s: float = 0.25,
    kcu_tolerance_s: float = 0.25,
) -> pd.DataFrame:
    """
    High-level function:
      1) read UWB -> get uwb_start_utc
      2) read KCU (anchor date = UWB date) -> build kcu_utc
      3) read Photogrammetry -> photo_utc = uwb_start_utc + time_s
      4) merge_asof by photo_utc
      5) finalize to single UTC + timestamp view
      6) optionally write CSV
    """
    uwb = read_uwb(uwb_csv)
    if uwb.empty:
        raise ValueError("UWB file appears empty or could not be parsed.")
    uwb_start_utc = uwb["uwb_utc"].iloc[0]

    kcu = read_kcu(kcu_csv, uwb_anchor_utc=uwb_start_utc)
    photo = read_photogrammetry(photo_csv, uwb_start_utc=uwb_start_utc)

    synced = synchronize_on_photo_time(
        photo_df=photo,
        uwb_df=uwb,
        kcu_df=kcu,
        uwb_tolerance_s=uwb_tolerance_s,
        kcu_tolerance_s=kcu_tolerance_s,
    )

    # Single UTC + timestamp, drop redundant columns, reorder
    synced = finalize_synced_columns(synced)

    # Optional write
    if output_csv:
        to_write = synced.copy()

        # Format 'utc' as HH:MM:SS.xx (centiseconds), rounded to 2 decimals
        # and avoid deprecated dtype checker.
        if "utc" in to_write.columns and isinstance(to_write["utc"].dtype, pd.DatetimeTZDtype):
            dt_round = to_write["utc"].dt.tz_convert("UTC").dt.round("10ms")  # 0.01 s precision
            centi = (dt_round.dt.microsecond // 10000).astype(int).astype(str).str.zfill(2)
            to_write["utc"] = dt_round.dt.strftime("%H:%M:%S") + "." + centi

        to_write.to_csv(output_csv, index=False)

    return synced


# --------------------------------- Example ------------------------------------
if __name__ == "__main__":
    # === EDIT THESE PATHS ===
    KCU_PATH   = r"KCU_pitot/output/KCU_output_09_10.csv"
    UWB_PATH   = r"UWB/output/uwb_flight_09_10.csv"
    PHOTO_PATH = r"Photogrammetry/output/09_10_merged_downloop_218.csv"
    OUT_PATH   = r"output/09_10_downloop_218_complete_dataset.csv"

    df_synced = couple_results(
        kcu_csv=KCU_PATH,
        uwb_csv=UWB_PATH,
        photo_csv=PHOTO_PATH,
        output_csv=OUT_PATH,      # set to None to skip writing
        uwb_tolerance_s=0.25,     # ±0.25 s nearest snap to UWB
        kcu_tolerance_s=0.25,     # ±0.25 s nearest snap to KCU
    )
    print(f"[✓] Synced rows: {len(df_synced)}")
    print(f"[→] Saved to: {OUT_PATH}")
