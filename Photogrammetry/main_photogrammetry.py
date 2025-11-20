"""
Runner for kite stereo photogrammetry with per-frame rotation corrections.

- Reads stereo calib (.pkl).
- Finds matched (left_idx, right_idx) via match_videos (sync CSV).
- Reads CSV with columns: frame_left, delta_yaw_deg, delta_pitch_deg, delta_roll_deg
  (header names are flexible; common variants like 'frame', 'yaw_deg', 'delta_yaw' work).
- Interpolates corrections for EVERY matched left frame in range [min..max] of CSV.
- Builds R_corr from yaw(Z)–pitch(Y)–roll(X) (ZYX order), degrees -> radians.
- Applies R_current = R_corr @ R_base if cfg.premultiply_rotation_correction=True
  (recommended when your deltas are defined in LEFT camera/world frame).
  Set False for post-multiplying if your deltas are in RIGHT cam frame.
- Passes R_current to process_stereo_pair(...) each frame.
- Optionally prints epipolar stats every N frames.

Outputs:
- output/cross3d_<video>.csv   (time_s, frame_idx, marker_id, x,y,z)
- output/epi_stats_<video>.csv (time_s, frame_idx, n_pairs, mean_vdisp, p95_vdisp, mean_2d_dist)
"""

import os, csv, pickle
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple, Optional

# --- project imports you already have ---
from Photogrammetry.Synchronisation.synchronisation_utils import match_videos, find_continuation_files
from Photogrammetry.stereo_photogrammetry_utils import (
    process_stereo_pair, StereoConfig, TrackerState
)
from Photogrammetry.kite_shape_reconstruction_utils import separate_LE_and_struts

# ------------------------------- I/O helpers --------------------------------

def load_calib(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        calib = pickle.load(f)
    for k in ["camera_matrix_1","dist_coeffs_1","camera_matrix_2","dist_coeffs_2","R","T"]:
        if k not in calib:
            raise KeyError(f"Missing '{k}' in calibration file.")
    return calib

def read_matched(sync_csv_path: str) -> List[Tuple[float,int,int]]:
    out = []
    with open(sync_csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        has_time = False
        if header:
            cols = [c.strip().lower() for c in header]
            has_time = "time_s" in cols
        if has_time:
            for row in r:
                if len(row) >= 3:
                    out.append((float(row[0]), int(row[1]), int(row[2])))
        else:
            for row in r:
                if len(row) >= 2:
                    out.append((float("nan"), int(row[0]), int(row[1])))
    return out

# -------------------------- rotation utils (ZYX) ----------------------------

def euler_ypr_leftframe(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build dR to apply in the LEFT camera/world frame with the same convention
    as correction_factor_calculator:
      - yaw about +Y
      - pitch about +X
      - roll about +Z
    Order: yaw -> pitch -> roll   so dR = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """
    y, x, z = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

    cy, sy = np.cos(y), np.sin(y)   # yaw about +Y
    cx, sx = np.cos(x), np.sin(x)   # pitch about +X
    cz, sz = np.cos(z), np.sin(z)   # roll about +Z

    Ry = np.array([[ cy, 0,  sy],
                   [  0, 1,   0],
                   [-sy, 0,  cy]], dtype=float)
    Rx = np.array([[ 1,  0,   0],
                   [ 0, cx, -sx],
                   [ 0, sx,  cx]], dtype=float)
    Rz = np.array([[ cz, -sz, 0],
                   [ sz,  cz, 0],
                   [  0,   0, 1]], dtype=float)

    return Ry @ Rx @ Rz


def _read_corrections_flex(path_or_df) -> pd.DataFrame:
    """
    Read a corrections table from CSV/Excel or pass-through a DataFrame.
    Flexible header mapping. Produces columns:
        ['frame_left','delta_yaw_deg','delta_pitch_deg','delta_roll_deg'] (floats)
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        p = str(path_or_df)
        if p.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

    # Build lowercase -> original name map
    lc_to_orig = {str(c).strip().lower(): c for c in df.columns}

    def pick(candidates: list) -> str:
        for key in candidates:
            if key in lc_to_orig:
                return lc_to_orig[key]
        raise ValueError(
            "Corrections CSV is missing any of: "
            + ", ".join(candidates)
            + f". Available columns: {list(df.columns)}"
        )

    # Flexible candidates
    frame_keys = ["frame_left", "left_frame", "frame", "frame_idx", "idx_left", "l_frame"]
    yaw_keys   = ["delta_yaw_deg", "dyaw_deg", "yaw_deg", "delta_yaw", "yaw"]
    pitch_keys = ["delta_pitch_deg", "dpitch_deg", "pitch_deg", "delta_pitch", "pitch"]
    roll_keys  = ["delta_roll_deg", "droll_deg", "roll_deg", "delta_roll", "roll"]

    # Pick actual column names present
    c_frame = pick(frame_keys)
    c_yaw   = pick(yaw_keys)
    c_pitch = pick(pitch_keys)
    c_roll  = pick(roll_keys)

    # Rename to canonical names
    df = df.rename(columns={
        c_frame: "frame_left",
        c_yaw:   "delta_yaw_deg",
        c_pitch: "delta_pitch_deg",
        c_roll:  "delta_roll_deg",
    })

    # Coerce to numeric and clean
    for k in ["frame_left","delta_yaw_deg","delta_pitch_deg","delta_roll_deg"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    df = df.dropna(subset=["frame_left"]).sort_values("frame_left")
    # If duplicates on the same frame exist, keep the last
    df = df.drop_duplicates(subset=["frame_left"], keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError("Corrections table became empty after cleaning.")
    return df

def build_interp_functions(df_corr: pd.DataFrame):
    """
    Returns callables yaw(f), pitch(f), roll(f) that linearly interpolate
    angles (deg) for arbitrary left-frame index f. Handles single-row CSVs too.
    Also returns the min and max frame indices present in the corrections.
    """
    f   = df_corr["frame_left"].to_numpy(dtype=float)
    yaw = df_corr["delta_yaw_deg"].to_numpy(dtype=float)
    pit = df_corr["delta_pitch_deg"].to_numpy(dtype=float)
    rol = df_corr["delta_roll_deg"].to_numpy(dtype=float)

    order = np.argsort(f)
    f, yaw, pit, rol = f[order], yaw[order], pit[order], rol[order]

    # If only one row, return constant functions
    if len(f) == 1:
        fyaw = lambda idx, v=float(yaw[0]): v
        fpit = lambda idx, v=float(pit[0]): v
        frol = lambda idx, v=float(rol[0]): v
        return fyaw, fpit, frol, float(f[0]), float(f[0])

    def _interp(ys):
        return lambda idx, xs=f, ys=ys: float(np.interp(float(idx), xs, ys))

    return _interp(yaw), _interp(pit), _interp(rol), float(f[0]), float(f[-1])

# ------------------------------ main runner ---------------------------------

def run_from_corrections(
    calib_file: str,
    left_video: str,
    right_video: str,
    sync_output_dir: str,
    corrections_csv: str,
    cfg: Optional[StereoConfig] = None,
):
    """
    Drive the pipeline using a CSV of per-frame (left) rotation corrections with interpolation.
    The runner ignores skip/take seconds; it uses the CSV frame range instead.
    """
    os.makedirs("output", exist_ok=True)
    video_base = os.path.splitext(os.path.basename(left_video))[0]

    calib = load_calib(calib_file)
    if cfg is None:
        cfg = StereoConfig()

    # 1) match videos (creates/returns sync CSV)
    sync_csv = match_videos(
        left_video, right_video,
        start_seconds=cfg.sync_start_seconds,
        match_duration=cfg.sync_match_duration,
        downsample_factor=cfg.sync_downsample_factor,
        plot=cfg.sync_plot_audio,
        output_dir=sync_output_dir,
        # FLASH params passed through as before:
        flash_occurs_after=cfg.flash_occurs_after,
        flash_occurs_before=cfg.flash_occurs_before,
        flash_center_fraction=cfg.flash_center_fraction,
        flash_min_jump=cfg.flash_min_jump,
        flash_slope_ratio=cfg.flash_slope_ratio,
        flash_baseline_window=cfg.flash_baseline_window,
        flash_brightness_floor=cfg.flash_brightness_floor,
        flash_plot=cfg.flash_plot,
    )
    matched = read_matched(sync_csv)
    if not matched:
        raise RuntimeError("No matched indices from sync step.")

    # 2) read corrections CSV (flexible headers) and build interpolators
    df_corr = _read_corrections_flex(corrections_csv)
    yaw_f, pit_f, rol_f, fmin, fmax = build_interp_functions(df_corr)
    print(f"[corr] frames {int(fmin)}..{int(fmax)}  "
          f"(rows={len(df_corr)}). Using premultiply={cfg.premultiply_rotation_correction}")

    # 3) iterate only matched pairs whose LEFT index is within [fmin..fmax]
    left_files = find_continuation_files(left_video)
    right_files = find_continuation_files(right_video)
    capL = cv2.VideoCapture(left_files[0])
    capR = cv2.VideoCapture(right_files[0])

    state = TrackerState()
    frame_counter = 0

    # outputs
    cross_rows = []
    epi_stats_rows = []

    base_R = np.array(calib["R"], dtype=float)

    # Helpful: keep an eye on thresholds if many pairs drop out
    print(f"[cfg] max_vertical_disparity={cfg.max_vertical_disparity}, "
          f"max_total_distance={cfg.max_total_distance}, "
          f"min_horizontal_disparity={cfg.min_horizontal_disparity}")

    for t_rel, idxL, idxR in matched:
        if idxL < fmin or idxL > fmax:
            continue

        # interpolate corrections (degrees) at this left frame
        dyaw = yaw_f(idxL)
        dpit = pit_f(idxL)
        drol = rol_f(idxL)
        R_corr = euler_ypr_leftframe(dyaw, dpit, drol)

        # compose with base_R (no accumulation across frames!)
        R_current = (R_corr @ base_R) if cfg.premultiply_rotation_correction else (base_R @ R_corr)

        # read frames
        capL.set(cv2.CAP_PROP_POS_FRAMES, int(idxL))
        capR.set(cv2.CAP_PROP_POS_FRAMES, int(idxR))
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            print(f"[skip] read fail at L{idxL}/R{idxR}")
            continue

        # process this pair with per-frame R_current
        cross_3d, labels, frame_counter, state = process_stereo_pair(
            left_bgr=frameL,
            right_bgr=frameR,
            calib=dict(calib, R=R_current),  # override for this call
            state=state,
            frame_counter=frame_counter,
            cfg=cfg,
            separate_fn=separate_LE_and_struts,
            R_current=R_current,  # (optional; utils already takes calib['R'])
        )

        # --- diagnostics: epipolar stats from current matches (if any) ---
        n_pairs = state.n_pairs
        mean_vdisp = p95_vdisp = mean_2d = np.nan
        if state.tracked_cross_left_klt and state.tracked_cross_right_klt:
            Lp = np.array([p[:2] for p in state.tracked_cross_left_klt], float)
            Rp = np.array([p for p in state.tracked_cross_right_klt], float)
            vdisp = np.abs(Lp[:, 1] - Rp[:, 1])
            d2 = np.linalg.norm(Lp - Rp, axis=1)
            mean_vdisp = float(np.mean(vdisp))
            p95_vdisp = float(np.percentile(vdisp, 95))
            mean_2d = float(np.mean(d2))
        epi_stats_rows.append({
            "time_s": float(t_rel),
            "frame_idx": int(idxL),
            "n_pairs": int(n_pairs),
            "mean_vdisp": mean_vdisp,
            "p95_vdisp": p95_vdisp,
            "mean_2d_dist": mean_2d,
            "dyaw_deg": float(dyaw),
            "dpitch_deg": float(dpit),
            "droll_deg": float(drol),
        })
        if len(epi_stats_rows) % 30 == 0:
            print(f"[epi] L{idxL}: n={n_pairs}, mean|Δy|={mean_vdisp:.2f}, p95|Δy|={p95_vdisp:.2f}, "
                  f"mean d2={mean_2d:.1f}, Δ=({dyaw:.2f},{dpit:.2f},{drol:.2f})°")

        # save cross points if present
        if cross_3d is not None:
            for (xyz, lbl) in zip(cross_3d, labels):
                cross_rows.append({
                    "time_s": float(t_rel),
                    "frame_idx": int(idxL),
                    "marker_id": ("" if lbl is None else str(lbl)),
                    "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                })

    capL.release(); capR.release()

    # write outputs
    pd.DataFrame(cross_rows).to_csv(f"output/cross3d_{video_base}.csv", index=False)
    pd.DataFrame(epi_stats_rows).to_csv(f"output/epi_stats_{video_base}.csv", index=False)
    print(f"[✓] Saved: output/cross3d_{video_base}.csv and output/epi_stats_{video_base}.csv")

# ------------------------------- script main ---------------------------------
if __name__ == "__main__":
    # === USER CONFIGURATION ===
    CALIB_FILE     = "Calibration/stereoscopic_calibration/stereo_calibration_output/final_stereo_calibration_V3.pkl"
    LEFT_VIDEO     = "input/left_videos/09_10_merged.mp4"
    RIGHT_VIDEO    = "input/right_videos/09_10_merged.mp4"
    SYNC_OUT_DIR   = "Synchronisation/synchronised_frame_indices"

    # Corrections: CSV/Excel with columns like:
    #   frame_left | delta_yaw_deg | delta_pitch_deg | delta_roll_deg
    # (header names are flexible; see _read_corrections_flex)
    CORRECTIONS    = "input/left_turn_frame_7182.csv"

    # Stereo / detection / debug tunables (see your Photogrammetry.stereo_photogrammetry_utils)
    cfg = StereoConfig(
        # --- matching thresholds ---
        max_vertical_disparity=10.0,
        max_total_distance=600.0,
        min_horizontal_disparity=200.0,

        # --- KLT parameters ---
        lk_win_size=(31, 31),
        lk_max_level=4,
        lk_term_count=40,
        lk_term_eps=0.03,

        # --- rectification for crosses ---
        rectify_alpha_cross=0.0,

        # --- brightness bump for crosses ---
        brighten_alpha=4.0,
        brighten_beta=20.0,
        gradient_brightness_contrast="lr",   # None => uniform alpha/beta; "lr" => gradient

        # --- debug overlay ---
        show_bright_frames=False,
        debug_every_n=30,
        show_debug_frame=True,
        debug_flip_180=True,
        debug_display_scale=0.3,
        debug_window_title_prefix="[DEBUG] L+R Frame",

        # --- refresh rule ---
        need_fresh_min_pairs=50,

        # --- epipolar lines overlay ---
        epipolar_lines=True,

        # --- SYNC (audio) ---
        sync_start_seconds=0.0,
        sync_match_duration=10.0,
        sync_downsample_factor=50,
        sync_plot_audio=True,

        # --- FLASH detection (if used by your matcher) ---
        flash_occurs_after=15,
        flash_occurs_before=39,
        flash_center_fraction=0.33,
        flash_min_jump=20.0,
        flash_slope_ratio=5.0,
        flash_baseline_window=5,
        flash_brightness_floor=0.0,
        flash_plot=True,

        # --- rotation correction application ---
        premultiply_rotation_correction=True,  # R_current = R_corr @ R (set False to do R @ R_corr)
    )

    # Run
    run_from_corrections(CALIB_FILE, LEFT_VIDEO, RIGHT_VIDEO, SYNC_OUT_DIR, CORRECTIONS, cfg)
