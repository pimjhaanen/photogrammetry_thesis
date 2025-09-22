"""This code can be used to run the stereo-photogrammetry main pipeline on a pair of synced videos.
It (1) loads stereo calibration, (2) synchronizes left/right videos via your sync CSV,
(3) iterates matched frame pairs, (4) calls `process_stereo_pair(...)` to detect/track crosses
+ ArUco and triangulate to 3D, and (5) saves 3 CSVs with cross and ArUco outputs.

You can adapt: input files/paths, time window (skip/end seconds), FPS, and all stereo/
detection/debug tunables through `StereoConfig` (imported from your processing module)."""

import os
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# --- Your project utilities ---
from Photogrammetry.Synchronisation.synchronisation_utils import match_videos, find_continuation_files

# Import the refactored core + config/state dataclasses
from Photogrammetry.stereo_photogrammetry_utils import (
    process_stereo_pair, StereoConfig, TrackerState
)

# Import the cross labeling function
from Photogrammetry.kite_shape_reconstruction_utils import separate_LE_and_struts


# ------------------------------ Helper I/O -----------------------------------

def _load_calibration(calib_pkl_path: str):
    """Load stereo calibration dict from a .pkl created by your calibration step.

    RELEVANT FUNCTION INPUTS:
    - calib_pkl_path: path to the pickle with keys:
      camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T
    """
    with open(calib_pkl_path, "rb") as f:
        calib = pickle.load(f)
    required_keys = ["camera_matrix_1", "dist_coeffs_1", "camera_matrix_2", "dist_coeffs_2", "R", "T"]
    missing = [k for k in required_keys if k not in calib]
    if missing:
        raise KeyError(f"Calibration file is missing keys: {missing}")
    return calib


def _read_matched_indices(sync_csv_path: str) -> List[Tuple[float, int, int]]:
    """Read time and left/right matched frame indices from the sync CSV produced by `match_videos`.

    RELEVANT FUNCTION INPUTS:
    - sync_csv_path: path to CSV created by `match_videos`

    Returns:
    - list of tuples: (time_s_relative_to_flash, left_frame_idx, right_frame_idx)

    Notes:
    - Supports both formats:
      * New: Time_s,Frame_Video1,Frame_Video2
      * Legacy: Frame_Video1,Frame_Video2  (time is reconstructed as None -> you can skip time filtering)
    """
    out: List[Tuple[float, int, int]] = []
    with open(sync_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Detect format
        has_time = False
        if header:
            cols = [h.strip().lower() for h in header]
            has_time = ("time_s" in cols)

        if has_time:
            # Expect: Time_s, Frame_Video1, Frame_Video2
            for row in reader:
                if len(row) < 3:
                    continue
                t = float(row[0])
                i1 = int(row[1])
                i2 = int(row[2])
                out.append((t, i1, i2))
        else:
            # Legacy: Frame_Video1, Frame_Video2 (no time column)
            # We will set time to NaN here; downstream code only uses indices for windowing.
            for row in reader:
                if len(row) < 2:
                    continue
                i1 = int(row[0])
                i2 = int(row[1])
                out.append((float("nan"), i1, i2))
    return out


# ------------------------------ Main runner ----------------------------------

def run_photogrammetry(
    calib_file: str,
    left_video_path: str,
    right_video_path: str,
    sync_output_dir: str,
    skip_seconds: int = 0,
    take_seconds: int = 20,
    fps: int = 30,
    cfg: Optional[StereoConfig] = None,
):
    """RELEVANT FUNCTION INPUTS:
    - calib_file: path to stereo calibration .pkl with keys {camera_matrix_1/2, dist_coeffs_1/2, R, T}.
    - left_video_path / right_video_path: input video files (or the first file of a continuation sequence).
    - sync_output_dir: directory where the sync CSV from `match_videos` is (or will be) stored.
    - skip_seconds: how many seconds to skip from the beginning of the *matched pair sequence* before processing.
    - take_seconds: how many seconds to process after skipping (window length).
    - fps: nominal frames per second to convert seconds -> indices (used for the window limits).
    - cfg: StereoConfig with all tunables (KLT, rectification, ArUco params, debug overlay, etc.).
           If None, defaults are used.

    OUTPUT:
    - Writes three CSV files into 'main simulation/output':
        * 3d_coordinates_<video_name>.csv        (now includes time_s)
        * aruco_pose_<video_name>.csv
        * aruco_coordinates_<video_name>.csv     (now includes time_s)
    """
    os.makedirs("main simulation/output", exist_ok=True)
    video_name = os.path.splitext(os.path.basename(left_video_path))[0]

    # 1) Load calibration
    calib = _load_calibration(calib_file)

    # 2) Synchronize videos and get the CSV with matched frames (and time relative to flash = 0)
    sync_csv = match_videos(
        left_video_path,
        right_video_path,
        start_seconds=0,             # keep as in your original call
        match_duration=600,
        downsample_factor=50,
        plot=False,
        output_dir=sync_output_dir,
    )

    matched = _read_matched_indices(sync_csv)
    if not matched:
        raise RuntimeError(f"No matched indices found in {sync_csv}")

    # 3) Prepare time window in "matched index" space
    start_index = max(0, int(skip_seconds * fps))
    end_index = min(len(matched), int((skip_seconds + take_seconds) * fps))

    # 4) Handle continuation files (if videos were split)
    left_files = find_continuation_files(left_video_path)
    right_files = find_continuation_files(right_video_path)
    cap_left = cv2.VideoCapture(left_files[0])
    cap_right = cv2.VideoCapture(right_files[0])

    # 5) Prepare processing config/state
    if cfg is None:
        cfg = StereoConfig()  # defaults as defined in the refactored module
    state = TrackerState()
    frame_counter = 0  # processing frame counter inside your pipeline

    # 6) Output accumulators
    cross_3d_all: List[List[Tuple[np.ndarray, str]]] = []  # per-frame list of (xyz, label)
    aruco_pose_rows = []    # 7×7 pose: one row per detected 7×7 marker per frame
    aruco_coords_rows = []  # 4×4 & 7×7 centers in 3D: one row per marker per frame
    frame_indices_used: List[Tuple[int, int]] = []
    times_used: List[float] = []

    # 7) Iterate over matched frame pairs
    for i in range(start_index, end_index):
        t_rel, idx_left, idx_right = matched[i]

        # If your matched indices can exceed the first file, you may need to switch caps here.
        # For simplicity, we assume indices fit in the current capture.
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, idx_left)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, idx_right)

        okL, frame_left = cap_left.read()
        okR, frame_right = cap_right.read()
        if not okL or not okR:
            print(f"[!] Skipping pair {idx_left}/{idx_right}: failed to read frame(s).")
            continue

        # --- CORE: call the refactored processor ---
        (
            cross_3d, labels,
            aruco_3d_4x4, aruco_ids_4x4,
            aruco_3d_7x7, aruco_ids_7x7,
            aruco_rvecs_7x7, aruco_tvecs_7x7,
            frame_counter,
            state,
        ) = process_stereo_pair(
            left_bgr=frame_left,
            right_bgr=frame_right,
            calib=calib,
            state=state,
            frame_counter=frame_counter,
            cfg=cfg,
            separate_fn=separate_LE_and_struts,
        )

        # If no crosses for this frame, continue (processor already guarded)
        if cross_3d is None:
            print(f"[INFO] Pair {idx_left}/{idx_right} produced no usable crosses; skipping.")
            continue

        # Save cross marker 3D data (pair each point with its label)
        cross_3d_all.append(list(zip(cross_3d, np.array(labels, dtype=object))))
        frame_indices_used.append((idx_left, idx_right))
        times_used.append(float(t_rel))

        # Save 7×7 ArUco pose (rvec/tvec lists from cv2.aruco.estimatePoseSingleMarkers)
        for j, aruco_id in enumerate(aruco_ids_7x7):
            rvec = aruco_rvecs_7x7[j] if j < len(aruco_rvecs_7x7) else None
            tvec = aruco_tvecs_7x7[j] if j < len(aruco_tvecs_7x7) else None

            rv = np.ravel(rvec) if rvec is not None and np.size(rvec) >= 3 else [np.nan, np.nan, np.nan]
            tv = np.ravel(tvec) if tvec is not None and np.size(tvec) >= 3 else [np.nan, np.nan, np.nan]

            aruco_pose_rows.append({
                "time_s": float(t_rel),
                "frame_idx": idx_left,
                "aruco_id": int(aruco_id),
                "rotation_x": float(rv[0]),
                "rotation_y": float(rv[1]),
                "rotation_z": float(rv[2]),
                "translation_x": float(tv[0]),
                "translation_y": float(tv[1]),
                "translation_z": float(tv[2]),
            })

        # Save 4×4 ArUco 3D centers
        for j, aruco_id in enumerate(aruco_ids_4x4):
            coord = aruco_3d_4x4[j]
            aruco_coords_rows.append({
                "time_s": float(t_rel),
                "frame_idx": idx_left,
                "type": "4x4",
                "aruco_id": int(aruco_id),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            })

        # Save 7×7 ArUco 3D centers (from stereo triangulation, not tvec)
        for j, aruco_id in enumerate(aruco_ids_7x7):
            coord = aruco_3d_7x7[j]
            aruco_coords_rows.append({
                "time_s": float(t_rel),
                "frame_idx": idx_left,
                "type": "7x7",
                "aruco_id": int(aruco_id),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            })

        print(f"[frame: {i - start_index:5d}/{end_index - start_index}] processed… "
              f"pairs in use: {state.n_pairs}, ArUco's detected: {len(aruco_ids_4x4)+len(aruco_ids_7x7)} ")

    # 8) Cleanup video resources
    cap_left.release()
    cap_right.release()

    # 9) Save results to CSV
    # Cross 3D coordinates (+ time)
    cross_rows = []
    for (frame_idx, _), points, t_rel in zip(frame_indices_used, cross_3d_all, times_used):
        for xyz, lbl in points:
            marker_id = "" if lbl is None else str(lbl)
            cross_rows.append({
                "time_s": float(t_rel),
                "frame_idx": frame_idx,
                "marker_id": marker_id,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            })
    pd.DataFrame(cross_rows).to_csv(
        f"output/3d_coordinates_{video_name}.csv", index=False
    )

    # 7×7 pose (rvec/tvec from PnP on left image)
    pd.DataFrame(aruco_pose_rows).to_csv(
        f"output/aruco_pose_{video_name}.csv", index=False
    )

    # 4×4 & 7×7 triangulated 3D centers (+ time)
    pd.DataFrame(aruco_coords_rows).to_csv(
        f"output/aruco_coordinates_{video_name}.csv", index=False
    )

    print(f"[✓] Processed {len(cross_3d_all)} valid frame pairs.")
    print(f"[→] Saved CSVs to: main simulation/output/ (basename: {video_name})")


# ------------------------------- script main ---------------------------------
if __name__ == "__main__":
    # === USER CONFIGURATION ===
    CALIB_FILE = "Calibration/stereoscopic_calibration/stereo_calibration_output/stereo_calibration_wide_84cm_wo_outliers.pkl"
    LEFT_VIDEO = "input/left_videos/25_06_test1.mp4"
    RIGHT_VIDEO = "input/right_videos/25_06_test1.mp4"
    SYNC_OUTPUT_DIR = "Synchronisation/synchronised_frame_indices"

    # Time window (in seconds) inside the matched pair sequence
    SKIP_SECONDS = 323          # start at 601 s into the matched sequence
    TAKE_SECONDS = 17           # process 20 s (601–621 s)
    FPS = 30                    # nominal fps for window math

    # Stereo / detection / debug tunables (override any defaults you want)
    cfg = StereoConfig(
        # --- matching thresholds ---
        max_vertical_disparity=70.0,
        max_total_distance=500.0,
        min_horizontal_disparity=0.0,

        # --- KLT parameters ---
        lk_win_size=(31, 31),
        lk_max_level=4,
        lk_term_count=40,
        lk_term_eps=0.03,

        # --- rectification ---
        rectify_alpha_aruco=0.0,
        rectify_alpha_cross=0.0,

        # --- brightness bump for crosses ---
        brighten_alpha=4.0,
        brighten_beta=20.0,

        # --- ArUco config ---
        aruco_4x4_dict="DICT_4X4_100",
        aruco_7x7_dict="DICT_7X7_50",
        aruco_adapt_win_min=5,
        aruco_adapt_win_max=15,
        aruco_adapt_win_step=10,
        aruco_corner_refine_method=cv2.aruco.CORNER_REFINE_SUBPIX,
        aruco_min_perimeter_rate=0.001,
        aruco_max_perimeter_rate=4.0,
        aruco_min_distance_to_border=1,
        aruco_marker_length_7x7_m=0.19,

        # --- debug overlay ---
        show_bright_frames=False,
        debug_every_n=30,
        show_debug_frame=True,
        debug_flip_180=True,
        debug_display_scale=0.3,
        debug_window_title_prefix="[DEBUG] L+R Frame",

        # --- refresh rule ---
        need_fresh_min_pairs=30,

        # --- epipolar lines overlay ---
        epipolar_lines=False,
    )

    # Run
    run_photogrammetry(
        calib_file=CALIB_FILE,
        left_video_path=LEFT_VIDEO,
        right_video_path=RIGHT_VIDEO,
        sync_output_dir=SYNC_OUTPUT_DIR,
        skip_seconds=SKIP_SECONDS,
        take_seconds=TAKE_SECONDS,
        fps=FPS,
        cfg=cfg,
    )
