import cv2
import numpy as np
import pickle
import os
import csv
from synchronisation.match_frames_2_videos import match_videos, find_continuation_files
from stereoscopic_photogrammetry_function import process_stereo_pair
import pandas as pd


if __name__ == "__main__":
    # === CONFIGURATION ===
    CALIB_FILE = "Calibration/stereoscopic_calibration/stereo_calibration_wide_84cm_filtered.pkl"
    LEFT_VIDEO = "main simulation/left_videos/25_06_test2_merged.mp4"
    RIGHT_VIDEO = "main simulation/right_videos/25_06_test2_merged.mp4"
    SYNC_OUTPUT_DIR = "synchronisation/matched_sync"

    # Extract the base name of the video (without the path and extension)
    video_name = os.path.splitext(os.path.basename(LEFT_VIDEO))[0]  # Extract '25_06_test2' from the file path

    # === STEP 1: LOAD CALIBRATION DATA ===
    with open(CALIB_FILE, "rb") as f:
        calib = pickle.load(f)

    mtx1 = calib["camera_matrix_1"]
    dist1 = calib["dist_coeffs_1"]
    mtx2 = calib["camera_matrix_2"]
    dist2 = calib["dist_coeffs_2"]
    R = calib["R"]
    T = calib["T"]

    # === STEP 4: Match settings ===
    skip_seconds = 601  # Start at 601 seconds (left video)
    start_seconds_for_matching = 0 #Start a minute before take-off with matching to minimize drift
    end_seconds = skip_seconds + 20  # Stop at 611 seconds (left video)
    fps = 30
    start_index = skip_seconds * fps
    end_index = end_seconds * fps

    # === STEP 2: SYNC VIDEO FILES ===
    output_file = match_videos(LEFT_VIDEO, RIGHT_VIDEO, 0,
                               match_duration=600, plot=False,
                               output_dir=SYNC_OUTPUT_DIR)
    SYNC_CSV = f"{output_file}"

    # === STEP 3: READ FRAME MATCHES ===
    matched_indices = []
    with open(SYNC_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            matched_indices.append((int(row[0]), int(row[1])))


    # === VIDEO HANDLING ===
    # Get continuation files for left and right videos
    left_video_files = find_continuation_files(LEFT_VIDEO)
    right_video_files = find_continuation_files(RIGHT_VIDEO)

    # Open the first video file for left and right
    cap_left = cv2.VideoCapture(left_video_files[0])
    cap_right = cv2.VideoCapture(right_video_files[0])

    # Get the total number of frames in the first video file
    total_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

    # === Check if the start index exceeds the number of frames in the first video ===
    if start_index >= total_frames_left:
        # If the start index exceeds the total frames, move to the next video in the sequence
        start_index -= total_frames_left
        cap_left = cv2.VideoCapture(left_video_files[1])
        cap_right = cv2.VideoCapture(right_video_files[1])

    frame_counter = 0
    cross_3d_all = []
    aruco_coords_data = []  # To store x, y, z per ArUco
    cross_labels_all = []
    frame_indices_used = []
    aruco_data = []  # For ArUco data (ID, rotation, translation)

    for i, (idx_left, idx_right) in enumerate(matched_indices):
        if i < start_index:
            continue
        if i >= end_index:
            break

        cap_left.set(cv2.CAP_PROP_POS_FRAMES, idx_left)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, idx_right)

        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        if not ret1 or not ret2:
            print(f"[!] Skipping frame pair {idx_left}, {idx_right} (read failed)")
            continue

        result = process_stereo_pair(frame_left, frame_right, calib, frame_counter)
        if result is None or result[0] is None:
            print(f"[INFO] Frame {idx_left}/{idx_right}"
                  f" skipped due to matching failure.")
            continue

        cross_3d, label, aruco_3d_4by4, aruco_ids_4x4, aruco_3d_7x7, \
        aruco_ids_7x7, aruco_rotations_7x7, aruco_translations_7x7,\
        frame_counter = result

        # Save cross marker 3D data (same as before)
        print(label)
        cross_3d_all.append(list(zip(cross_3d, np.array(label))))  # pairs each point with its label
        frame_indices_used.append((idx_left, idx_right))

        # Save ArUco data (ID, rotation, translation)
        for i, aruco_id in enumerate(aruco_ids_7x7):
            rotation = aruco_rotations_7x7[i][0].flatten()\
                if len(aruco_rotations_7x7[i]) > 0 else [np.nan, np.nan, np.nan]
            translation = aruco_translations_7x7[i][0].flatten() if\
                len(aruco_translations_7x7[i]) > 0 else [np.nan, np.nan, np.nan]

            aruco_data.append({
                "frame_idx": idx_left,
                "aruco_id": aruco_id,
                "rotation_x": rotation[0],
                "rotation_y": rotation[1],
                "rotation_z": rotation[2],
                "translation_x": translation[0],
                "translation_y": translation[1],
                "translation_z": translation[2]
            })

        # Save 4x4 ArUco coordinates
        for i, aruco_id in enumerate(aruco_ids_4x4):
            coord = aruco_3d_4by4[i]
            aruco_coords_data.append({
                "frame_idx": idx_left,
                "type": "4x4",
                "aruco_id": aruco_id,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2])
            })

        # Save 7x7 ArUco coordinates
        for i, aruco_id in enumerate(aruco_ids_7x7):
            coord = aruco_3d_7x7[i]
            aruco_coords_data.append({
                "frame_idx": idx_left,
                "type": "7x7",
                "aruco_id": aruco_id,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2])
            })

    cap_left.release()
    cap_right.release()

    # Save results
    os.makedirs("main simulation/output", exist_ok=True)

    # === Save cross marker 3D data ===
    cross_data = []
    for i, coords in enumerate(cross_3d_all):
        frame_idx = frame_indices_used[i][0]
        for point in coords:
            try:
                xyz, label = point
            except (ValueError, TypeError):
                xyz = point
                label = None

            cross_data.append({
                "frame_idx": frame_idx,
                "marker_id": label if label is not None else "",
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2])
            })

    cross_df = pd.DataFrame(cross_data)
    cross_df.to_csv(f"main simulation/output/3d_coordinates_{video_name}.csv"
                    , index=False)

    # === Save ArUco marker pose data ===
    aruco_df = pd.DataFrame(aruco_data)
    aruco_df.to_csv(f"main simulation/output/aruco_pose_{video_name}.csv"
                    , index=False)

    aruco_coords_df = pd.DataFrame(aruco_coords_data)
    aruco_coords_df.to_csv(f"main simulation/output/aruco_coordinates_{video_name}.csv", index=False)

    print(f"[✓] Processed {len(cross_3d_all)} valid frame pairs.")
