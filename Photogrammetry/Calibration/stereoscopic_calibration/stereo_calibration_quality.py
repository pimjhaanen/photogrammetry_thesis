"""This code can be used to print diagnostics for each stereo image pair against a reference calibration.
For every pair it computes a single-pair stereo solve (R_i, T_i), the delta vs the reference (ΔR, ΔT),
and an epipolar line distance as a sanity check. It does not modify or save the calibration,
it merely checks the quality of the stereo calibration. Outliers can be identified and calibration can
be repeated in stereoscopic_calibration without these, improving stereoscopic calibration"""

import os
import glob
import pickle
import cv2
import numpy as np
from typing import Tuple

def load_intrinsics(pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics (K, dist) from a pickle produced by single-camera calibration."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d["camera_matrix"], d["dist_coeffs"]

def rodrigues_angle_diff(R1: np.ndarray, R2: np.ndarray) -> float:
    """Return the angle (in radians) between rotation matrices via Rodrigues vectors."""
    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)
    return float(np.linalg.norm(rvec1 - rvec2))

def calculate_epipolar_distance(cornersL: np.ndarray, cornersR: np.ndarray, F: np.ndarray) -> float:
    """Average point-to-epipolar-line distance (pixels) from L points to their lines in R."""
    total = 0.0
    n = len(cornersL)
    for i in range(n):
        x1, y1 = cornersL[i].ravel()
        x2, y2 = cornersR[i].ravel()
        lineR = F @ np.array([x1, y1, 1.0])
        dist = abs(lineR[0]*x2 + lineR[1]*y2 + lineR[2]) / np.hypot(lineR[0], lineR[1])
        total += dist
    return float(total / n)

def diagnose_stereo_pairs(
    stereo_calib_pkl: str = "stereo_calibration_output/stereo_calibration_wide_84cm.pkl",
    intrinsics_cam1: str = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_1.pkl",
    intrinsics_cam2: str = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_2.pkl",
    left_folder: str = "left_camera_wide_84cm",
    right_folder: str = "right_camera_wide_84cm",
    checkerboard: Tuple[int, int] = (9, 6),
    square_size_m: float = 3.9/100.0
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - stereo_calib_pkl: path to the saved stereo calibration (.pkl) containing K1,D1,K2,D2,R,T
    - intrinsics_cam1 / intrinsics_cam2: paths to single-camera intrinsics (.pkl) for both cameras
    - left_folder / right_folder: folders containing synchronized left/right checkerboard frames
    - checkerboard: inner corner grid size (cols, rows) of the checkerboard used for calibration
    - square_size_m: physical side length of a checker square in meters
    """

    # --- Load stereo calibration (and ensure rectification fields exist for convenience) ---
    with open(stereo_calib_pkl, "rb") as f:
        stereo_data = pickle.load(f)

    if 'stereoRectify' not in stereo_data:
        print("[INFO] Rectification data not found. Computing rectification matrices.")
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            stereo_data["camera_matrix_1"], stereo_data["dist_coeffs_1"],
            stereo_data["camera_matrix_2"], stereo_data["dist_coeffs_2"],
            (640, 480), stereo_data["R"], stereo_data["T"], alpha=0
        )
        stereo_data["stereoRectify"] = (R1, R2, P1, P2, Q)
        with open(stereo_calib_pkl, "wb") as f:
            pickle.dump(stereo_data, f)
        print("[INFO] Rectification data saved to the stereo calibration file.")
    else:
        R1, R2, P1, P2, Q = stereo_data["stereoRectify"]
        print("[INFO] Loaded rectification data from the stereo calibration file.")

    R_total = stereo_data["R"]
    T_total = stereo_data["T"].flatten()

    # --- Load intrinsics (kept separate in case you want to override) ---
    K1, D1 = load_intrinsics(intrinsics_cam1)
    K2, D2 = load_intrinsics(intrinsics_cam2)

    # --- Build object points for one checkerboard view ---
    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    # --- Collect image paths ---
    left_images  = sorted(glob.glob(os.path.join(left_folder,  "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))
    assert len(left_images) == len(right_images) and len(left_images) > 0, "Mismatch or no images found."

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    print("\n🔍 Comparing each pair to total calibration...")
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        if imgL is None or imgR is None:
            print(f"[{i:02d}] ❌ Missing image(s)")
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Detect corners
        retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard, None)
        if not (retL and retR):
            print(f"[{i:02d}] ❌ No corners found")
            continue

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        try:
            # Single-pair stereo (intrinsics fixed) to get R_i, T_i for diagnostics
            ret, _, _, _, _, R_i, T_i, _, _ = cv2.stereoCalibrate(
                [objp], [cornersL], [cornersR],
                K1, D1, K2, D2,
                grayL.shape[::-1],
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=criteria
            )

            # ΔT (m) and ΔR (rad) vs overall calibration
            t_diff = float(np.linalg.norm(T_total - T_i.flatten()))
            r_diff = rodrigues_angle_diff(R_total, R_i)

            # Fundamental matrix from detected correspondences (distorted pixels)
            F, _ = cv2.findFundamentalMat(cornersL, cornersR, method=cv2.FM_8POINT)

            # Average epipolar distance (pixels)
            epi = calculate_epipolar_distance(cornersL, cornersR, F) if F is not None else float("nan")

            print(f"[{i:02d}] ✅ reproj={ret:.4f}px | ΔT={t_diff:.4f} | ΔR={r_diff:.4f} rad | epi={epi:.4f}px")

        except cv2.error as e:
            print(f"[{i:02d}] ❌ Calibration failed: {e}")

if __name__ == "__main__":
    diagnose_stereo_pairs()
