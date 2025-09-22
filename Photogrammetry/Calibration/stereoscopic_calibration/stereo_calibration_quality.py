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
    stereo_calib_pkl: str = "stereo_calibration_output/stereo_calibration_wide_84cm_wo_outliers.pkl",
    intrinsics_cam1: str = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_1.pkl",
    intrinsics_cam2: str = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_2.pkl",
    left_folder: str = "left_camera_wide_84cm",
    right_folder: str = "right_camera_wide_84cm",
    checkerboard: Tuple[int, int] = (9, 6),
    square_size_m: float = 3.9/100.0,
    # --- NEW ---
    exclude_indices: Tuple[int, ...] = (4, 5, 31, 32, 33, 34, 35, 36, 37),   # e.g. (4,5,31,32,33,34,35,36,37)
    # optional caps (set to None to disable)
    max_reproj_px: float = None,
    max_epi_px: float = None,
    max_dt_m: float = None,
    max_dr_rad: float = None,
    # print which frames were included/excluded at the end
    report_selection: bool = True
) -> None:
    """Diagnose stereo calibration pair-by-pair and compare against total calibration.

    Exclusions:
      - 'exclude_indices' skips those pairs up-front (by loop index).
      - Thresholds (max_reproj_px, max_epi_px, max_dt_m, max_dr_rad) auto-flag and exclude frames post-metrics.
        Set a threshold to None to disable that particular filter.
    """

    with open(stereo_calib_pkl, "rb") as f:
        stereo_data = pickle.load(f)

    if 'stereoRectify' not in stereo_data:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            stereo_data["camera_matrix_1"], stereo_data["dist_coeffs_1"],
            stereo_data["camera_matrix_2"], stereo_data["dist_coeffs_2"],
            (640, 480), stereo_data["R"], stereo_data["T"], alpha=0
        )
        stereo_data["stereoRectify"] = (R1, R2, P1, P2, Q)
        with open(stereo_calib_pkl, "wb") as f:
            pickle.dump(stereo_data, f)
        print("[INFO] Rectification data saved.")
    else:
        R1, R2, P1, P2, Q = stereo_data["stereoRectify"]
        print("[INFO] Loaded rectification data.")

    R_total = stereo_data["R"]
    T_total = stereo_data["T"].flatten()

    K1, D1 = load_intrinsics(intrinsics_cam1)
    K2, D2 = load_intrinsics(intrinsics_cam2)

    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    left_images  = sorted(glob.glob(os.path.join(left_folder,  "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))
    assert len(left_images) == len(right_images) and len(left_images) > 0, "Mismatch or no images found."

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # accumulators for averages (of kept frames)
    reproj_errors, t_diffs, r_diffs, epi_dists = [], [], [], []

    # tracking selections
    excluded_manual = set(exclude_indices)
    excluded_auto = []
    kept_indices = []

    print("\n🔍 Comparing each pair to total calibration...")
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        if i in excluded_manual:
            print(f"[{i:02d}] ⏭️  Skipped (manual exclude)")
            continue

        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        if imgL is None or imgR is None:
            print(f"[{i:02d}] ❌ Missing image(s)")
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard, None)
        if not (retL and retR):
            print(f"[{i:02d}] ❌ No corners found")
            continue

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        try:
            ret, _, _, _, _, R_i, T_i, _, _ = cv2.stereoCalibrate(
                [objp], [cornersL], [cornersR],
                K1, D1, K2, D2,
                grayL.shape[::-1],
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=criteria
            )

            t_diff = float(np.linalg.norm(T_total - T_i.flatten()))
            r_diff = rodrigues_angle_diff(R_total, R_i)

            F, _ = cv2.findFundamentalMat(cornersL, cornersR, method=cv2.FM_8POINT)
            epi = calculate_epipolar_distance(cornersL, cornersR, F) if F is not None else float("nan")

            line = f"[{i:02d}] ✅ reproj={ret:.4f}px | ΔT={t_diff:.4f} | ΔR={r_diff:.4f} rad | epi={epi:.4f}px"

            # auto-exclusion based on thresholds
            fails = []
            if (max_reproj_px is not None) and (ret > max_reproj_px):
                fails.append(f"reproj>{max_reproj_px}")
            if (max_dt_m is not None) and (t_diff > max_dt_m):
                fails.append(f"ΔT>{max_dt_m}")
            if (max_dr_rad is not None) and (r_diff > max_dr_rad):
                fails.append(f"ΔR>{max_dr_rad}")
            if (max_epi_px is not None) and (not np.isnan(epi)) and (epi > max_epi_px):
                fails.append(f"epi>{max_epi_px}")

            if fails:
                print(line + "  -> 🚫 excluded (" + ", ".join(fails) + ")")
                excluded_auto.append(i)
                continue

            print(line)
            # keep
            kept_indices.append(i)
            reproj_errors.append(ret)
            t_diffs.append(t_diff)
            r_diffs.append(r_diff)
            if not np.isnan(epi):
                epi_dists.append(epi)

        except cv2.error as e:
            print(f"[{i:02d}] ❌ Calibration failed: {e}")

    # print averages over kept frames
    if reproj_errors:
        print("\n📊 Average over kept frames:")
        print(f"   reproj={np.mean(reproj_errors):.4f}px | "
              f"ΔT={np.mean(t_diffs):.4f} | "
              f"ΔR={np.mean(r_diffs):.4f} rad | "
              f"epi={np.mean(epi_dists) if epi_dists else float('nan'):.4f}px")

    if report_selection:
        kept_str = ", ".join(f"{k:02d}" for k in kept_indices)
        excluded_manual_str = ", ".join(f"{k:02d}" for k in sorted(excluded_manual))
        excluded_auto_str = ", ".join(f"{k:02d}" for k in excluded_auto)
        print("\n✅ Kept indices:   [" + kept_str + "]")
        if excluded_manual:
            print("⏭️  Manually excluded:", "[" + excluded_manual_str + "]")
        if excluded_auto:
            print("🚫 Auto-excluded:     [" + excluded_auto_str + "]")



if __name__ == "__main__":
    diagnose_stereo_pairs()
