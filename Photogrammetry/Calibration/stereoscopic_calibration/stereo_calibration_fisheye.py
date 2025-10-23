"""Stereo fisheye calibration from checkerboards.

- Loads fisheye intrinsics (K, D) for both cameras from your .pkl files
  (produced by your fisheye single-camera calibration).
- Detects checkerboard corners in left/right frames, refines to subpixel,
  and runs cv2.fisheye.stereoCalibrate with FIX_INTRINSIC.
- Saves R, T, E, F (and copies intrinsics) to stereo_calibration_output/<basename>.pkl
"""

import os, glob, pickle
import cv2
import numpy as np


# ------------------------------ I/O helpers ----------------------------------

def get_images(patterns):
    files = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)

def load_intrinsics(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    K = np.asarray(d["camera_matrix"], dtype=np.float64)
    D = np.asarray(d["dist_coeffs"], dtype=np.float64)
    return K, D


# ------------------------------ math helpers ---------------------------------

def _skew(tvec3):
    t = np.asarray(tvec3, dtype=np.float64).ravel()
    return np.array([[    0, -t[2],  t[1]],
                     [ t[2],     0, -t[0]],
                     [-t[1],  t[0],     0]], dtype=np.float64)


# --------------------------- main calibration --------------------------------

def stereo_calibrate_fisheye_from_checkerboards(
    left_glob="left_camera/*.jpg",
    right_glob="right_camera/*.jpg",
    checkerboard=(9, 6),            # inner corners (nx, ny)
    square_size_m=3.9/100.0,        # physical square size
    intrinsics_cam1="../single-camera calibration/single_calibration_output/calibration_fisheye_wide_cam1.pkl",
    intrinsics_cam2="../single-camera calibration/single_calibration_output/calibration_fisheye_wide_cam2.pkl",
    exclude_frames=None,
    output_basename="stereo_fisheye_calibration"
):
    """RELEVANT FUNCTION INPUTS:
    - left_glob / right_glob: folders with frames from both cameras
    - checkerboard: inner-corner grid (cols, rows)
    - square_size_m: physical square size (meters)
    - intrinsics_cam1 / intrinsics_cam2: fisheye single-cam .pkl files
    - exclude_frames: list of integer indices to skip
    - output_basename: filename stem for results
    """
    os.makedirs("stereo_calibration_output", exist_ok=True)
    exclude_frames = set(exclude_frames or [])

    # Load fisheye intrinsics (float64)
    K1, D1 = load_intrinsics(intrinsics_cam1)
    K2, D2 = load_intrinsics(intrinsics_cam2)

    # Prepare object points in board coordinates (shape (1, N, 3), float64)
    nx, ny = checkerboard
    objp = np.zeros((1, nx*ny, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2).astype(np.float64)
    objp *= float(square_size_m)

    objpoints = []   # list of (1, N, 3)
    imgL_pts  = []   # list of (1, N, 2)
    imgR_pts  = []   # list of (1, N, 2)

    left_images  = get_images(left_glob)
    right_images = get_images(right_glob)
    assert len(left_images) == len(right_images), "Mismatch in number of left/right frames"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    img_size = None

    # Robust detection flags help on GoPro frames
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK)

    for idx, (lpath, rpath) in enumerate(zip(left_images, right_images)):
        if idx in exclude_frames:
            print(f"[{idx:03d}] ❌ Skipping excluded frame")
            continue

        imgL = cv2.imread(lpath); imgR = cv2.imread(rpath)
        if imgL is None or imgR is None:
            print(f"[{idx:03d}] Could not read one of the images.")
            continue
        if imgL.shape[:2] != imgR.shape[:2]:
            print(f"[{idx:03d}] Size mismatch L/R: {imgL.shape} vs {imgR.shape} — skipping.")
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        img_size = grayL.shape[::-1]  # (w, h)

        retL, cornersL = cv2.findChessboardCorners(grayL, (nx, ny), cb_flags)
        retR, cornersR = cv2.findChessboardCorners(grayR, (nx, ny), cb_flags)
        print(f"[{idx:03d}] {os.path.basename(lpath)}: left={nx*ny if retL else 0}, right={nx*ny if retR else 0}")

        if not (retL and retR):
            continue

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

        imgL_pts.append(cornersL.reshape(1, -1, 2).astype(np.float64))
        imgR_pts.append(cornersR.reshape(1, -1, 2).astype(np.float64))
        objpoints.append(objp.copy())

    if not objpoints:
        raise RuntimeError("No valid stereo pairs detected. Aborting.")

    # Fisheye stereo calibration. Keep intrinsics fixed (we trust single-cam fisheye).
    flags = (cv2.fisheye.CALIB_FIX_INTRINSIC |
             cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    rms, K1o, D1o, K2o, D2o, R, T = cv2.fisheye.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgL_pts,
        imagePoints2=imgR_pts,
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        imageSize=img_size,  # <-- camelCase!
        R=None, T=None,
        flags=flags,
        criteria=criteria
    )

    # Compute E and F for convenience (even with fisheye intrinsics this is useful diagnostically)
    E = _skew(T) @ R
    K1inv = np.linalg.inv(K1o)
    K2invT = np.linalg.inv(K2o).T
    F = K2invT @ E @ K1inv

    baseline_m = float(np.linalg.norm(T.ravel()))
    print(f"\nRMS reprojection error (fisheye): {rms:.4f} px")
    print(f"Baseline: {baseline_m:.3f} m")
    print(f"R:\n{R}\nT (m):\n{T}\nE:\n{E}\nF:\n{F}")

    out_path = os.path.join("stereo_calibration_output", f"{output_basename}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": "fisheye",
            "R": R, "T": T,
            "E": E, "F": F,
            "reprojection_error": float(rms),
            "camera_matrix_1": K1o, "dist_coeffs_1": D1o,
            "camera_matrix_2": K2o, "dist_coeffs_2": D2o,
            "image_size": tuple(map(int, img_size)),
            "baseline_m": baseline_m
        }, f)
    print(f"\n✅ Saved: {out_path}")
    return out_path


# ---------------------------- script entrypoint -------------------------------

if __name__ == "__main__":
    stereo_calibrate_fisheye_from_checkerboards(
        left_glob="left_camera_ireland/*.jpg",
        right_glob="right_camera_ireland/*.jpg",
        checkerboard=(9, 6),
        square_size_m=3.9/100.0,
        intrinsics_cam1="../single-camera calibration/single_calibration_output/calibration_fisheye_wide_cam1.pkl",
        intrinsics_cam2="../single-camera calibration/single_calibration_output/calibration_fisheye_wide_cam2.pkl",
        exclude_frames=None,
        output_basename="stereo_calibration_ireland_fisheye"
    )
