"""This code can be used to perform a stereo calibration with two cameras using a checkerboard.
It loads the intrinsic calibrations of both cameras and estimates the relative pose (R, T, E, F).
You can exclude faulty frames and define the board geometry for accurate calibration.
Be sure to run stereo_calibration_quality afterwards to check the quality, and possibly
exclude frames here to try and recalibrate, depending on prefered quality."""

import os, glob, pickle
import cv2
import numpy as np
def get_images(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)
def load_intrinsics(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d["camera_matrix"], d["dist_coeffs"]

def stereo_calibrate_from_checkerboards(
    left_glob="left_camera_wide_84cm/*.jpg",
    right_glob="right_camera_wide_84cm/*.jpg",
    checkerboard=(9, 6),
    square_size_m=3.9/100.0,
    intrinsics_cam1="../single-camera calibration/calibration_checkerboard_wide_camera_1.pkl",
    intrinsics_cam2="../single-camera calibration/calibration_checkerboard_wide_camera_2.pkl",
    exclude_frames=None,
    output_basename="stereo_calibration_wide_84cm"
):
    """RELEVANT FUNCTION INPUTS:
    - left_glob / right_glob: folders with frames from both cameras (e.g. frame_0001.jpg, …)
    - checkerboard: number of inner corners (cols, rows) of the calibration board
    - square_size_m: physical size of one checker square in meters
    - intrinsics_cam1 / intrinsics_cam2: .pkl files with intrinsic calibration parameters
    - exclude_frames: list of indices of frames to skip
    - output_basename: filename (without extension) for the saved stereo calibration results"""

    os.makedirs("stereo_calibration_output", exist_ok=True)
    exclude_frames = set(exclude_frames or [])

    mtx1, dist1 = load_intrinsics(intrinsics_cam1)
    mtx2, dist2 = load_intrinsics(intrinsics_cam2)

    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints, imgL_pts, imgR_pts = [], [], []
    left_images = get_images([left_glob])
    right_images = get_images([right_glob])

    assert len(left_images) == len(right_images), "Mismatch in number of left/right frames"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    img_size = None

    for idx, (lpath, rpath) in enumerate(zip(left_images, right_images)):
        if idx in exclude_frames:
            print(f"[{idx}] ❌ Skipping excluded frame")
            continue

        imgL = cv2.imread(lpath); imgR = cv2.imread(rpath)
        if imgL is None or imgR is None:
            print(f"[{idx}] Could not read one of the images.")
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        img_size = grayL.shape[::-1]

        retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard, None)
        print(f"[{idx}] {os.path.basename(lpath)}: left={len(cornersL) if retL else 0}, right={len(cornersR) if retR else 0}")

        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgL_pts.append(cornersL)
            imgR_pts.append(cornersR)

    if not objpoints:
        raise RuntimeError("No valid stereo pairs detected. Aborting.")

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgL_pts, imgR_pts,
        mtx1, dist1, mtx2, dist2,
        img_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"\nRMS reprojection error: {ret:.4f} px")
    print(f"R:\n{R}\nT (m):\n{T}\nE:\n{E}\nF:\n{F}")

    out_path = os.path.join("stereo_calibration_output", f"{output_basename}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "R": R, "T": T, "E": E, "F": F,
            "reprojection_error": ret,
            "camera_matrix_1": mtx1, "dist_coeffs_1": dist1,
            "camera_matrix_2": mtx2, "dist_coeffs_2": dist2
        }, f)
    print(f"\n✅ Saved: {out_path}")
    return out_path

if __name__ == "__main__":


    stereo_calibrate_from_checkerboards(
        left_glob="left_camera_ireland/*.jpg",
        right_glob="right_camera_ireland/*.jpg",
        intrinsics_cam1="../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_1.pkl",
        intrinsics_cam2="../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_2.pkl",
        exclude_frames= None,
        output_basename="stereo_calibration_ireland"
    )
