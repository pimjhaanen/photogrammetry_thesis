"""This code can be used to perform a single-camera calibration using a chequerboard.
It uses chequerboard corner detection for robust calibration.
You can adapt board dimensions, square size, and marker size to your printed board."""

import os, glob, pickle
import cv2
import numpy as np

def get_images(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)

def calibrate_checkerboard(
    image_glob: str = 'Calibration_Videos_camera_1/checkerboard_wide/*.png',
    checkerboard=(9, 6),
    square_size_m: float = 3.8767/100.0,
    visualize: bool = False,
    save_undistorted: bool = False,
    output_basename: str = 'calibration_checkerboard_wide_camera_1_revised',
    undist_out_suffix: str = '_undist'
):

    """RELEVANT FUNCTION INPUTS:
    - image_glob: folder containing calibration frames (e.g. frame_0001.jpg, …)
    - checkerboard: number of squares in the chequerboard (rows, cols)
    - square_size_m: physical side length of each square in meters
    - visualize: if True, shows detected markers/corners on the images
    - save_undistorted: if True, saves undistorted images
    - output_basename: filename (without extension) for the saved calibration results
    - undist_out_suffix: defines a suffix for the saved undistorted images (if True)"""

    out_dir = 'single_calibration_output'
    os.makedirs(out_dir, exist_ok=True)

    # Prepare object points (Z=0 plane)
    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints, imgpoints = [], []
    images = get_images(["Calibration_Videos_camera_1/checkerboard_wide/*.png",
                              "Calibration_Videos_camera_1/checkerboard_wide/*.jpg"])

    if not images:
        print(f"[checkerboard] No images matched: {image_glob}")
        return None

    gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[checkerboard] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            msg = f"{fname}: {len(corners2)} corners"
            if visualize:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, checkerboard, corners2, True)
                cv2.imshow("Checkerboard detection", cv2.resize(vis, None, fx=0.35, fy=0.35))
                cv2.waitKey(1)
        else:
            msg = f"{fname}: No corners found"

        print(msg)

    if visualize:
        print("[checkerboard] Close preview window to continue...")
        while True:
            if cv2.getWindowProperty("Checkerboard detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) == 27:
                break
        cv2.destroyAllWindows()

    if not objpoints:
        print("[checkerboard] No valid detections, aborting.")
        return None

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # Reprojection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += err
    mean_error /= len(objpoints)
    print("[checkerboard] Reprojection error:", mean_error)

    # Save calibration
    out_pkl = os.path.join(out_dir, f"{output_basename}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}, f)
    print(f"[checkerboard] Saved: {out_pkl}")

    # Optional undistorted export
    if save_undistorted:
        undist_dir = os.path.join(out_dir, f"{output_basename}{undist_out_suffix}")
        os.makedirs(undist_dir, exist_ok=True)
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            h, w = img.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
            und = cv2.undistort(img, mtx, dist, None, newK)
            out_path = os.path.join(undist_dir, os.path.basename(fname))
            cv2.imwrite(out_path, und)
        print(f"[checkerboard] Wrote undistorted images to: {undist_dir}")

    return {"camera_matrix": mtx, "dist_coeffs": dist, "reprojection_error": mean_error}

if __name__ == "__main__":
    calibrate_checkerboard(visualize=True, save_undistorted=False)
