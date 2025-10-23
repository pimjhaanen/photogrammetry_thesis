"""
Fisheye single-camera calibration (OpenCV) using a checkerboard.

- Uses cv2.fisheye.calibrate with 4-coeff fisheye model (k1..k4, no tangential).
- Optionally saves undistorted images using cv2.fisheye.initUndistortRectifyMap.
- Optionally SHOWS the first N undistorted frames at the end to visually verify.
- Prints mean reprojection error.

Tip: make sure the footage mode/resolution during calibration and during use
are IDENTICAL (no EIS, no Linear/Horizon), otherwise metric scale will drift.
"""

import os, glob, pickle
import cv2
import numpy as np


def get_images(patterns):
    files = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)


def calibrate_checkerboard_fisheye(
    image_globs=(
                 'Calibration_Videos_camera_2/checkerboard_wide/*.jpg'),
    checkerboard=(9, 6),             # inner corners: (cols, rows) OR (nx, ny)
    square_size_m: float = 3.9/100.0,
    visualize: bool = False,
    save_undistorted: bool = False,
    show_undistorted: bool = False,   # <--- NEW: preview first N undistorted frames after calib
    show_first_n: int = 10,           # how many to preview
    output_basename: str = 'calibration_fisheye_cam2',
    undist_out_suffix: str = '_undist',
    subpix_win=(11, 11),
    subpix_eps=1e-3,
    subpix_iters=30,
    balance: float = 0.0,            # 0=strong crop, 1=keep full FOV (more blank borders)
    fov_scale: float = 1.0           # 1.0 usually fine
):
    """
    Inputs:
      image_globs        : list/tuple of glob patterns to load calibration frames
      checkerboard       : inner-corner grid size (nx, ny)
      square_size_m      : physical size of one square (meters)
      visualize          : show detected corners live
      save_undistorted   : write undistorted frames using fisheye maps
      show_undistorted   : show first N undistorted frames after calibration
      output_basename    : base name for saved .pkl and undist images
      balance, fov_scale : controls for fisheye new camera matrix (undistortion)

    Returns:
      dict with {camera_matrix, dist_coeffs, reprojection_error, image_size}
      written to 'single_calibration_output/<output_basename>.pkl'
    """

    out_dir = 'single_calibration_output'
    os.makedirs(out_dir, exist_ok=True)

    # Prepare object points (Z=0 plane) for one board view
    nx, ny = checkerboard
    objp = np.zeros((1, nx*ny, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= float(square_size_m)

    # Criteria
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     int(subpix_iters), float(subpix_eps))

    objpoints, imgpoints = [], []

    images = get_images(image_globs)
    if not images:
        print(f"[fisheye] No images matched: {image_globs}")
        return None

    gray_shape = None
    win_name = "Fisheye checkerboard detection"

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[fisheye] Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        # Robust flags help on GoPro frames
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), flags)
        if not ret:
            print(f"{fname}: no corners found")
            continue

        # Sub-pixel refinement
        corners = cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), term_criteria)

        # Fisheye expects (1, N, 2) and (1, N, 3) shapes per view
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(1, -1, 2))

        if visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (nx, ny), corners, True)
            disp = cv2.resize(vis, None, fx=0.5, fy=0.5)
            cv2.imshow(win_name, disp)
            cv2.waitKey(1)

        print(f"{fname}: {corners.shape[0]} corners")

    if visualize:
        print("[fisheye] Close the window to continue...")
        while True:
            if cv2.waitKey(100) == 27:
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("[fisheye] No valid detections, aborting.")
        return None

    # Allocate K, D (fisheye has 4 coeffs)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    # Recommended flags for fisheye
    calib_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                   cv2.fisheye.CALIB_CHECK_COND |
                   cv2.fisheye.CALIB_FIX_SKEW)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=gray_shape,   # (w, h)
        K=K, D=D,
        rvecs=None, tvecs=None,
        flags=calib_flags,
        criteria=term_criteria
    )

    # Mean reprojection error (per view) using fisheye.projectPoints
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.fisheye.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, D
        )
        proj = proj.reshape(-1, 2)
        corners = imgpoints[i].reshape(-1, 2)
        err = cv2.norm(corners, proj, cv2.NORM_L2)
        total_err += err * err
        total_pts += proj.shape[0]
    mean_err = np.sqrt(total_err / max(total_pts, 1))
    print(f"[fisheye] RMS reported by calibrate: {rms:.4f} px")
    print(f"[fisheye] Mean reprojection error:    {mean_err:.4f} px")

    # Save calibration
    out_pkl = os.path.join(out_dir, f"{output_basename}.pkl")
    payload = {
        "model": "fisheye",
        "camera_matrix": K,
        "dist_coeffs": D,
        "reprojection_error": float(mean_err),
        "image_size": tuple(map(int, gray_shape))
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"[fisheye] Saved: {out_pkl}")

    # Optional: undistort all inputs with fisheye maps (save to disk)
    if save_undistorted or show_undistorted:
        # Precompute map once per size (assumes all images same size)
        w, h = gray_shape
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=balance, fov_scale=fov_scale
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1
        )

    if save_undistorted:
        undist_dir = os.path.join(out_dir, f"{output_basename}{undist_out_suffix}")
        os.makedirs(undist_dir, exist_ok=True)
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            out_path = os.path.join(undist_dir, os.path.basename(fname))
            cv2.imwrite(out_path, und)
        print(f"[fisheye] Wrote undistorted images to: {undist_dir}")

    # Optional: SHOW first N undistorted frames for visual check
    if show_undistorted:
        print("[fisheye] Showing first undistorted frames. Press any key to step, ESC to quit.")
        win = "Undistorted preview"
        for i, fname in enumerate(images[:max(0, int(show_first_n))]):
            img = cv2.imread(fname)
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            disp = und if max(und.shape[:2]) <= 1200 else cv2.resize(und, None, fx=0.5, fy=0.5)
            cv2.imshow(win, disp)
            k = cv2.waitKey(0)
            if k == 27:  # ESC
                break
        cv2.destroyAllWindows()

    return payload


if __name__ == "__main__":
    calibrate_checkerboard_fisheye(
        visualize=False,            # show corner detection while collecting
        save_undistorted=False,     # write undistorted images to disk
        show_undistorted=True,      # <--- set True to preview first 10 undistorted frames
        show_first_n=10,
        output_basename='calibration_fisheye_wide_cam2',
        balance=0.0,
        fov_scale=1.0,
    )
