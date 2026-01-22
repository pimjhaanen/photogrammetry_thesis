import os
import glob
import pickle
import cv2
import numpy as np


def get_images(folder: str):
    files = []
    files += glob.glob(os.path.join(folder, "*.jpg"))
    files += glob.glob(os.path.join(folder, "*.jpeg"))
    files += glob.glob(os.path.join(folder, "*.png"))
    return sorted(files)


def load_intrinsics(pkl_path: str):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d["camera_matrix"], d["dist_coeffs"]


def load_stereo(stereo_pkl: str):
    with open(stereo_pkl, "rb") as f:
        d = pickle.load(f)
    return d


def put_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def find_chessboard_corners(img_bgr, checkerboard=(9, 6)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, flags)
    if not ret:
        return False, None, gray

    # subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners, gray


def draw_epipolar_through_corners_rectified(img_bgr, corners, thickness=2):
    """
    After rectification, epipolar lines are horizontal.
    For each detected corner, draw its epipolar line at y = corner_y.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    ys = np.round(corners.reshape(-1, 2)[:, 1]).astype(int)
    ys = np.clip(ys, 0, h - 1)

    # avoid drawing the same y 54 times (cluster to unique y values)
    unique_ys = np.unique(ys)

    for y in unique_ys:
        cv2.line(out, (0, y), (w, y), (0, 255, 255), thickness)

    # also draw the corners so it's clear what we're aligning
    for (x, y) in corners.reshape(-1, 2):
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)  # red dots

    return out


def main():
    stereo_calib_pkl = "stereo_calibration_output/final_stereo_calibration_V3.pkl"
    intrinsics_cam1  = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_1.pkl"
    intrinsics_cam2  = "../single-camera calibration/single_calibration_output/calibration_checkerboard_wide_camera_2.pkl"
    left_folder  = "left_camera_wide_84cm"
    right_folder = "right_camera_wide_84cm"

    # 0-based index in sorted file list
    IDX = 5

    # checkerboard inner corners (columns, rows)
    CHECKERBOARD = (9, 6)

    out_root = "stereo_export_frame_005_split_epi_on_corners"
    ensure_dir(out_root)

    # Stage folders (split by camera)
    stages = ["a_raw", "b_undistort", "c_rectified", "c_rectified_epipolar", "d_cropped"]
    for st in stages:
        ensure_dir(os.path.join(out_root, st, "left"))
        ensure_dir(os.path.join(out_root, st, "right"))

    left_images  = get_images(left_folder)
    right_images = get_images(right_folder)

    if len(left_images) == 0 or len(right_images) == 0:
        raise RuntimeError("No images found in left/right folders.")
    if len(left_images) != len(right_images):
        raise RuntimeError(f"Mismatch: {len(left_images)} left vs {len(right_images)} right.")
    if IDX >= len(left_images):
        raise RuntimeError(f"IDX={IDX} out of range (have {len(left_images)} pairs).")

    stereo = load_stereo(stereo_calib_pkl)
    K1, D1 = load_intrinsics(intrinsics_cam1)
    K2, D2 = load_intrinsics(intrinsics_cam2)

    R = stereo["R"]
    T = stereo["T"]

    # image size from first image
    sample = cv2.imread(left_images[0])
    if sample is None:
        raise RuntimeError(f"Could not read first left image: {left_images[0]}")
    h, w = sample.shape[:2]
    img_size = (w, h)

    # Rectification params
    if "stereoRectify" in stereo:
        sr = stereo["stereoRectify"]
        if len(sr) >= 5:
            R1, R2, P1, P2, Q = sr[:5]
        else:
            raise RuntimeError(f"stereoRectify has unexpected length: {len(sr)}")
    else:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, img_size, R, T, alpha=0
        )

    # maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    # undistort-only
    newK1, _ = cv2.getOptimalNewCameraMatrix(K1, D1, img_size, 0)
    newK2, _ = cv2.getOptimalNewCameraMatrix(K2, D2, img_size, 0)

    # Read raw pair
    lp, rp = left_images[IDX], right_images[IDX]
    imgL_raw = cv2.imread(lp)
    imgR_raw = cv2.imread(rp)
    if imgL_raw is None or imgR_raw is None:
        raise RuntimeError(f"Could not read IDX={IDX}:\n  L={lp}\n  R={rp}")

    base = f"frame_{IDX:03d}"

    # (a) RAW
    cv2.imwrite(os.path.join(out_root, "a_raw", "left",  f"{base}.png"),
                put_label(imgL_raw, f"(a) Raw | {base}"))
    cv2.imwrite(os.path.join(out_root, "a_raw", "right", f"{base}.png"),
                put_label(imgR_raw, f"(a) Raw | {base}"))

    # (b) UNDISTORT
    L_und = cv2.undistort(imgL_raw, K1, D1, None, newK1)
    R_und = cv2.undistort(imgR_raw, K2, D2, None, newK2)
    cv2.imwrite(os.path.join(out_root, "b_undistort", "left",  f"{base}.png"),
                put_label(L_und, f"(b) Undistort | {base}"))
    cv2.imwrite(os.path.join(out_root, "b_undistort", "right", f"{base}.png"),
                put_label(R_und, f"(b) Undistort | {base}"))

    # (c) RECTIFY
    L_rec = cv2.remap(imgL_raw, map1x, map1y, cv2.INTER_LINEAR)
    R_rec = cv2.remap(imgR_raw, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_root, "c_rectified", "left",  f"{base}.png"),
                put_label(L_rec, f"(c) Rectified | {base}"))
    cv2.imwrite(os.path.join(out_root, "c_rectified", "right", f"{base}.png"),
                put_label(R_rec, f"(c) Rectified | {base}"))

    # Detect corners on RECTIFIED images, then draw epipolar lines exactly through those corner y's
    okL, cornersL, _ = find_chessboard_corners(L_rec, CHECKERBOARD)
    okR, cornersR, _ = find_chessboard_corners(R_rec, CHECKERBOARD)

    if not okL or not okR:
        print("[WARN] Chessboard corners not found on rectified images. "
              "Falling back to drawing no epipolar lines.")
        L_epi = L_rec.copy()
        R_epi = R_rec.copy()
    else:
        L_epi = draw_epipolar_through_corners_rectified(L_rec, cornersL, thickness=2)
        R_epi = draw_epipolar_through_corners_rectified(R_rec, cornersR, thickness=2)

    cv2.imwrite(os.path.join(out_root, "c_rectified_epipolar", "left",  f"{base}.png"),
                put_label(L_epi, f"(c) Rectified + epipolar (through corners) | {base}"))
    cv2.imwrite(os.path.join(out_root, "c_rectified_epipolar", "right", f"{base}.png"),
                put_label(R_epi, f"(c) Rectified + epipolar (through corners) | {base}"))

    # (d) CROP (use common valid region so left/right crop match)
    maskL = cv2.cvtColor(L_rec, cv2.COLOR_BGR2GRAY) > 0
    maskR = cv2.cvtColor(R_rec, cv2.COLOR_BGR2GRAY) > 0
    common = maskL & maskR
    ys, xs = np.where(common)

    if len(xs) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        L_crop = L_rec[y0:y1 + 1, x0:x1 + 1]
        R_crop = R_rec[y0:y1 + 1, x0:x1 + 1]
    else:
        L_crop, R_crop = L_rec, R_rec

    cv2.imwrite(os.path.join(out_root, "d_cropped", "left",  f"{base}.png"),
                put_label(L_crop, f"(d) Cropped | {base}"))
    cv2.imwrite(os.path.join(out_root, "d_cropped", "right", f"{base}.png"),
                put_label(R_crop, f"(d) Cropped | {base}"))

    print("[DONE] Exported split images for IDX=5 with epipolar lines THROUGH the detected checkerboard corners.")
    print("Output folder:", out_root)
    print("Left raw:", lp)
    print("Right raw:", rp)


if __name__ == "__main__":
    main()
