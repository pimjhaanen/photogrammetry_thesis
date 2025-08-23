import cv2
import numpy as np
import glob
import pickle
import os

# === CONFIG ===
CHECKERBOARD = (9, 6)
square_size = 3.9 / 100.0  # in meters
left_folder = "left_camera_wide_84cm"
right_folder = "right_camera_wide_84cm"
save_filename = "stereo_calibration_wide_84cm_filtered.pkl"

calib_cam1 = "../single-camera calibration/calibration_checkerboard_wide_camera_1.pkl"
calib_cam2 = "../single-camera calibration/calibration_checkerboard_wide_camera_2.pkl"

exclude_frames = [3, 4, 5, 21, 22, 23, 24, 25]  # ⬅️ modify this list based on your findings

# === Load intrinsic calibrations ===
def load_calib(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["camera_matrix"], data["dist_coeffs"]

mtx1, dist1 = load_calib(calib_cam1)
mtx2, dist2 = load_calib(calib_cam2)

# === Object points in 3D ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

# === Load image pairs ===
left_images = sorted(glob.glob(os.path.join(left_folder, "*.jpg")))
right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))
assert len(left_images) == len(right_images), "Mismatch in number of left/right frames"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# === Loop through image pairs ===
for idx, (l_img_path, r_img_path) in enumerate(zip(left_images, right_images)):
    if idx in exclude_frames:
        print(f"[{idx}] ❌ Skipping excluded frame")
        continue

    imgL = cv2.imread(l_img_path)
    imgR = cv2.imread(r_img_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    print(f"[{idx}] {os.path.basename(l_img_path)}: corners left = {len(cornersL) if retL else 0}, right = {len(cornersR) if retR else 0}")

    if retL and retR:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# === Stereo calibration ===
ret, newmtx1, newdist1, newmtx2, newdist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtx1,
    dist1,
    mtx2,
    dist2,
    grayL.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# === Report ===
print(f"\nStereo calibration RMS reprojection error: {ret:.4f} pixels")
print(f"Rotation matrix R:\n{R}")
print(f"Translation vector T (in meters):\n{T}")
print(f"Essential matrix E:\n{E}")
print(f"Fundamental matrix F:\n{F}")

# === Save to file ===
with open(save_filename, "wb") as f:
    pickle.dump({
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "reprojection_error": ret,
        "camera_matrix_1": newmtx1,
        "dist_coeffs_1": newdist1,
        "camera_matrix_2": newmtx2,
        "dist_coeffs_2": newdist2
    }, f)

print(f"\n✅ Stereo calibration saved to: {save_filename}")
