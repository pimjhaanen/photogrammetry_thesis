import cv2
import numpy as np
import pickle
import glob
import os

# === Load previously saved calibration ===
with open("stereo_calibration_wide_54cm_combined.pkl", "rb") as f:
    data = pickle.load(f)

R, T = data["R"], data["T"]
mtx1, dist1 = data["camera_matrix_1"], data["dist_coeffs_1"]
mtx2, dist2 = data["camera_matrix_2"], data["dist_coeffs_2"]
objpoints = []  # fill in below
imgpoints_left = []  # fill in below
imgpoints_right = []  # fill in below

# === Reload original data used for stereo calibration ===

# === CONFIG ===
CHECKERBOARD = (9, 6)
square_size = 3.9 / 100.0  # in meters
left_folder = "left_camera/54cm_wide"
right_folder = "right_camera/54cm_wide"
left_folder_2 = "left_camera_1_wide_54cm"
right_folder_2 = "right_camera_2_wide_54cm"
save_filename = "stereo_calibration_wide_54cm_combined_refined.pkl"

calib_cam1 = "../single-camera calibration/calibration_checkerboard_wide_camera_1.pkl"
calib_cam2 = "../single-camera calibration/calibration_checkerboard_wide_camera_2.pkl"

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
# === Load image pairs ===
left_images_1 = sorted(glob.glob(os.path.join(left_folder, "*.jpg")))
right_images_1 = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))

left_images_2 = sorted(glob.glob(os.path.join(left_folder_2, "*.jpg")))
right_images_2 = sorted(glob.glob(os.path.join(right_folder_2, "*.jpg")))

print(len(left_images_1), len( left_images_2), len(right_images_1), len(right_images_2))

# Combine both sets
left_images = left_images_1 + left_images_2
right_images = right_images_1 + right_images_2

assert len(left_images) == len(right_images), "Mismatch in number of left/right frames"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for l_img_path, r_img_path in zip(left_images, right_images):
    imgL = cv2.imread(l_img_path)
    imgR = cv2.imread(r_img_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    print(f"{os.path.basename(l_img_path)}: corners left = {len(cornersL) if retL else 0}, right = {len(cornersR) if retR else 0}")

    if retL and retR:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# === Per-pair reprojection error ===
pair_errors = []

for i in range(len(objpoints)):
    # === LEFT ===
    retval_L, rvec_L, tvec_L = cv2.solvePnP(objpoints[i], imgpoints_left[i], mtx1, dist1)
    proj_L, _ = cv2.projectPoints(objpoints[i], rvec_L, tvec_L, mtx1, dist1)
    err_L = cv2.norm(imgpoints_left[i], proj_L, cv2.NORM_L2) / len(proj_L)

    # === RIGHT ===
    retval_R, rvec_R, tvec_R = cv2.solvePnP(objpoints[i], imgpoints_right[i], mtx2, dist2)
    proj_R, _ = cv2.projectPoints(objpoints[i], rvec_R, tvec_R, mtx2, dist2)
    err_R = cv2.norm(imgpoints_right[i], proj_R, cv2.NORM_L2) / len(proj_R)

    pair_error = (err_L + err_R) / 2
    pair_errors.append(pair_error)

    print(f"Pair {i+1}: Left = {err_L:.3f} px, Right = {err_R:.3f} px, Avg = {pair_error:.3f} px")

import matplotlib.pyplot as plt
plt.hist(pair_errors, bins=20)
plt.title("Per-pair reprojection error distribution")
plt.xlabel("Avg reprojection error (px)")
plt.ylabel("Number of image pairs")
plt.grid(True)
plt.show()

# === Reject worst pairs
mean_error = np.mean(pair_errors)
threshold =  .5* mean_error

print(f"\n🔍 Mean reprojection error per pair: {mean_error:.4f}")
print(f"📉 Rejecting pairs with error > {threshold:.4f}")

filtered_objpoints = []
filtered_imgpoints_L = []
filtered_imgpoints_R = []

for i, err in enumerate(pair_errors):
    if err <= threshold:
        filtered_objpoints.append(objpoints[i])
        filtered_imgpoints_L.append(imgpoints_left[i])
        filtered_imgpoints_R.append(imgpoints_right[i])
    else:
        print(f"❌ Removing pair {i+1} (error: {err:.2f})")

# === Recalibrate with filtered data ===
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, _, _, _, _, R_new, T_new, E_new, F_new = cv2.stereoCalibrate(
    filtered_objpoints,
    filtered_imgpoints_L,
    filtered_imgpoints_R,
    mtx1,
    dist1,
    mtx2,
    dist2,
    (5120, 2880),  # replace with your image size (w,h)
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=criteria
)

print(f"\nStereo calibration RMS reprojection error: {ret:.4f} pixels")
print(f"Rotation matrix R:\n{R_new}")
print(f"Translation vector T (in meters):\n{T_new}")

# === Save refined calibration ===
with open("stereo_calibration_wide_54cm_refined.pkl", "wb") as f:
    pickle.dump({
        "R": R_new,
        "T": T_new,
        "E": E_new,
        "F": F_new,
        "reprojection_error": ret,
        "camera_matrix_1": mtx1,
        "dist_coeffs_1": dist1,
        "camera_matrix_2": mtx2,
        "dist_coeffs_2": dist2
    }, f)

print(f"\n✅ Refined stereo calibration saved.")
print(f"New stereo RMS reprojection error: {ret:.4f}")
