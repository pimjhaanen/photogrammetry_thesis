import cv2
import numpy as np
import pickle
import os
import glob

# === CONFIG ===
CHECKERBOARD = (9, 6)
square_size = 3.9 / 100.0  # meters
left_folder = "left_camera_wide_84cm"
right_folder = "right_camera_wide_84cm"
stereo_calib_file = "stereo_calibration_wide_84cm.pkl"
calib_cam1 = "../single-camera calibration/calibration_checkerboard_wide_camera_1.pkl"
calib_cam2 = "../single-camera calibration/calibration_checkerboard_wide_camera_2.pkl"

# === Load stereo calibration ===
with open(stereo_calib_file, "rb") as f:
    stereo_data = pickle.load(f)

# Check if the rectification data exists
if 'stereoRectify' not in stereo_data:
    print("[INFO] Rectification data not found. Computing rectification matrices.")

    # Compute stereo rectification if it's not available in the calibration file
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        stereo_data["camera_matrix_1"], stereo_data["dist_coeffs_1"],
        stereo_data["camera_matrix_2"], stereo_data["dist_coeffs_2"],
        (640, 480), stereo_data["R"], stereo_data["T"], alpha=0
    )
    R_total = stereo_data["R"]
    T_total = stereo_data["T"].flatten()

    # Save the rectification data back to the stereo calibration file
    stereo_data["stereoRectify"] = (R1, R2, P1, P2, Q)
    with open(stereo_calib_file, "wb") as f:
        pickle.dump(stereo_data, f)
    print("[INFO] Rectification data saved to the stereo calibration file.")
else:
    # Load the rectification data if it exists
    R1, R2, P1, P2, Q = stereo_data["stereoRectify"]
    print("[INFO] Loaded rectification data from the stereo calibration file.")


# === Load intrinsic calibrations ===
def load_calib(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["camera_matrix"], data["dist_coeffs"]


mtx1, dist1 = load_calib(calib_cam1)
mtx2, dist2 = load_calib(calib_cam2)

# === Object points ===
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

left_images = sorted(glob.glob(os.path.join(left_folder, "*.jpg")))
right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))
assert len(left_images) == len(right_images), "Mismatch in left/right image counts"


# Function to calculate the epipolar line distance for matching points
def calculate_epipolar_distance(cornersL, cornersR, F):
    """
    Calculate the distance between matching epipolar lines for corresponding points.
    Returns the average distance.
    """
    total_distance = 0
    n_points = len(cornersL)

    for i in range(n_points):
        # Get the corresponding points in both images
        x1, y1 = cornersL[i].ravel()
        x2, y2 = cornersR[i].ravel()

        # Calculate the epipolar line for the point in the right image
        line1 = F @ np.array([x1, y1, 1]).T  # Epipolar line in right image

        # Compute the distance from the point in the left image to the epipolar line
        distance = abs(line1[0] * x2 + line1[1] * y2 + line1[2]) / np.sqrt(line1[0] ** 2 + line1[1] ** 2)

        total_distance += distance

    # Return the average distance
    return total_distance / n_points


def rodrigues_angle_diff(R1, R2):
    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)
    return np.linalg.norm(rvec1 - rvec2)


print(f"\n🔍 Comparing each pair to total calibration...")

# Process each pair of images
for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if not (retL and retR):
        print(f"[{i:02d}] ❌ No corners found")
        continue

    cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

    try:
        ret, _, _, _, _, R_i, T_i, _, _ = cv2.stereoCalibrate(
            [objp], [cornersL], [cornersR],
            mtx1, dist1,
            mtx2, dist2,
            grayL.shape[::-1],
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        # Calculate differences in translation and rotation
        t_diff = np.linalg.norm(T_total - T_i.flatten())
        r_diff = rodrigues_angle_diff(R_total, R_i)

        # Compute the Fundamental matrix
        F, _ = cv2.findFundamentalMat(cornersL, cornersR, method=cv2.FM_8POINT)

        # Calculate the epipolar distance for the current frame
        epipolar_distance = calculate_epipolar_distance(cornersL, cornersR, F)

        print(
            f"[{i:02d}] ✅ reproj: {ret:.4f} | ΔT: {t_diff:.4f} | ΔR: {r_diff:.4f} rad | Epipolar Dist: {epipolar_distance:.4f}")
    except cv2.error as e:
        print(f"[{i:02d}] ❌ Calibration failed: {e}")
