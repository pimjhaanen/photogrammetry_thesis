import cv2
import numpy as np
import glob
import pickle

# Checkerboard dimensions
CHECKERBOARD = (9, 6)
square_size = 3.9 / 100.0  # cm to meters

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

objpoints = []
imgpoints = []

images = glob.glob('Calibration_Videos_camera_1/checkerboard_linear/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

    print(f"{fname}: {len(corners)} corners detected" if corners is not None else f"{fname}: No markers found")
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

    resized_img = cv2.resize(img, None, fx=0.35, fy=0.35)
    #cv2.imshow("Detected Corners", resized_img)
    while True:
        if cv2.getWindowProperty("Detected Corners", cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(100)

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix:", mtx)
print("Distortion coefficients:", dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)
print("Reprojection Error for Checkerboard Calibration: ", mean_error)

# Save calibration data and reprojection error to pickle
output_file = "calibration_checkerboard_linear_camera_1.pkl"
with open(output_file, "wb") as f:
    pickle.dump({
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "reprojection_error": mean_error
    }, f)
print(f"Saved calibration to {output_file}")

# Undistort images
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    resized_undistorted = cv2.resize(undistorted, None, fx=0.35, fy=0.35)
    cv2.imshow("Undistorted", resized_undistorted)
    while True:
        if cv2.getWindowProperty("Undistorted", cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(100)

cv2.destroyAllWindows()
