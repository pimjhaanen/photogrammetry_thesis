import cv2
import numpy as np
import glob
import pickle

# Define the pattern size and circle diameter (in meters)
pattern_size = (4, 11)
circle_diameter = 2.5 / 100.0  # Circle diameter in meters

# Prepare the object points based on the pattern size and circle diameter
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp[:, 0] *= 2 * circle_diameter
objp[:, 1] *= circle_diameter
objp[:, 0] += (objp[:, 1] % (2 * circle_diameter)) * 0.5

objpoints = []
imgpoints = []

# Load images
images = glob.glob('Calibration_Videos_camera_2/circles_linear/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the asymmetric circle grid
    params = cv2.SimpleBlobDetector_Params()
    ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        print(f"{fname}: Detected {len(centers)} circles")
        objpoints.append(objp)
        imgpoints.append(centers)

        # Visualize the detected corners
        vis_img = img.copy()
        cv2.drawChessboardCorners(vis_img, pattern_size, centers, ret)

        resized_img = cv2.resize(vis_img, None, fx=0.35, fy=0.35)
        cv2.imshow("Detected Circles", resized_img)

        while True:
            if cv2.getWindowProperty("Detected Circles", cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(100)

    else:
        print(f"{fname}: No circles detected")

cv2.destroyAllWindows()

# Camera calibration
if objpoints and imgpoints:
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
    print("Reprojection Error for Circle Calibration: ", mean_error)

    # Save calibration data to file with reprojection error
    output_file = "calibration_circles_linear_camera_2.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "reprojection_error": mean_error
        }, f)
    print(f"Saved calibration to {output_file}")
else:
    print("No valid calibration data available.")
