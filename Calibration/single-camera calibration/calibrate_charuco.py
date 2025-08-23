import cv2
import numpy as np
import glob
import pickle

# Define the ArUco dictionary and ChArUco board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard_create(
    squaresX=5,
    squaresY=7,
    squareLength=5.55 / 100.0,
    markerLength=2.8 / 100.0,
    dictionary=aruco_dict
)

# Arrays to store detected corners and IDs
all_corners = []
all_ids = []

# Load images
images = glob.glob('Calibration_Videos_camera_2/charuco_wide/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    num_aruco = 0
    num_charuco = 0

    if ids is not None:
        num_aruco = len(ids)
        # Try to get ChArUco corners
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        if charuco_ids is not None and len(charuco_ids) > 4:
            num_charuco = len(charuco_ids)
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

        # Always draw ArUco markers if present
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        resized_img = cv2.resize(img, None, fx=0.35, fy=0.35)
        cv2.imshow("Detected Corners", resized_img)

        while True:
            if cv2.getWindowProperty("Detected Corners", cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.waitKey(100)

    else:
        print(f"{fname}: No ArUco markers found")

    if num_aruco > 0 or num_charuco > 0:
        print(f"{fname}: {num_aruco} ArUco markers, {num_charuco} ChArUco corners detected")
    else:
        print(f"{fname}: No useful markers or corners detected")

cv2.destroyAllWindows()

# Check if any valid ChArUco corners were found
if all_corners and all_ids:
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=gray.shape[::-1],  # Use the size of the last processed image
        cameraMatrix=None,
        distCoeffs=None
    )

    # Calculate reprojection error for ChArUco corners
    mean_error = 0
    valid_sets = 0

    for i in range(len(all_corners)):
        if all_corners[i] is None or all_ids[i] is None or len(all_ids[i]) < 1:
            print(f"Skipping invalid charuco set at index {i}")
            continue

        # Get corresponding 3D object points for the charuco_ids
        obj_points = board.chessboardCorners[all_ids[i].flatten()].reshape(-1, 1, 3)  # (N, 1, 3)
        img_points = all_corners[i].reshape(-1, 1, 2)  # (N, 1, 2)

        # Project the 3D object points
        imgpoints2, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], mtx, dist)

        # Compute the error
        error = cv2.norm(img_points, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        valid_sets += 1

    if valid_sets > 0:
        mean_error /= valid_sets
        print("Reprojection Error for ChArUco Calibration:", mean_error)
    else:
        print("No valid sets for reprojection error calculation.")

    # Save calibration data and reprojection error to pickle
    calibration_data = {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "reprojection_error": mean_error
    }
    with open("calibration_charuco_wide_camera_2.pkl", "wb") as f:
        pickle.dump(calibration_data, f)

    print("Camera matrix:", mtx)
    print("Distortion coefficients:", dist)
else:
    print("No valid ChArUco corners found for calibration.")
