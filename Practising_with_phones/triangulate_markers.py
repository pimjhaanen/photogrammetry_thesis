import cv2
import numpy as np
import os

# CONFIGURATION
baseline_cm = 43.0  # distance between cameras in cm
focal_length_px = 1460  # rough estimate for iPhone cameras at 1080p (in pixels)
frame_left = "video_left_camera/extracted_frames/frame_0000.jpg"
frame_right = "video_right_camera/extracted_frames/frame_0000.jpg"

# Load images
imgL = cv2.imread(frame_left)
imgR = cv2.imread(frame_right)

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

def detect_marker_centers(image, name="image"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw markers and their IDs
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Show the image with markers drawn
    scale = 0.5
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.imshow(name, resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute and return marker centres
    centers = {}
    if ids is not None:
        for i, c in enumerate(corners):
            center = c[0].mean(axis=0)
            centers[ids[i][0]] = center
    return centers

centersL = detect_marker_centers(imgL, "Left Camera")
centersR = detect_marker_centers(imgR, "Right Camera")




# Triangulate matching markers
common_ids = set(centersL.keys()) & set(centersR.keys())
print(f"Common marker IDs: {common_ids}")

results = {}
for mid in common_ids:
    xL, yL = centersL[mid]
    xR, yR = centersR[mid]
    disparity = xL - xR
    if disparity == 0:
        continue  # avoid divide-by-zero
    Z = (focal_length_px * baseline_cm) / disparity
    X = (xL - imgL.shape[1] / 2) * Z / focal_length_px
    Y = (yL - imgL.shape[0] / 2) * Z / focal_length_px
    X -= baseline_cm / 2

    results[int(mid)] = (X, Y, Z)
    print(f"Marker ID {mid}: X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm")

# Optional: save results
import json
with open("triangulated_markers_frame0.json", "w") as f:
    json.dump(results, f, indent=4)


