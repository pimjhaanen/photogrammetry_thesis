import cv2
import numpy as np
import os

# === CONFIGURATION ===
baseline_cm = 43.0
focal_length_px = 1460
left_dir = "video_left_camera/extracted_frames"
right_dir = "video_right_camera/extracted_frames"
output_dir = "time_series"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "point_cloud_time_series.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

def detect_marker_centers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    centers = {}
    if ids is not None:
        for i, c in enumerate(corners):
            center = c[0].mean(axis=0)
            centers[int(ids[i][0])] = center
    return centers

def triangulate_points(xL, xR, yL, yR, width, height):
    disparity = xL - xR
    if disparity == 0:
        return None
    Z = (focal_length_px * baseline_cm) / disparity
    X = (xL - width / 2) * Z / focal_length_px
    Y = (yL - height / 2) * Z / focal_length_px
    X -= baseline_cm / 2
    return (X, Y, Z)

# === MAIN LOOP ===
left_files = sorted(f for f in os.listdir(left_dir) if f.endswith(".jpg"))
right_files = sorted(f for f in os.listdir(right_dir) if f.endswith(".jpg"))

n = min(len(left_files), len(right_files))
frame_count = 0
total_points = 0
all_data = []

for i in range(n):
    frame_left = os.path.join(left_dir, left_files[i])
    frame_right = os.path.join(right_dir, right_files[i])

    imgL = cv2.imread(frame_left)
    imgR = cv2.imread(frame_right)

    if imgL is None or imgR is None:
        continue

    height, width = imgL.shape[:2]
    centersL = detect_marker_centers(imgL)
    centersR = detect_marker_centers(imgR)
    common_ids = set(centersL.keys()) & set(centersR.keys())

    for mid in common_ids:
        xL, yL = centersL[mid]
        xR, yR = centersR[mid]
        pos = triangulate_points(xL, xR, yL, yR, width, height)
        if pos:
            all_data.append([i, pos[0], pos[1], pos[2]])
            total_points += 1

# === SAVE TO .NPY ===
if all_data:
    full_data = np.array(all_data, dtype=np.float32)
    np.save(output_path, full_data)
    print(f"Saved {len(full_data)} points to: {output_path}")
else:
    print("No points triangulated.")
