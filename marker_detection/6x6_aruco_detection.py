import cv2
import numpy as np

# ==== SETTINGS ====
video_files = [
    "wide_straight_1.MP4",
    "wide_straight_2.MP4",
    "wide_edge.MP4",
    "linear_straight_1.MP4",
    "linear_straight_2.MP4",
    "linear_edge.MP4"
]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
font = cv2.FONT_HERSHEY_SIMPLEX
frame_step = 5  # Check every 5th frame

results = {}

for video_path in video_files:
    print(f"\n🚀 Processing {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        results[video_path] = None
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total frames: {total_frames}")

    min_detected_size = float('inf')
    frame_index = 0

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {frame_index}")
            frame_index += frame_step
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for marker_corners in corners:
                marker_corners = marker_corners[0]
                x_coords = marker_corners[:, 0]
                y_coords = marker_corners[:, 1]
                width = int(np.max(x_coords) - np.min(x_coords))
                height = int(np.max(y_coords) - np.min(y_coords))
                size = max(width, height)
                min_detected_size = min(min_detected_size, size)

            print(f"✅ Frame {frame_index}: Marker size ~ {size} px")
        else:
            print(f"❌ Frame {frame_index}: No marker detected")

        frame_index += frame_step

    cap.release()

    results[video_path] = None if min_detected_size == float('inf') else min_detected_size

# ==== FINAL REPORT ====
print("\n📊 Detection Summary:")
for video, size in results.items():
    if size is None:
        print(f"{video}: ❌ No marker detected")
    else:
        print(f"{video}: ✅ Smallest detected marker size = {size} px")

cv2.destroyAllWindows()
