import cv2
import numpy as np

# ==== SETTINGS ====
video_files = [
    "GX010255.MP4"
]
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.minMarkerPerimeterRate = 0.001
parameters.maxMarkerPerimeterRate = 4.0
parameters.minDistanceToBorder = 1

font = cv2.FONT_HERSHEY_SIMPLEX
frame_step = 30
results = {}

# ==== VIDEO LOOP ====
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
    frame_index = 3090

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
            for i, marker_corners in enumerate(corners):
                marker_corners = marker_corners[0]
                x_coords = marker_corners[:, 0]
                y_coords = marker_corners[:, 1]
                width = int(np.max(x_coords) - np.min(x_coords))
                height = int(np.max(y_coords) - np.min(y_coords))
                size = max(width, height)
                min_detected_size = min(min_detected_size, size)

                # Draw marker outline and ID
                cv2.polylines(frame, [np.int32(marker_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"Size: {size}px", (int(x_coords[0]), int(y_coords[0]) - 10),
                            font, 0.6, (0, 255, 0), 2)

            print(f"✅ Frame {frame_index}: Marker size ~ {size} px")
        else:
            print(f"❌ Frame {frame_index}: No marker detected")
            cv2.putText(frame, "No marker detected", (30, 30), font, 0.7, (0, 0, 255), 2)

        display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # Show the resized frame
        cv2.imshow("ArUco Detection", display_frame)
        key = cv2.waitKey(300)  # Wait 300 ms
        if key == 27:  # ESC to break
            break

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
