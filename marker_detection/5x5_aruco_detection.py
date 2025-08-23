import cv2
import numpy as np

# ==== SETTINGS ====
video_path = "wide_straight_2.MP4"
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters_create()
font = cv2.FONT_HERSHEY_SIMPLEX
frame_step = 5  # every 10th frame (≈ every 1/6 second for 30fps)

# ==== VIDEO ====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

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
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for marker_corners in corners:
            marker_corners = marker_corners[0]
            x_coords = marker_corners[:, 0]
            y_coords = marker_corners[:, 1]
            width = int(np.max(x_coords) - np.min(x_coords))
            height = int(np.max(y_coords) - np.min(y_coords))
            size = max(width, height)
            min_detected_size = min(min_detected_size, size)

            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            cv2.putText(frame, f"{size}px", (center_x, center_y), font, 0.6, (0, 255, 0), 2)

        print(f"✅ Frame {frame_index}: Marker size ~ {size} px")
    else:
        print(f"❌ Frame {frame_index}: No marker detected")

    # Resize for display
    #resized_frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    #cv2.imshow("Aruco Detection (sampled)", resized_frame)

    key = cv2.waitKey(100)  # wait for 100 ms
    if key == ord('q'):
        print("🛑 Stopped manually")
        break

    frame_index += frame_step

cap.release()
cv2.destroyAllWindows()

# ==== FINAL REPORT ====
if min_detected_size < float('inf'):
    print(f"\n✅ Smallest detected marker size in sampled frames: {min_detected_size} pixels")
else:
    print("\n❌ No marker detected in any sampled frame.")
