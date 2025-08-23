import cv2
import numpy as np

# === SETTINGS ===
video_path = "linear_straight_crosses.MP4"  # Your video path
frame_step = 10
min_area = 60
max_area = 5000

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞️ Total frames: {total_frames}")
frame_index = 0
min_detected_size = float('inf')

while frame_index < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ Could not read frame {frame_index}")
        frame_index += frame_step
        continue

    print(f"🔍 Processing frame {frame_index}")

    # Convert to HSV and isolate red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = frame.shape[:2]
    sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])


        size = int(np.sqrt(area))  # Estimate "diameter"
        min_detected_size = min(min_detected_size, size)
        sizes.append(size)

        # Draw
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(frame, f"{size}px", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Reporting
    if sizes:
        print(f"✅ {len(sizes)} red cross(es) detected in frame {frame_index}: {sizes}")
    else:
        print(f"❌ No red crosses detected in frame {frame_index}")

    # Show debug output

    resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
    cv2.imshow("Red Cross Detection", resized)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    frame_index += frame_step

cap.release()
cv2.destroyAllWindows()

# === FINAL REPORT ===
if min_detected_size < float('inf'):
    print(f"\n✅ Smallest detected red cross size: {min_detected_size:.1f} px")
else:
    print("\n❌ No red crosses detected at all.")
