import cv2
import numpy as np

# === SETTINGS ===
video_path = "wide_red_dots.MP4"  # Change to your file
frame_step = 10
min_area = 60     # Area for ~8.7 px diameter
max_area = 5000

# === Blob Detector Setup ===
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False  # Colour filtering is handled manually
params.filterByArea = True
params.minArea = 1
params.maxArea = max_area
params.filterByCircularity = True
params.minCircularity = 0.75
params.filterByInertia = False
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞️ Total frames: {total_frames}")
frame_index = 0
min_detected_diameter = float('inf')

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

    # Blur for better blob detection
    red_mask_blurred = cv2.GaussianBlur(red_mask, (5, 5), 0)

    # Detect blobs
    keypoints = detector.detect(red_mask_blurred)

    if keypoints:
        sizes = []
        frame_h, frame_w = frame.shape[:2]
        x_min = 0
        x_max =  frame_w // 3
        y_min = frame_h // 3
        y_max = 2 * frame_h // 3

        for kp in keypoints:
            x, y = kp.pt
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue  # skip if not in center region

            diameter = kp.size

            sizes.append(int(diameter))
            min_detected_diameter = min(min_detected_diameter, diameter)
            cv2.circle(frame, (int(x), int(y)), int(diameter / 2), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(diameter)}px", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"✅ {len(sizes)} red blob(s) detected in frame {frame_index}: {sizes}")
    else:
        print(f"❌ No red blobs detected in frame {frame_index}")

    # Show frame
    resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    cv2.imshow("Red Blob Detection", resized)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    frame_index += frame_step

cap.release()
cv2.destroyAllWindows()

# === REPORT ===
if min_detected_diameter < float('inf'):
    print(f"\n✅ Smallest detected red circle diameter: {min_detected_diameter:.1f} px")
else:
    print("\n❌ No red blobs detected at all.")
