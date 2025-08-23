import cv2
import numpy as np
import csv

# ==== SETTINGS ====
video_files = [
    "wide_straight_1.MP4",
    "wide_straight_2.MP4",
    "wide_edge.MP4",
    "linear_straight_1.MP4",
    "linear_straight_2.MP4",
    "linear_edge.MP4"
]

aruco_types = {
    "4x4": cv2.aruco.DICT_4X4_100,
    "5x5": cv2.aruco.DICT_5X5_100,
    "6x6": cv2.aruco.DICT_6X6_100,
    "7x7": cv2.aruco.DICT_7X7_100,
}

frame_step = 5
circle_min_area = 6024
circle_max_area = 5000

# ==== FUNCTIONS ====

def detect_aruco(video_path, aruco_dict_type):
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.001
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minDistanceToBorder = 1

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total frames: {total_frames}")

    seen_sizes = {}
    frame_index = 0

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {frame_index}")
            frame_index += frame_step
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None:
            for marker_corners in corners:
                pts = marker_corners[0]
                width = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                height = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                size = max(width, height)

                seen_sizes[size] = seen_sizes.get(size, 0) + 1
                print(f"✅ Frame {frame_index}: Marker size ~ {size} px")
        else:
            print(f"❌ Frame {frame_index}: No marker detected")

        frame_index += frame_step

    cap.release()

    for size, count in sorted(seen_sizes.items()):
        if count >= 4:
            return size

    print("⚠️ No marker size occurred at least twice.")
    return None



def detect_black_circle(video_path):
    cap = cv2.VideoCapture(video_path)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = circle_min_area
    params.maxArea = circle_max_area
    params.filterByCircularity = True
    params.minCircularity = 0.9
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    seen_sizes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        keypoints = detector.detect(blur)

        for kp in keypoints:
            size = int(kp.size)
            seen_sizes[size] = seen_sizes.get(size, 0) + 1

        for _ in range(frame_step - 1):
            cap.read()

    cap.release()

    for size, count in sorted(seen_sizes.items()):
        if count >= 4:
            return size
    return None

# ==== MAIN ====

results = []
"""
for marker_type, dict_id in aruco_types.items():
    print(f"\n🔍 Testing ArUco {marker_type}")
    for video in video_files:
        print(f"   → {video}")
        min_size = detect_aruco(video, dict_id)
        results.append([marker_type, video, min_size if min_size is not None else "Not detected"])

# Black circle blob detection
print(f"\n🔍 Testing Black Circle Detection")
for video in video_files:
    print(f"   → {video}")
    min_size = detect_black_circle(video)
    results.append(["circle", video, min_size if min_size is not None else "Not detected"])
"""
# ==== SAVE TO CSV ====
# === SETTINGS ===

frame_step = 10
min_area = 30     # roughly 6 px diameter → area ≈ 28.3
max_area = 5000   # just to ignore very large blobs

# === Blob Detector Setup ===
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0  # Black blobs
params.filterByArea = True
params.minArea = min_area
params.maxArea = max_area
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByInertia = False
params.filterByConvexity = False
params.minArea = 30


detector = cv2.SimpleBlobDetector_create(params)
for video_path in video_files:
    # === LOAD VIDEO ===
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        exit()
    print(print(f"   → {video_path}"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total frames: {total_frames}")
    frame_index = 500
    min_detected_diameter = float('inf')

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {frame_index}")
            frame_index += frame_step
            continue

        print(f"🔍 Processing frame {frame_index}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        keypoints = detector.detect(blurred)

        if keypoints:
            sizes = []
            for kp in keypoints:
                x, y = kp.pt
                diameter = kp.size
                sizes.append(int(diameter))
                min_detected_diameter = min(min_detected_diameter, diameter)
                cv2.circle(frame, (int(x), int(y)), int(diameter / 2), (0, 255, 0), 2)
                cv2.putText(frame, f"{int(diameter)}px", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"✅ {len(sizes)} blob(s) detected in frame {frame_index}: {sizes} pixels")
        else:
            print(f"❌ No blobs detected in frame {frame_index}")

        # Show frame
        resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
        cv2.imshow("Blob Detection", resized)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        frame_index += frame_step

    cap.release()
    cv2.destroyAllWindows()

# === REPORT ===
if min_detected_diameter < float('inf'):
    print(f"\n✅ Smallest detected circle diameter: {min_detected_diameter:.1f} px")
else:
    print("\n❌ No blobs detected at all.")


print("\n✅ All done. Results saved to detection_results.csv")
