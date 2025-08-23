import cv2
import numpy as np
from scipy.optimize import curve_fit
from Accuracy_analysis.Triangulation_and_matching import sort_markers_gridwise

# Global list for storing clicked points in full-resolution
clicked_points = []

def click_marker(event, x, y, flags, param):
    global clicked_points
    scale, full_image = param["scale"], param["original"]

    if event == cv2.EVENT_LBUTTONDOWN:
        # Rescale clicked (x, y) to original resolution
        x_full = int(x / scale)
        y_full = int(y / scale)
        print(f"Clicked at ({x_full}, {y_full})")
        clicked_points.append((x_full, y_full))

        # Draw on the resized image (for feedback)
        img = param["resized"]
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(img, str(len(clicked_points)), (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Click Markers", img)

def collect_initial_labels(frame, expected_n=8, label="Left", scale=0.3):
    """
    Show a resized frame and collect marker positions, returning full-res coordinates.
    """
    global clicked_points
    clicked_points = []

    resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    clone = resized.copy()

    cv2.imshow("Click Markers", clone)
    param = {
        "scale": scale,
        "original": frame,
        "resized": clone
    }
    cv2.setMouseCallback("Click Markers", click_marker, param)

    print(f"[{label}] Click on {expected_n} markers in order, then press any key when done.")

    while True:
        cv2.imshow("Click Markers", clone)
        key = cv2.waitKey(1)
        if key != -1 and len(clicked_points) == expected_n:
            break

    cv2.destroyWindow("Click Markers")
    return clicked_points



def initialize_marker_tracking(frame_left, frame_right, expected_n=8):
    print("[INIT] Left frame:")
    markers_left = collect_initial_labels(frame_left, expected_n=expected_n, label="Left")

    print("[INIT] Right frame:")
    markers_right = collect_initial_labels(frame_right, expected_n=expected_n, label="Right")

    return markers_left, markers_right

def initialize_marker_tracking_auto(cross_left_raw, cross_right_raw, expected_n=8):
    """
    Automatically initialise marker tracking by sorting gridwise.
    Assumes that marker layout is consistent between views.
    """
    if len(cross_left_raw) != expected_n or len(cross_right_raw) != expected_n:
        print("[ERROR] Not enough markers for auto initialisation.")
        return None, None

    print("[INIT] Automatically assigning IDs based on grid layout")
    sorted_left = sort_markers_gridwise(cross_left_raw)
    sorted_right = sort_markers_gridwise(cross_right_raw)
    return sorted_left, sorted_right



def track_markers(prev_markers, current_detections, max_dist=100):
    matched = []
    print(f"[DEBUG] Raw detections this frame: {len(current_detections)}")
    print(f"[DEBUG] Attempting to track {len(prev_markers)} markers to {len(current_detections)} detections")
    for prev in prev_markers:
        dists = [np.linalg.norm(np.array(prev) - np.array(det)) for det in current_detections]
        if not dists:
            matched.append(None)
            continue
        min_idx = np.argmin(dists)
        if dists[min_idx] < max_dist:
            matched.append(current_detections[min_idx])
        else:
            matched.append(None)
    return matched if all(m is not None for m in matched) else None


def draw_tracked_markers(image, markers, color=(0, 255, 255), label="Tracked", scale=0.3):
    img = image.copy()
    for i, pt in enumerate(markers):
        pt_int = tuple(map(int, pt))
        cv2.drawMarker(img, pt_int, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(img, str(i + 1), (pt_int[0] + 5, pt_int[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(label, img_small)
    cv2.waitKey(1)


def detect_circles_blob(frame, colour, draw_on=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if colour == "red":
        mask = cv2.inRange(hsv, (172, 130, 140), (179, 190, 230))
        min_size_px = 6
    elif colour == "blue":
        mask = cv2.inRange(hsv, (112, 90, 90), (128, 190, 180))
        min_size_px = 3
    else:
        return [], frame

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blurred)

    image_draw = frame.copy() if draw_on is None else draw_on.copy()
    centers = []

    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = float(kp.size)
        if size >= min_size_px:
            centers.append((kp.pt[0], kp.pt[1], size))

        cv2.circle(image_draw, (x, y), int(size / 2), (0, 255, 0), 2)
        cv2.putText(image_draw, f"{i+1}: {int(size)}px", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return centers, image_draw


def detect_aruco_pose(frame, mtx, dist, marker_length=0.15):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.minMarkerPerimeterRate = 0.05
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.perspectiveRemovePixelPerCell = 6

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        print(f"Detected ArUco IDs: {ids.flatten()}")
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
        return corners, rvecs, tvecs
    else:
        print("No ArUco markers detected.")
        return None, None, None


def draw_debug_window(image, cross_points, label="Crosses", scale=0.3):
    img_copy = image.copy()
    for pt in cross_points:
        pt_int = tuple(map(int, pt))
        cv2.drawMarker(img_copy, pt_int, (0, 255, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=30, thickness=2)
    img_small = cv2.resize(img_copy, (0, 0), fx=scale, fy=scale)
    cv2.imshow(label, img_small)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        return img_copy, False
    return img_copy, True

def deduplicate_centers(centres, min_dist=10):
    unique = []
    for pt in centres:
        if all(np.linalg.norm(np.array(pt) - np.array(u)) > min_dist for u in unique):
            unique.append(pt)
    return unique

def detect_crosses(frame, debug=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Updated thresholds based on your clicked HSV values
    lower_red_1 = np.array([0, 100, 150])    # Lower boundary for red (part 1)
    upper_red_1 = np.array([10, 255, 255])   # Upper boundary for red (part 1)

    lower_red_2 = np.array([170, 100, 150])  # Lower boundary for red (part 2)
    upper_red_2 = np.array([179, 255, 255])  # Upper boundary for red (part 2)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    if debug:
        cv2.imshow("HSV Mask", mask)
        cv2.imshow("Masked Frame", masked_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 40 < area < 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.0:
                pad = 2
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w + pad, gray.shape[1])
                y2 = min(y + h + pad, gray.shape[0])
                gray_patch = gray[y1:y2, x1:x2]

                corners = cv2.goodFeaturesToTrack(
                    gray_patch, maxCorners=4, qualityLevel=0.01, minDistance=5
                )

                cx = int(x + w / 2)
                cy = int(y + h / 2)
                centres.append((cx, cy))

    centres = deduplicate_centers(centres, min_dist=100)

    debug2 = False
    if debug2:
        debug_img = frame.copy()
        print(f"[DEBUG] Detected {len(centres)} crosses")
        return centres, draw_debug_window(debug_img, centres, label="crosses", scale=0.3)[0]
    else:
        return centres, frame.copy()


def get_subpixel_centers_corner(frame, circles):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    centers = []
    for (x, y, r) in circles:
        center = np.array([[x, y]], dtype=np.float32)
        cv2.cornerSubPix(gray, center, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        centers.append((center[0][0], center[0][1]))
    return centers


def gaussian_2d(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy_tuple
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()


def get_subpixel_centers_gaussian(frame, circles):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    centers = []
    for (x, y, r) in circles:
        x1 = int(max(x - r, 0))
        y1 = int(max(y - r, 0))
        x2 = int(min(x + r, gray.shape[1]))
        y2 = int(min(y + r, gray.shape[0]))
        roi = gray[y1:y2, x1:x2]

        if roi.size < 9:
            continue

        try:
            x_indices = np.arange(roi.shape[1])
            y_indices = np.arange(roi.shape[0])
            x_mesh, y_mesh = np.meshgrid(x_indices, y_indices)
            initial_guess = (np.max(roi), roi.shape[1] / 2, roi.shape[0] / 2, 3, 3, np.min(roi))
            popt, _ = curve_fit(gaussian_2d, (x_mesh, y_mesh), roi.ravel(), p0=initial_guess)
            xo_fit = popt[1] + x1
            yo_fit = popt[2] + y1
            centers.append((xo_fit, yo_fit))
        except Exception:
            continue

    return centers
