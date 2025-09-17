"""This code can be used to detect circular blobs and red cross markers, refine their sub-pixel centers,
and estimate ArUco marker poses for camera pose alignment. It combines HSV masking, blob detection,
corner/gaussian sub-pixel refinement, and ArUco pose estimation to support photogrammetry workflows."""

import cv2
import numpy as np
from scipy.optimize import curve_fit

# Global list for storing clicked points in full-resolution
clicked_points = []

def detect_circles_blob(frame, colour, draw_on=None):
    """RELEVANT FUNCTION INPUTS:
    - frame: BGR image (H×W×3) to search for circular blobs.
    - colour: target color class to mask; one of {"red", "blue"} with predefined HSV ranges.
    - draw_on: optional BGR image to draw detections on; if None, a copy of `frame` is used.
    """
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
    """RELEVANT FUNCTION INPUTS:
    - frame: input BGR image containing ArUco markers.
    - mtx: 3×3 intrinsic camera matrix from calibration.
    - dist: distortion coefficients matching `mtx` (e.g., k1..k6, p1, p2).
    - marker_length: physical marker side length (in meters) used for pose scaling.
    """
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
    """RELEVANT FUNCTION INPUTS:
    - image: BGR image to visualize with overlaid cross markers.
    - cross_points: iterable of (x, y) pixel locations to draw as crosses.
    - label: window title for the debug view.
    - scale: display downscale factor for the shown window (0<scale<=1).
    """
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
    """RELEVANT FUNCTION INPUTS:
    - centres: list of 2D points (x, y) or (x, y, r) to be deduplicated.
    - min_dist: minimum Euclidean distance (in pixels) required between unique points.
    """
    unique = []
    for pt in centres:
        if all(np.linalg.norm(np.array(pt) - np.array(u)) > min_dist for u in unique):
            unique.append(pt)
    return unique


def detect_crosses(frame, debug=False):
    """RELEVANT FUNCTION INPUTS:
    - frame: BGR image that may contain red cross-shaped markers.
    - debug: if True, shows intermediate HSV mask and masked frame windows.
    """
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
    """RELEVANT FUNCTION INPUTS:
    - frame: BGR image containing circular markers.
    - circles: iterable of detected circles as (x, y, r) tuples in pixel units.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    centers = []
    for (x, y, r) in circles:
        center = np.array([[x, y]], dtype=np.float32)
        cv2.cornerSubPix(gray, center, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        centers.append((center[0][0], center[0][1]))
    return centers


def gaussian_2d(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """RELEVANT FUNCTION INPUTS:
    - xy_tuple: (X, Y) meshgrid arrays covering the ROI (from np.meshgrid).
    - amplitude: peak amplitude parameter of the 2D Gaussian.
    - xo, yo: centroid coordinates of the Gaussian (in ROI pixel coordinates).
    - sigma_x, sigma_y: standard deviations along x and y axes (pixels).
    - offset: constant background intensity added to the Gaussian.
    """
    x, y = xy_tuple
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()


def get_subpixel_centers_gaussian(frame, circles):
    """RELEVANT FUNCTION INPUTS:
    - frame: BGR image containing circular markers.
    - circles: iterable of detected circles as (x, y, r) tuples in pixel units;
               each circle defines the ROI used for Gaussian fitting.
    """
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
