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

import cv2
import numpy as np
from typing import List, Tuple

def subpixel_cross_center(gray, approx_xy, patch=25, canny=(50,150), hough_thresh=25, min_len=8, max_gap=3):
    """
    Refine a cross center by line-fitting in a small ROI.
    gray: grayscale image
    approx_xy: (x,y) integer-ish seed (e.g., from contour bbox center)
    Returns: (xc, yc) float or None if fail.
    """
    import numpy as np, cv2
    x0, y0 = map(int, approx_xy)
    h, w = gray.shape[:2]
    x1 = max(0, x0 - patch); y1 = max(0, y0 - patch)
    x2 = min(w, x0 + patch + 1); y2 = min(h, y0 + patch + 1)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return None

    edges = cv2.Canny(roi, canny[0], canny[1])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None or len(lines) < 2:
        return None

    # Fit two dominant orientations (cluster by angle)
    segs = []
    for l in lines[:,0,:]:
        xA,yA,xB,yB = map(float, l)
        dx, dy = (xB-xA), (yB-yA)
        if dx==0 and dy==0: continue
        ang = np.arctan2(dy, dx)
        segs.append((ang, xA,yA,xB,yB))
    if len(segs) < 2: return None

    # Normalize angles to [0,pi) and cluster into two groups ~ orthogonal
    angs = np.array([ (a%(np.pi)) for a, *_ in segs ])
    # pick first angle, split by closeness vs ~orthogonal
    a0 = angs[0]
    grp1 = [s for s,a in zip(segs, angs) if abs((a-a0+np.pi/2)%(np.pi)-np.pi/2) < np.pi/4]
    grp2 = [s for s,a in zip(segs, angs) if abs((a-a0+np.pi/2)%(np.pi)-np.pi/2) >= np.pi/4]
    if len(grp1)==0 or len(grp2)==0:
        grp1, grp2 = segs[:len(segs)//2], segs[len(segs)//2:]

    def fit_line(points):
        # total least squares line fit: returns (vx,vy,xc,yc) as in fitLine
        pts = np.array(points, float)
        vx, vy, xc, yc = cv2.fitLine(pts, cv2.DIST_L2, 0, 1e-2, 1e-2)
        return float(vx), float(vy), float(xc), float(yc)

    # Collect endpoints and fit one line per group
    p1 = []; p2 = []
    for _,xA,yA,xB,yB in grp1: p1 += [(xA,yA), (xB,yB)]
    for _,xA,yA,xB,yB in grp2: p2 += [(xA,yA), (xB,yB)]
    vx1,vy1,xc1,yc1 = fit_line(p1)
    vx2,vy2,xc2,yc2 = fit_line(p2)

    # Line intersection (in ROI coords): p = p1 + t * v1 ; p = p2 + s * v2
    A = np.array([[vx1, -vx2],[vy1, -vy2]], float)
    b = np.array([xc2 - xc1, yc2 - yc1], float)
    det = np.linalg.det(A)
    if abs(det) < 1e-9:  # nearly parallel
        return None
    t, s = np.linalg.solve(A, b)
    xi = xc1 + t*vx1
    yi = yc1 + t*vy1

    # Map back to full image coords
    return (xi + x1, yi + y1)


def _fit_line_through_points(xy: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = a*x + b via least squares. xy: (N,2)
    Returns (a, b). Handles verticals by swapping axes at caller.
    """
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def _line_intersection(a1: float, b1: float, a2: float, b2: float) -> Tuple[float, float]:
    """Intersect y=a1 x + b1 and y=a2 x + b2."""
    denom = (a1 - a2)
    if abs(denom) < 1e-9:
        return np.nan, np.nan
    x = (b2 - b1) / denom
    y = a1 * x + b1
    return float(x), float(y)

def _angle_of_segment(p0, p1) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return float(np.degrees(np.arctan2(dy, dx)))

def _cluster_by_angle(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster angles into two bins ~90° apart. Returns boolean masks (cluster1, cluster2).
    Use k-means on angle modulo 180 to avoid wrap issues.
    """
    ang = angles.copy()
    ang = (ang + 180.0) % 180.0  # map to [0,180)
    data = ang.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel().astype(bool)
    # make first cluster the one with smaller center (just to be deterministic)
    if centers[0, 0] > centers[1, 0]:
        labels = ~labels
        centers = centers[[1, 0]]
    return labels, ~labels

def _refine_with_subpix(gray_patch: np.ndarray, p: Tuple[float, float]) -> Tuple[float, float]:
    """One-pt cornerSubPix refinement around p."""
    x, y = p
    win = (5, 5)
    zero_zone = (-1, -1)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray_patch, pts, win, zero_zone, term)
    return float(pts[0, 0, 0]), float(pts[0, 0, 1])

def detect_crosses(frame: np.ndarray, debug: bool = False) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Detect red crosses and return sub-pixel centers by intersecting two dominant arms.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red ranges – yours, kept relatively tight
    lower_red_1 = np.array([0, 20, 30],  dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 20, 30], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    centres: List[Tuple[float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (20 < area < 7000):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0.0
        if not (0.3 < aspect < 3.0):
            continue

        pad = 4
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame.shape[1])
        y2 = min(y + h + pad, frame.shape[0])

        patch_gray = gray[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]

        # Edges for Hough
        edges = cv2.Canny(patch_mask, 50, 150, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=12, minLineLength=max(6, min(w, h)//3), maxLineGap=3)

        cx, cy = (x + w / 2.0, y + h / 2.0)  # fallback
        refined = subpixel_cross_center(gray, (cx, cy), patch=25)
        if refined is not None:
            cx, cy = refined
        centres.append((float(cx), float(cy)))

        ok = False
        if lines is not None and len(lines) >= 2:
            segs = []
            angs = []
            ptsA = []
            ptsB = []

            for l in lines[:, 0, :]:
                xA, yA, xB, yB = map(int, l.tolist())
                segs.append(((xA, yA), (xB, yB)))
                angs.append(_angle_of_segment((xA, yA), (xB, yB)))
                ptsA.append([xA, yA])
                ptsB.append([xB, yB])

            angs = np.array(angs, dtype=np.float32)
            c1, c2 = _cluster_by_angle(angs)

            def fit_from_cluster(mask_sel: np.ndarray):
                sel = np.where(mask_sel)[0]
                if sel.size < 2:
                    return None
                pts = []
                for idx in sel:
                    (xa, ya), (xb, yb) = segs[idx]
                    # densify a bit to stabilize LS fit
                    pts += [[xa, ya], [xb, yb]]
                pts = np.asarray(pts, np.float32)
                # choose axis to reduce vertical ill-conditioning:
                # try both fits and pick the one with smaller residual
                a1, b1 = _fit_line_through_points(pts)  # y = a1 x + b1
                # vertical-esque alternative: x = ay + b  -> convert to y = (x - b)/a
                a2, b2 = _fit_line_through_points(pts[:, ::-1])  # swap axes
                # compute residuals quickly and pick better
                y_hat1 = a1 * pts[:, 0] + b1
                r1 = np.mean((pts[:, 1] - y_hat1) ** 2)
                x_hat2 = a2 * pts[:, 1] + b2
                y_hat2 = (pts[:, 0] - b2) / (a2 + 1e-12)
                r2 = np.mean((pts[:, 1] - y_hat2) ** 2)
                if r1 <= r2:
                    return ("y=ax+b", a1, b1)
                else:
                    # convert x = a*y + b -> y = (x - b)/a
                    return ("y=(x-b)/a", a2, b2)

            L1 = fit_from_cluster(c1)
            L2 = fit_from_cluster(c2)

            if L1 and L2:
                # Compute intersection in patch coords
                # normalize both to y = a*x + b form
                def to_ab(L):
                    if L[0] == "y=ax+b":
                        return L[1], L[2]
                    else:
                        a2, b2 = L[1], L[2]
                        a = 1.0 / (a2 + 1e-12)
                        b = -b2 / (a2 + 1e-12)
                        return a, b
                a1_, b1_ = to_ab(L1)
                a2_, b2_ = to_ab(L2)
                theta1 = np.degrees(np.arctan(a1_))
                theta2 = np.degrees(np.arctan(a2_))
                angle_diff = abs(((theta1 - theta2 + 90) % 180) - 90)  # shortest mod-180 diff

                if 70 <= angle_diff <= 110:
                    px, py = _line_intersection(a1_, b1_, a2_, b2_)
                    if np.isfinite(px) and np.isfinite(py):
                        # ensure inside patch
                        if 0 <= px < (x2 - x1) and 0 <= py < (y2 - y1):
                            # sub-pixel refine on grayscale
                            rx, ry = _refine_with_subpix(patch_gray, (px, py))
                            cx = x1 + rx
                            cy = y1 + ry
                            ok = True

        if not ok:
            # soft PCA fallback on skeleton-ish pixels
            pts_bin = cv2.findNonZero(patch_mask)
            if pts_bin is not None and len(pts_bin) > 20:
                pts = pts_bin.reshape(-1, 2).astype(np.float32)
                # coarse center as mean, refine with cornerSubPix
                m = pts.mean(axis=0)
                rx, ry = _refine_with_subpix(patch_gray, (m[0], m[1]))
                cx = x1 + rx
                cy = y1 + ry
            # else keep bbox center fallback

        centres.append((float(cx), float(cy)))

    # de-duplicate nearby centers
    centres = deduplicate_centers(centres, min_dist=60)  # keep your helper

    if debug:
        dbg = frame.copy()
        for (cx, cy) in centres:
            cv2.drawMarker(dbg, (int(round(cx)), int(round(cy))), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.imshow("crosses", cv2.resize(dbg, None, fx=0.4, fy=0.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return centres, frame.copy()


def detect_crosses2(frame, debug=False):
    """RELEVANT FUNCTION INPUTS:
    - frame: BGR image that may contain red cross-shaped markers.
    - debug: if True, shows intermediate HSV mask and masked frame windows.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Updated thresholds based on your clicked HSV values
    lower_red_1 = np.array([0, 20, 30], dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([165, 20, 30], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

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
        mask = cv2.resize(mask, None, fx=0.3, fy=0.3)
        masked_frame = cv2.resize(masked_frame, None, fx=0.3, fy=0.3)
        cv2.imshow("HSV Mask", mask)
        cv2.imshow("Masked Frame", masked_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 7000:
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
                refined = subpixel_cross_center(gray, (cx, cy), patch=25)
                if refined is not None:
                    cx, cy = refined
                centres.append((float(cx), float(cy)))

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
