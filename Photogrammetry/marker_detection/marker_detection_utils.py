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

def apply_lr_gradient_gain(
    img_bgr: np.ndarray,
    bright_side_gain=(4.0, 20.0),   # (alpha, beta) for the brighter half
    dark_side_gain=(5.0, 25.0),     # (alpha, beta) for the darker half
    bands: int = 5,                 # 1 = smooth ramp; >1 = step-wise bands
    image: str = "left",            # Whether its the left or right frame
    smooth: bool = True  ,          # if True: smooth ramp even when bands>1
    show_rio: bool = True          # if True: shows ROI for brightness consideration
) -> np.ndarray:
    """
    Apply a left<->right gradient brightness/contrast on HSV V to handle uneven lighting.
    Bright vs dark side is decided from a fixed ROI: x=[450,1850], y=[600,800].
    The brighter side uses bright_side_gain; the darker side uses dark_side_gain.
    Also draws the ROI, the midline, and 'B'/'D' on the corresponding halves.
    """
    h, w = img_bgr.shape[:2]

    # --- decide bright/dark using the fixed rectangular ROI on V channel ---

    x0, x1 = 450*2, 1850*2
    y0, y1 = 600*2, 800*2
    if image=='left': #estimated disparity, different block
        x0+=300
        x1+=300

    # clamp to image bounds
    x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h))

    hsv_tmp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V_all   = hsv_tmp[..., 2]

    if x1 <= x0 or y1 <= y0:
        # fallback to halves if ROI invalid
        mean_left  = float(np.median(V_all[:, :w//2])) if w >= 2 else float(np.median(V_all))
        mean_right = float(np.median(V_all[:, w//2:])) if w >= 2 else float(np.median(V_all))
        mid_x = w // 2
        roi_ok = False
    else:
        mid_x = (x0 + x1) // 2
        V_left  = V_all[y0:y1, x0:mid_x]
        V_right = V_all[y0:y1, mid_x:x1]
        mean_left  = float(np.median(V_left))  if V_left.size  else float(np.median(V_all))
        mean_right = float(np.median(V_right)) if V_right.size else float(np.median(V_all))
        roi_ok = True

    # assign endpoint gains left->right
    left_is_brighter = (mean_left >= mean_right)
    if left_is_brighter:
        a_left,  b_left  = bright_side_gain
        a_right, b_right = dark_side_gain
    else:
        a_left,  b_left  = dark_side_gain
        a_right, b_right = bright_side_gain

    # --- build horizontal alpha/beta maps ---
    if smooth or bands <= 1:
        t_line = np.linspace(0.0, 1.0, w, dtype=np.float32)        # smooth ramp
    else:
        # step-wise bands: t = i/(bands-1) per band (bands=2 => t=0 left, t=1 right)
        edges    = np.linspace(0.0, 1.0, bands + 1, dtype=np.float32)
        t_values = np.linspace(0.0, 1.0, bands,     dtype=np.float32)
        x        = np.linspace(0.0, 1.0, w,         dtype=np.float32)
        band_idx = np.digitize(x, edges[1:-1], right=False)
        t_line   = t_values[band_idx]

    alpha_line = a_left  + (a_right  - a_left)  * t_line
    beta_line  = b_left  + (b_right  - b_left)  * t_line
    alpha_map  = np.tile(alpha_line, (h, 1)).astype(np.float32)
    beta_map   = np.tile(beta_line,  (h, 1)).astype(np.float32)

    # --- apply on V only (preserve hue/sat) ---
    Vf    = V_all.astype(np.float32)
    V_out = np.clip(Vf * alpha_map + beta_map, 0, 255).astype(np.uint8)
    hsv_out = hsv_tmp.copy()
    hsv_out[..., 2] = V_out
    out_bgr = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    # --- draw ROI, midline, and labels on the returned frame ---
    # yellow lines
    line_color = (0, 255, 255)
    if roi_ok and show_rio:
        cv2.rectangle(out_bgr, (x0, y0), (x1, y1), line_color, 2)
        cv2.line(out_bgr, (mid_x, y0), (mid_x, y1), line_color, 2)

        # label positions (roughly centered in each half of the ROI)
        cx_left  = (x0 + mid_x) // 2
        cx_right = (mid_x + x1) // 2
        cy_text  = y0 + int(0.25 * (y1 - y0))  # upper quarter inside ROI

        # Draw big black boxes behind the letters for visibility
        def put_big_label(img, text, pos):
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
            (tw, th), base = cv2.getTextSize(text, font, scale, thick)
            tx, ty = pos
            pad = 6
            cv2.rectangle(img,
                          (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + base + pad),
                          (0, 0, 0), -1)
            cv2.putText(img, text, (tx, ty), font, scale, line_color, thick, cv2.LINE_AA)

        if left_is_brighter:
            put_big_label(out_bgr, "B", (cx_left - 15,  cy_text))
            put_big_label(out_bgr, "D", (cx_right - 15, cy_text))
        else:
            put_big_label(out_bgr, "D", (cx_left - 15,  cy_text))
            put_big_label(out_bgr, "B", (cx_right - 15, cy_text))
    else:
        # fallback annotation at image mid if ROI invalid
        cv2.line(out_bgr, (w//2, 0), (w//2, h-1), line_color, 2)
        txt = "B|D" if left_is_brighter else "D|B"
        cv2.putText(out_bgr, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, line_color, 3, cv2.LINE_AA)

    return out_bgr

def detect_crosses(frame, debug_mask=False, show_annot=False, min_area=60, min_dist=50):
    """
    Detect red crosses and return only the kept (deduplicated) centres, an annotated image,
    and print a table where K-IDs match the labels drawn on the debug frame.

    Returns:
      centres : List[(cx, cy)] for kept detections (after dedup)
      annotated_frame : BGR image with K-IDs drawn (same shape as input)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- red mask (wrap-around near 0/180) ---
    lower_red_1 = np.array([0,   100,  45], dtype=np.uint8)
    upper_red_1 = np.array([6, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([172, 110,  45], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)

    # --- morphology ---
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if debug_mask:
        msmall = cv2.resize(mask, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("HSV Mask", msmall); cv2.waitKey(0); cv2.destroyAllWindows()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = frame.shape[:2]
    candidates = []  # diagnostics for all candidates BEFORE dedup

    for raw_id, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0 or w <= 0:
            continue

        ar = float(w) / float(h)
        if not (0.5 < ar < 2.2):
            continue

        # shape cues
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = float(area) / hull_area
        extent   = float(area) / float(w * h)

        # center
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # mean HSV within the contour
        cnt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_bgr = cv2.mean(frame, mask=cnt_mask)[:3]
        mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]  # [H,S,V]

        candidates.append({
            "raw_id": raw_id,
            "center": (cx, cy),
            "area": float(area),
            "bbox_wh": (int(w), int(h)),
            "aspect_ratio": float(ar),
            "mean_hsv": (int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])),
            "solidity": float(solidity),
            "extent": float(extent),
        })

    # --- deduplicate by distance (greedy) on candidate centers ---
    def _dedup_by_distance(points, r=min_dist):
        kept = []
        taken = np.zeros(len(points), dtype=bool)
        r2 = r * r
        for i, p in enumerate(points):
            if taken[i]:
                continue
            kept.append(i)
            (x0, y0) = p
            # mark neighbors as taken
            for j in range(i + 1, len(points)):
                if taken[j]:
                    continue
                (x1, y1) = points[j]
                if (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) <= r2:
                    taken[j] = True
        return kept

    centres_all = [c["center"] for c in candidates]
    kept_idx = _dedup_by_distance(centres_all, r=min_dist)

    # Prepare annotated output and centres list
    annotated = frame.copy()
    centres = []

    # ---- print only kept detections so numbers match the overlay ----
    print(f"[detect_crosses] kept {len(kept_idx)} / {len(contours)} (post filters: {len(candidates)})")
    for kept_id, ci in enumerate(kept_idx):
        d = candidates[ci]
        centres.append(d["center"])
        cx, cy = d["center"]

        # draw cross
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # --- yellow text on black box ---
        label = f"K{kept_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        # text anchor (top-left of text box)
        tx = cx + 6
        ty = cy - 6

        # black rectangle background (with small padding)
        pad = 3
        x1 = max(0, tx - pad)
        y1 = max(0, ty - th - pad)
        x2 = min(annotated.shape[1] - 1, tx + tw + pad)
        y2 = min(annotated.shape[0] - 1, ty + baseline + pad)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

        # yellow text
        cv2.putText(annotated, label, (tx, ty), font, scale, (0, 255, 255), thick, cv2.LINE_AA)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optional quick view; remove if you don't want a window
    if show_annot:
        # print row
        print(f"K{kept_id:02d} (raw #{d['raw_id']:02d}) @ {d['center']}  "
              f"area={d['area']:.1f}  wh={d['bbox_wh'][0]}x{d['bbox_wh'][1]}  "
              f"ar={d['aspect_ratio']:.2f}  HSV={d['mean_hsv']}  "
              f"sol={d['solidity']:.2f}  ext={d['extent']:.2f}")
        preview = cv2.resize(annotated, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        cv2.imshow("Cross detections (K-IDs match console)", preview)
        cv2.waitKey(1)

    return centres, annotated

def detect_crosses_accuracy_test(frame, debug=False):
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
        if 10 < area :
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.3 < aspect_ratio < 3.0:
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
