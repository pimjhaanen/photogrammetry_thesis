import cv2
from Accuracy_analysis.marker_detection_functions import detect_circles_blob, detect_aruco_pose, detect_crosses, get_subpixel_centers_corner, get_subpixel_centers_gaussian
from Accuracy_analysis.Triangulation_and_matching import triangulate_points, sort_markers_gridwise
from Accuracy_analysis.plotting_utils import transform_to_aruco_frame
from scipy.spatial import cKDTree
import pandas as pd
from scipy.spatial import cKDTree
from Accuracy_analysis.Triangulation_and_matching import match_stereo_points
import numpy as np
import numpy as np
from Accuracy_analysis.marker_detection_functions import (
    detect_crosses
)
from Accuracy_analysis.Triangulation_and_matching import triangulate_points

# Store previous matched positions globally or manage externally
prev_blue_left = None
prev_blue_right = None
prev_cross_left = None
prev_cross_right = None
n_pairs = 40
len_initial_matches = 41
# KLT parameters
lk_params = dict(winSize=(31, 31),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.03))

# Global states for KLT tracking
prev_gray_left = None
prev_gray_right = None
tracked_cross_left_klt = None
tracked_cross_right_klt = None


def _coords_only(pts):
    # [(x,y)] or [(x,y,label)] -> [(x,y)]
    if not pts:
        return []
    return [(float(p[0]), float(p[1])) for p in pts]

def track_klt(prev_img, next_img, prev_pts):
    """Track points using optical flow (KLT) and keep associated labels."""
    if prev_pts is None or len(prev_pts) == 0:
        return [], None

    # Check if input has labels
    if isinstance(prev_pts[0], (list, tuple)) and len(prev_pts[0]) == 3:
        prev_pts_xy = [pt[:2] for pt in prev_pts]
        labels = [pt[2] for pt in prev_pts]
    else:
        prev_pts_xy = prev_pts
        labels = [None] * len(prev_pts)  # If no labels

    prev_pts_np = np.array(prev_pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_pts_np, None, **lk_params)

    # Keep only successfully tracked points
    good_next = next_pts[status == 1].reshape(-1, 2).tolist()
    good_labels = [lbl for lbl, s in zip(labels, status.flatten()) if s == 1]

    # Combine position and label
    tracked = [(pt[0], pt[1], lbl) for pt, lbl in zip(good_next, good_labels)]

    return tracked, good_next


# Global cross tracking state
initialized_cross_tracking = False
tracked_cross_left = None
tracked_cross_right = None

# Optional: global blue tracking (if you add similar logic later)
prev_blue_left = None
prev_blue_right = None



def separate_LE_and_struts(cross_centres, gray_img, patch_size=20, horizontal_spacing=200):
    """
    Label cross points as 'LE' (leading edge) or numbered struts (0–7) based on background brightness.
    Returns list of (x, y, label) tuples.
    """
    background_brightness = []

    # Step 1: Estimate brightness for each point (excluding red cross)
    cross_centres = [(c[0], c[1]) for c in cross_centres]  # tolerate (x,y,label)

    for (cx, cy) in cross_centres:
        # Small square patch around the cross
        x1 = max(int(cx - patch_size // 2), 0)
        y1 = max(int(cy - patch_size // 2), 0)
        x2 = min(int(cx + patch_size // 2), gray_img.shape[1])
        y2 = min(int(cy + patch_size // 2), gray_img.shape[0])

        patch = gray_img[y1:y2, x1:x2]
        mean_brightness = np.mean(patch)
        background_brightness.append((cx, cy, mean_brightness))

    # Step 2: Compute global average background brightness
    avg_brightness = np.mean([b for _, _, b in background_brightness])

    # Step 3: Separate into LE and struts
    LE_points = []
    strut_candidates = []
    for cx, cy, brightness in background_brightness:
        if brightness > 1.2 * avg_brightness:
            LE_points.append((cx, cy, "LE"))
        else:
            strut_candidates.append((cx, cy))

    # Step 4: Group struts by horizontal clusters
    strut_candidates.sort(key=lambda p: p[0])  # sort by x-position
    strut_clusters = []
    current_cluster = []

    for pt in strut_candidates:
        if not current_cluster:
            current_cluster.append(pt)
        else:
            first_x = current_cluster[0][0]
            if abs(pt[0] - first_x) < horizontal_spacing:

                current_cluster.append(pt)
            else:
                strut_clusters.append(current_cluster)
                current_cluster = [pt]

    if current_cluster:
        strut_clusters.append(current_cluster)

    # Step 5: Assign strut numbers
    labeled_struts = []
    for i, cluster in enumerate(strut_clusters):
        strut_id = str(min(i, 7))  # cap at 7
        for cx, cy in cluster:
            labeled_struts.append((cx, cy, strut_id))

    return LE_points + labeled_struts


def process_stereo_pair(left, right, calib, frame_counter, max_displacement=10):
    """
    Processes a stereo frame pair:
      - Rectifies for ArUco and cross detection
      - Detects ArUco (4x4, 7x7) and cross markers
      - Initializes or updates KLT tracking for crosses
      - (Optional) Debug overlay every 30th frame
      - Triangulates 3D points for crosses and ArUco centers

    Returns (unchanged signature/order):
      cross_3d, labels, aruco_3d_4by4, aruco_ids_4x4,
      aruco_3d_7by7, aruco_ids_7x7, aruco_rotations_7x7,
      aruco_translations_7x7, frame_counter + 1
    """
    # Globals used by the calling code
    global initialized_cross_tracking, tracked_cross_left, tracked_cross_right
    global prev_blue_left, prev_blue_right
    global prev_gray_left, prev_gray_right, tracked_cross_left_klt, tracked_cross_right_klt
    global n_pairs, len_initial_matches
    # --- Camera params and rectification setup ---
    mtx1 = calib["camera_matrix_1"]; dist1 = calib["dist_coeffs_1"]
    mtx2 = calib["camera_matrix_2"]; dist2 = calib["dist_coeffs_2"]
    R = calib["R"]; T = calib["T"]
    h, w = left.shape[:2]
    image_size = (w, h)

    # A) Rectify for ArUco (alpha = 0)
    R1a, R2a, P1a, P2a, Qa, _, _ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=0
    )
    map1x_a, map1y_a = cv2.initUndistortRectifyMap(mtx1, dist1, R1a, P1a, image_size, cv2.CV_32FC1)
    map2x_a, map2y_a = cv2.initUndistortRectifyMap(mtx2, dist2, R2a, P2a, image_size, cv2.CV_32FC1)
    left_aruco  = cv2.remap(left,  map1x_a, map1y_a, cv2.INTER_LINEAR)
    right_aruco = cv2.remap(right, map2x_a, map2y_a, cv2.INTER_LINEAR)

    # B) Rectify for crosses (alpha = 0; keep identical behavior)
    R1c, R2c, P1c, P2c, Qc, _, _ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=0
    )
    map1x_c, map1y_c = cv2.initUndistortRectifyMap(mtx1, dist1, R1c, P1c, image_size, cv2.CV_32FC1)
    map2x_c, map2y_c = cv2.initUndistortRectifyMap(mtx2, dist2, R2c, P2c, image_size, cv2.CV_32FC1)
    left_cross  = cv2.remap(left,  map1x_c, map1y_c, cv2.INTER_LINEAR)
    right_cross = cv2.remap(right, map2x_c, map2y_c, cv2.INTER_LINEAR)

    # Contrast/brightness bump (unchanged)
    left_cross  = cv2.convertScaleAbs(left_cross,  alpha=4, beta=20)
    right_cross = cv2.convertScaleAbs(right_cross, alpha=4, beta=20)

    # (Optional quick look at brightened frames)
    show_bright_frames = False
    if show_bright_frames:
        fx = fy = 0.3
        cv2.imshow("Left Cross Bright",  cv2.resize(left_cross,  None, fx=fx, fy=fy))
        cv2.imshow("Right Cross Bright", cv2.resize(right_cross, None, fx=fx, fy=fy))
        cv2.waitKey(0); cv2.destroyAllWindows()

    # --- 1) ArUco detection (4x4 + 7x7) ---
    aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_dict_7x7 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)

    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 15
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.001
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.minDistanceToBorder = 1

    # Detect on rectified frames
    corners_l_4x4, ids_l_4x4, _ = cv2.aruco.detectMarkers(left_aruco,  aruco_dict_4x4, parameters=parameters)
    corners_r_4x4, ids_r_4x4, _ = cv2.aruco.detectMarkers(right_aruco, aruco_dict_4x4, parameters=parameters)

    corners_l_7x7, ids_l_7x7, _ = cv2.aruco.detectMarkers(left_aruco,  aruco_dict_7x7, parameters=parameters)
    corners_r_7x7, ids_r_7x7, _ = cv2.aruco.detectMarkers(right_aruco, aruco_dict_7x7, parameters=parameters)

    aruco_centers_L_4x4, aruco_centers_R_4x4, aruco_ids_4x4 = [], [], []
    aruco_rotations_7x7, aruco_translations_7x7 = [], []
    aruco_centers_L_7x7, aruco_centers_R_7x7, aruco_ids_7x7 = [], [], []

    # 4x4: compute centers and keep common IDs
    if ids_l_4x4 is not None and ids_r_4x4 is not None:
        centers_l_4x4 = {id_: np.mean(c[0], axis=0) for id_, c in zip(ids_l_4x4.flatten(), corners_l_4x4)}
        centers_r_4x4 = {id_: np.mean(c[0], axis=0) for id_, c in zip(ids_r_4x4.flatten(), corners_r_4x4)}
        common_ids_4x4 = set(centers_l_4x4.keys()) & set(centers_r_4x4.keys())
        for aruco_id in common_ids_4x4:
            aruco_centers_L_4x4.append(centers_l_4x4[aruco_id])
            aruco_centers_R_4x4.append(centers_r_4x4[aruco_id])
            aruco_ids_4x4.append(aruco_id)

    # 7x7: centers + pose (as before)
    if ids_l_7x7 is not None and ids_r_7x7 is not None:
        centers_l_7x7 = {id_: np.mean(c[0], axis=0) for id_, c in zip(ids_l_7x7.flatten(), corners_l_7x7)}
        centers_r_7x7 = {id_: np.mean(c[0], axis=0) for id_, c in zip(ids_r_7x7.flatten(), corners_r_7x7)}
        common_ids_7x7 = set(centers_l_7x7.keys()) & set(centers_r_7x7.keys())
        for aruco_id in common_ids_7x7:
            aruco_centers_L_7x7.append(centers_l_7x7[aruco_id])
            aruco_centers_R_7x7.append(centers_r_7x7[aruco_id])
            aruco_ids_7x7.append(aruco_id)

        # Pose from left (kept identical to your code)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_l_7x7, 0.19, mtx1, dist1)
        aruco_rotations_7x7.append(rvec)
        aruco_translations_7x7.append(tvec)

    print(f"[INFO] ArUco detected in left/right frame: {len(aruco_centers_L_4x4) + len(aruco_centers_L_7x7)}")

    # --- 2) Cross detection ---
    cross_left_raw, _ = detect_crosses(left_cross)
    cross_right_raw, _ = detect_crosses(right_cross)
    print(f"[INFO] Crosses detected in L/R: {len(cross_left_raw)}, {len(cross_right_raw)}")

    gray_left = cv2.cvtColor(left_cross, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_cross, cv2.COLOR_BGR2GRAY)

    # Always re-separate for labels on current LEFT detections
    cross_left_raw = separate_LE_and_struts(_coords_only(cross_left_raw), gray_left)

    # --- 3) Stereo matching (init or KLT update) ---
    if not n_pairs or n_pairs < 30 or n_pairs < len_initial_matches:
        print(f"[INFO] Fresh stereo matching at frame {frame_counter}")
        tracked_cross_left, tracked_cross_right = match_stereo_points(
            cross_left_raw, cross_right_raw,
            max_vertical_disparity=70, max_total_distance=500
        )
        if not tracked_cross_left or not tracked_cross_right:
            return None, frame_counter + 1

        # Use the actual fresh count here
        len_initial_matches = len(tracked_cross_left)
        print(f"length initial matches: {len_initial_matches}")

        # Re-label LEFT for this frame (pass coords only)
        labeled_tracked_cross_left = separate_LE_and_struts(
            _coords_only(tracked_cross_left), gray_left
        )
        tracked_cross_left_klt = labeled_tracked_cross_left
        tracked_cross_right_klt = tracked_cross_right

        # IMPORTANT: keep the variables you draw/use in sync
        tracked_cross_left = tracked_cross_left_klt
        tracked_cross_right = tracked_cross_right_klt

        prev_gray_left, prev_gray_right = gray_left, gray_right

    else:
        print(len_initial_matches)

        # 1) Update LEFT with LK (as before)
        if prev_gray_left is not None and tracked_cross_left_klt:
            tracked_cross_left_klt, _ = track_klt(prev_gray_left, gray_left, tracked_cross_left_klt)

        # 2) Re-label LEFT from the current frame (produces triples (x,y,label))
        tracked_cross_left_klt = separate_LE_and_struts(
            _coords_only(tracked_cross_left_klt), gray_left
        )

        # 3) DO NOT KLT THE RIGHT. Recompute RIGHT by stereo using the *LABELED* left list
        #    (match_stereo_points expects pt[2] to exist for left points)
        left_m, right_m = match_stereo_points(
            tracked_cross_left_klt,  # <-- pass labeled triples here
            cross_right_raw,
            max_vertical_disparity=70, max_total_distance=500
        )

        # 4) Adopt the stereo-synced order for both sides
        tracked_cross_left_klt = left_m
        tracked_cross_right_klt = right_m

        # 5) Keep draw/usage variables updated EVERY frame
        tracked_cross_left = tracked_cross_left_klt
        tracked_cross_right = tracked_cross_right_klt

        prev_gray_left, prev_gray_right = gray_left, gray_right

    # --- Report the true usable stereo-pair count ---
    n_pairs = min(len(tracked_cross_left) if tracked_cross_left else 0,
                  len(tracked_cross_right) if tracked_cross_right else 0)
    print(f"[INFO] Using {n_pairs} stereo-matched crosses (KLT or fresh)")
    if n_pairs > len(cross_right_raw) or n_pairs > len(cross_left_raw):
        print("[WARN] Using more stereo pairs than current raw detections (likely from KLT carry-over).")

    # --- (Optional) Epipolar lines (kept disabled) ---
    epipolar_lines = False
    if epipolar_lines:
        for i, pt in enumerate(tracked_cross_left):
            y = int(pt[1])
            cv2.line(left_cross, (0, y), (left_cross.shape[1], y), (255, 0, 255), 1)
            cv2.putText(left_cross, str(i), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        for i, pt in enumerate(tracked_cross_right):
            y = int(pt[1])
            cv2.line(right_cross, (0, y), (right_cross.shape[1], y), (255, 0, 255), 1)
            cv2.putText(right_cross, str(i), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # --- Debug overlay every 30th frame (single LEFT frame, 180° flipped, fixed scale) ---
    # --- Debug overlay every 30th frame (L+R, 180° flip, no save) ---
    # --- Debug overlay every 30th frame (L+R, 180° flip, no save, safe close) ---
    debug_frame = True
    if frame_counter % 30 == 0 and debug_frame:
        FLIP_180 = True
        DISPLAY_SCALE = 0.3  # e.g., 0.3–1.0. Increase if it looks too small.

        # -------- Helper: annotate one frame (no coords; white-on-black labels) --------
        def annotate_frame(base_img, crosses, aruco4=None, aruco7=None,
                           flip=True, font_scale=1.6, thickness=2):
            """
            base_img: np.ndarray (BGR)
            crosses:  iterable of (x,y) or (x,y,label)
            aruco4/7: iterable of (x,y) or None
            flip:     if True, rotate 180° and map coords accordingly
            """
            img = base_img.copy()
            H, W = img.shape[:2]

            # rotate image first so text is upright in final orientation
            if flip:
                img = cv2.rotate(img, cv2.ROTATE_180)

            def map_xy(x, y):
                xi, yi = int(round(x)), int(round(y))
                if flip:
                    return (W - 1 - xi, H - 1 - yi)
                return (xi, yi)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)  # white
            box_color = (0, 0, 0)  # black

            # Cross markers: index + label (no x-position printed)
            for i, pt in enumerate(crosses):
                if len(pt) == 3:
                    cx, cy, label = pt
                    label_str = f"{i}: {label}"
                else:
                    cx, cy = pt
                    label_str = str(i)

                x, y = map_xy(cx, cy)
                cv2.circle(img, (x, y), 6, (0, 255, 0), 2)
                (w_txt, h_txt), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
                tl = (x + 10, y - h_txt)
                br = (x + 10 + w_txt, y + baseline)
                cv2.rectangle(img, tl, br, box_color, -1)
                cv2.putText(img, label_str, (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # ArUco markers: generic labels only
            if aruco4:
                for (ax, ay) in aruco4:
                    x, y = map_xy(ax, ay)
                    cv2.circle(img, (x, y), 8, (255, 0, 0), 2)
                    label = "ArUco 4x4"
                    (w_txt, h_txt), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    tl = (x + 10, y - h_txt)
                    br = (x + 10 + w_txt, y + baseline)
                    cv2.rectangle(img, tl, br, box_color, -1)
                    cv2.putText(img, label, (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            if aruco7:
                for (ax, ay) in aruco7:
                    x, y = map_xy(ax, ay)
                    cv2.circle(img, (x, y), 8, (255, 0, 0), 2)
                    label = "ArUco 7x7"
                    (w_txt, h_txt), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    tl = (x + 10, y - h_txt)
                    br = (x + 10 + w_txt, y + baseline)
                    cv2.rectangle(img, tl, br, box_color, -1)
                    cv2.putText(img, label, (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            return img

        # ---- Make sure both frames have same base size (avoids weird tiny stacking) ----
        Lh, Lw = left_cross.shape[:2]
        Rh, Rw = right_cross.shape[:2]
        if (Rh, Rw) != (Lh, Lw):
            right_cross_disp_base = cv2.resize(right_cross, (Lw, Lh), interpolation=cv2.INTER_AREA)
        else:
            right_cross_disp_base = right_cross

        # Safely pick ArUco arrays if they exist; otherwise None
        left_aruco4 = aruco_centers_L_4x4 if 'aruco_centers_L_4x4' in locals() else None
        left_aruco7 = aruco_centers_L_7x7 if 'aruco_centers_L_7x7' in locals() else None
        right_aruco4 = aruco_centers_R_4x4 if 'aruco_centers_R_4x4' in locals() else None
        right_aruco7 = aruco_centers_R_7x7 if 'aruco_centers_R_7x7' in locals() else None

        # ---- Annotate both frames ----
        left_annot = annotate_frame(left_cross, tracked_cross_left, left_aruco4, left_aruco7,
                                    flip=FLIP_180, font_scale=1.6, thickness=2)
        right_annot = annotate_frame(right_cross_disp_base, tracked_cross_right, right_aruco4, right_aruco7,
                                     flip=FLIP_180, font_scale=1.6, thickness=2)

        # ---- Scale for display (one fixed factor for both) ----
        if not (0 < DISPLAY_SCALE <= 1.5):
            DISPLAY_SCALE = 0.6  # sanity
        left_disp = cv2.resize(left_annot, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
        right_disp = cv2.resize(right_annot, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)

        # Ensure equal heights before hstack
        hL, wL = left_disp.shape[:2]
        hR, wR = right_disp.shape[:2]
        if hL != hR:
            target_h = min(hL, hR)

            def resize_to_h(img, target_h):
                h, w = img.shape[:2]
                new_w = max(1, int(round(w * (target_h / h))))
                return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

            left_disp = resize_to_h(left_disp, target_h)
            right_disp = resize_to_h(right_disp, target_h)

        combined = np.hstack((left_disp, right_disp))

        # ---- One clean window; safe close; no saving ----
        win_name = f"[DEBUG] L+R Frame {frame_counter}  (ESC / Q / Enter = close)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # DPI-safe resizable window
        # Force window to the image size in pixels (helps on high-DPI displays)
        cv2.resizeWindow(win_name, combined.shape[1], combined.shape[0])
        cv2.imshow(win_name, combined)

        # Allow closing via ESC/Q/Enter *or* clicking X
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q'), 13):  # ESC, q/Q, Enter
                break
            # if user clicked X, window disappears -> break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        # Safe cleanup (prevents the "NULL window" error)
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass
        cv2.waitKey(1)  # flush events

    # --- 4) Triangulation ---
    if not tracked_cross_left or not tracked_cross_right:
        return None, frame_counter + 1

    # Use the same n_pairs for triangulation to avoid shape mismatches
    n_cross = min(len(tracked_cross_left), len(tracked_cross_right))
    cross_pts_L = [tuple(map(float, pt[:2])) for pt in tracked_cross_left[:n_cross]]
    cross_pts_R = [tuple(map(float, pt[:2])) for pt in tracked_cross_right[:n_cross]]
    cross_3d = triangulate_points(cross_pts_L, cross_pts_R, P1c, P2c)

    aruco_3d_4by4 = []
    if aruco_centers_L_4x4 and aruco_centers_R_4x4:
        aruco_3d_4by4 = triangulate_points(aruco_centers_L_4x4, aruco_centers_R_4x4, P1a, P2a)

    aruco_3d_7by7 = []
    if aruco_centers_L_7x7 and aruco_centers_R_7x7:
        aruco_3d_7by7 = triangulate_points(aruco_centers_L_7x7, aruco_centers_R_7x7, P1a, P2a)

    # Build labels aligned with left points used for triangulation
    labels = []
    for pt in tracked_cross_left[:n_cross]:
        if len(pt) == 3:
            labels.append(pt[2])
        else:
            labels.append("")

    return (cross_3d, labels,
            aruco_3d_4by4, aruco_ids_4x4,
            aruco_3d_7by7, aruco_ids_7x7,
            aruco_rotations_7x7, aruco_translations_7x7,
            frame_counter + 1)



def show_debug_frame_pair(i, frame_left, frame_right, fx=0.3, fy=0.3, fps=30, skip_seconds=22, show_every=200):
    """
    Show a frame pair every `show_every` iterations after `skip_seconds` have passed.
    Waits for key press to proceed.
    """
    if i < skip_seconds * fps:
        return

    if (i - skip_seconds * fps) % show_every != 0:
        return

    left_small = cv2.resize(frame_left, None, fx=fx, fy=fy)
    right_small = cv2.resize(frame_right, None, fx=fx, fy=fy)

    cv2.imshow("Left Debug Frame", left_small)
    cv2.imshow("Right Debug Frame", right_small)
    print(f"[DEBUG] Showing frame pair {i}, press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
