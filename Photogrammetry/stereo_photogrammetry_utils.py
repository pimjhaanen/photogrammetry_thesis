"""This code can be used to process stereo frames for a kite photogrammetry pipeline:
it rectifies left/right images, detects ArUco (4×4 & 7×7) and cross markers,
tracks crosses over time with KLT (left only), stereo-matches crosses with epipolar
constraints, and triangulates matched points to 3D. It also supports periodic
debug overlays for quick inspection.

All tunable parameters (Aruco dictionaries/params, 7×7 marker size, KLT settings,
stereo-match thresholds, rectification, brightness bump, and debug overlay options)
are exposed at the top via the StereoConfig dataclass.

It also provides grid-wise sorting of 2D markers, stereo matching with epipolar constraints and the
Hungarian algorithm, and triangulation of corresponding points into 3D.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment
from Photogrammetry.marker_detection.marker_detection_utils import detect_crosses, apply_lr_gradient_gain

# -------------------------- Common type aliases --------------------------

Point2D = Union[Tuple[float, float], Tuple[float, float, Union[int, str]]]
Point3D = np.ndarray  # shape (N, 3)


# =============================== CONFIG (all tunables) ===============================

@dataclass
class StereoConfig:
    """RELEVANT FUNCTION INPUTS:
    - max_vertical_disparity: allowed |y_L - y_R| in rectified pixels (epipolar constraint).
    - max_total_distance:     max 2D distance between candidate pair (px).
    - min_horizontal_disparity: require x_L - x_R >= this (px) to avoid reversed matches.

    - lk_win_size:  window size (w,h) for KLT.
    - lk_max_level: pyramid levels for KLT.
    - lk_term_count:  KLT termination max iterations.
    - lk_term_eps:    KLT termination epsilon.

    - rectify_alpha_aruco: stereoRectify alpha for ArUco rectification (0..1).
    - rectify_alpha_cross: stereoRectify alpha for cross rectification (0..1).

    - brighten_alpha: contrast multiplier for cross frames (cv2.convertScaleAbs).
    - brighten_beta:  brightness offset for cross frames (cv2.convertScaleAbs).

    - aruco_4x4_dict:  name or numeric ID of 4×4 dictionary (e.g. 'DICT_4X4_100').
    - aruco_7x7_dict:  name or numeric ID of 7×7 dictionary (e.g. 'DICT_7X7_50').
    - aruco_adapt_win_min/max/step: adaptive threshold window sizes.
    - aruco_corner_refine_method:   cv2.aruco.CORNER_REFINE_* enum (int).
    - aruco_min_perimeter_rate:     min perimeter (relative to image size).
    - aruco_max_perimeter_rate:     max perimeter (relative to image size).
    - aruco_min_distance_to_border: min distance to border in pixels.
    - aruco_marker_length_7x7_m:    physical side length (meters) for pose on 7×7 markers.

    - show_bright_frames: show the contrast-boosted L/R cross frames (debug).
    - debug_every_n:   show overlay every N frames (None/0 to disable).
    - debug_flip_180:  flip images 180° in the debug window (upright annotations).
    - debug_display_scale: resize factor for debug window (0.1..1.5).
    - debug_window_title_prefix: prefix string for debug window titles.

    - need_fresh_min_pairs: re-run fresh stereo matching if we have fewer than this count,
                            or fewer than the initial matched count.

    - epipolar_lines: overlay horizontal epipolar lines when debugging (False by default).
    """
    # Stereo matching thresholds
    max_vertical_disparity: float = 70.0
    max_total_distance: float = 500.0
    min_horizontal_disparity: float = 0.0

    # KLT (Lucas–Kanade) parameters
    lk_win_size: Tuple[int, int] = (31, 31)
    lk_max_level: int = 4
    lk_term_count: int = 40
    lk_term_eps: float = 0.03

    # Rectification
    rectify_alpha_aruco: float = 0.0
    rectify_alpha_cross: float = 0.0

    # Preprocessing (brightness/contrast bump on cross frames)
    brighten_alpha: float = 4.0
    brighten_beta: float = 20.0

    # ArUco detection configuration
    aruco_4x4_dict: Union[str, int] = "DICT_4X4_100"
    aruco_7x7_dict: Union[str, int] = "DICT_7X7_50"
    aruco_adapt_win_min: int = 5
    aruco_adapt_win_max: int = 15
    aruco_adapt_win_step: int = 10
    aruco_corner_refine_method: int = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_min_perimeter_rate: float = 0.001
    aruco_max_perimeter_rate: float = 4.0
    aruco_min_distance_to_border: int = 1
    aruco_marker_length_7x7_m: float = 0.19  # your original value

    # Debug / visualization
    show_bright_frames: bool = False
    show_debug_frame: bool = False
    debug_every_n: Optional[int] = 30
    debug_flip_180: bool = True
    debug_display_scale: float = 0.3
    debug_window_title_prefix: str = "[DEBUG] L+R Frame"

    # Matching refresh rule
    need_fresh_min_pairs: int = 30

    # Optional epipolar line overlay
    epipolar_lines: bool = False

    # --- AUDIO SYNC (match_videos) ---
    sync_start_seconds: float = 0.0          # where to start audio extraction
    sync_match_duration: float = 300.0       # seconds of audio used for correlation
    sync_downsample_factor: int = 50         # decimation factor (44.1 kHz / factor)
    sync_plot_audio: bool = False            # quick visual check of audio alignment

    # --- FLASH DETECTION (detect_flash_start_frame) ---
    flash_occurs_after: float = 0.0          # search window start (s)
    flash_occurs_before: Optional[float] = None  # search window end (s) or None
    flash_center_fraction: float = 1/3       # central ROI box fraction
    flash_min_jump: float = 20.0             # min Δ brightness to trigger
    flash_slope_ratio: float = 5.0           # Δ must exceed this × baseline slope
    flash_baseline_window: int = 5           # samples used for baseline slope
    flash_brightness_floor: float = 0.0      # min post-jump brightness
    flash_plot: bool = True                  # plot brightness/slope + marker


@dataclass
class TrackerState:
    """Holds tracking state across frames (replaces module-level globals)."""
    prev_gray_left: Optional[np.ndarray] = None
    prev_gray_right: Optional[np.ndarray] = None
    tracked_cross_left_klt: Optional[List[Point2D]] = None
    tracked_cross_right_klt: Optional[List[Tuple[float, float]]] = None
    len_initial_matches: int = 0
    n_pairs: int = 0


# ================================== HELPERS ==================================

def _coords_only(pts: Iterable[Point2D]) -> List[Tuple[float, float]]:
    """Drop labels if present, keep (x, y) only."""
    return [(float(p[0]), float(p[1])) for p in (pts or [])]


def _get_aruco_dict(dict_name_or_id: Union[str, int]):
    """Resolve a cv2.aruco predefined dictionary by name or numeric id."""
    if isinstance(dict_name_or_id, int):
        return cv2.aruco.getPredefinedDictionary(dict_name_or_id)
    name = str(dict_name_or_id).upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name_or_id}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _build_klt_criteria(cfg: StereoConfig) -> Tuple[int, int, float]:
    """Build OpenCV termination criteria tuple for calcOpticalFlowPyrLK."""
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.lk_term_count, cfg.lk_term_eps)


def klt_track(prev_img: np.ndarray, next_img: np.ndarray, prev_pts: List[Point2D], cfg: StereoConfig) -> List[Point2D]:
    """Track labeled/unstyled points from prev_img -> next_img with KLT and preserve labels."""
    if not prev_pts:
        return []
    has_lbl = len(prev_pts[0]) == 3
    prev_xy = np.array([p[:2] for p in prev_pts], np.float32).reshape(-1, 1, 2)
    labels = [p[2] if has_lbl else None for p in prev_pts]

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_img, next_img, prev_xy, None,
        winSize=cfg.lk_win_size,
        maxLevel=cfg.lk_max_level,
        criteria=_build_klt_criteria(cfg),
    )
    if next_pts is None or status is None:
        return []

    good_next = next_pts[status.flatten() == 1].reshape(-1, 2)
    good_labels = [lbl for lbl, s in zip(labels, status.flatten()) if s == 1]

    out: List[Point2D] = []
    for (x, y), lbl in zip(good_next, good_labels):
        out.append((float(x), float(y), lbl) if has_lbl else (float(x), float(y)))
    return out


def _stereo_rectify_maps(
    K1: np.ndarray, D1: np.ndarray, K2: np.ndarray, D2: np.ndarray,
    R: np.ndarray, T: np.ndarray, size: Tuple[int, int], alpha: float
):
    """Compute stereo rectification and remap grids for a given alpha."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=alpha
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    return (R1, R2, P1, P2, Q), (map1x, map1y, map2x, map2y)


def _aruco_centers(img: np.ndarray, dictionary, params):
    """Detect markers and return {id: (cx, cy)} mapping + raw corners list."""
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=params)
    centers: Dict[int, np.ndarray] = {}
    if ids is not None:
        for id_, c in zip(ids.flatten(), corners):
            centers[int(id_)] = np.mean(c[0], axis=0)
    return centers, corners or []


def _common_id_centers(L: Dict[int, np.ndarray], R: Dict[int, np.ndarray]):
    """Keep only marker centers that exist in both L and R."""
    common = sorted(set(L).intersection(R))
    L_pts = [tuple(map(float, L[i])) for i in common]
    R_pts = [tuple(map(float, R[i])) for i in common]
    return L_pts, R_pts, common


def _annotate_frame(
    base_img: np.ndarray,
    crosses: Iterable[Point2D],
    aruco4: Optional[Iterable[Tuple[float, float]]] = None,
    aruco7: Optional[Iterable[Tuple[float, float]]] = None,
    flip: bool = True,
    font_scale: float = 1.6,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay cross indices/labels and ArUco dots for quick inspection (optionally 180° flip)."""
    img = base_img.copy()
    H, W = img.shape[:2]
    if flip:
        img = cv2.rotate(img, cv2.ROTATE_180)

    def map_xy(x, y):
        xi, yi = int(round(x)), int(round(y))
        return (W - 1 - xi, H - 1 - yi) if flip else (xi, yi)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)

    for i, pt in enumerate(crosses or []):
        if len(pt) == 3:
            cx, cy, label = pt
            lbl = f"{i}: {label}"
        else:
            cx, cy = pt
            lbl = str(i)
        x, y = map_xy(cx, cy)
        cv2.circle(img, (x, y), 6, (0, 255, 0), 2)
        (w_txt, h_txt), base = cv2.getTextSize(lbl, font, font_scale, thickness)
        tl, br = (x + 10, y - h_txt), (x + 10 + w_txt, y + base)
        cv2.rectangle(img, tl, br, box_color, -1)
        cv2.putText(img, lbl, (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    for p in aruco4 or []:
        x, y = map_xy(*p)
        cv2.circle(img, (x, y), 8, (255, 0, 0), 2)
        cv2.putText(img, "ArUco 4x4", (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    for p in aruco7 or []:
        x, y = map_xy(*p)
        cv2.circle(img, (x, y), 8, (255, 0, 0), 2)
        cv2.putText(img, "ArUco 7x7", (x + 10, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img


# ============================= CORE FRAME PROCESS =============================

def process_stereo_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    calib: Dict[str, np.ndarray],
    state: TrackerState,
    frame_counter: int,
    cfg: StereoConfig,
    separate_fn: Callable[[Iterable[Tuple[float, float]], np.ndarray], List[Point2D]],
) -> Tuple[
    Optional[np.ndarray], List[Union[int, str]],
    List[np.ndarray], List[int],
    List[np.ndarray], List[int],
    List[np.ndarray], List[np.ndarray],
    int, TrackerState
]:
    """RELEVANT FUNCTION INPUTS:
    - left_bgr, right_bgr: BGR frames (same size).
    - calib: {'camera_matrix_1','dist_coeffs_1','camera_matrix_2','dist_coeffs_2','R','T'}.
    - state: TrackerState to persist across frames.
    - frame_counter: current frame index (int).
    - cfg: StereoConfig with *all* tunables centralized.
    - separate_fn: function to label crosses on the LEFT image as (x,y,label).

    RETURNS:
    - cross_3d, labels,
      aruco_3d_4x4, aruco_ids_4x4,
      aruco_3d_7x7, aruco_ids_7x7,
      aruco_rotations_7x7, aruco_translations_7x7,
      next_frame_counter, updated_state.
    """
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R, T = calib["R"], calib["T"]
    h, w = left_bgr.shape[:2]
    size = (w, h)

    # A) Rectify for ArUco
    (_, _, P1a, P2a, Qa), mapsA = _stereo_rectify_maps(K1, D1, K2, D2, R, T, size, cfg.rectify_alpha_aruco)
    map1x_a, map1y_a, map2x_a, map2y_a = mapsA
    left_aruco  = cv2.remap(left_bgr,  map1x_a, map1y_a, cv2.INTER_LINEAR)
    right_aruco = cv2.remap(right_bgr, map2x_a, map2y_a, cv2.INTER_LINEAR)

    # B) Rectify for crosses
    (_, _, P1c, P2c, Qc), mapsC = _stereo_rectify_maps(K1, D1, K2, D2, R, T, size, cfg.rectify_alpha_cross)
    map1x_c, map1y_c, map2x_c, map2y_c = mapsC
    left_cross  = cv2.remap(left_bgr,  map1x_c, map1y_c, cv2.INTER_LINEAR)
    right_cross = cv2.remap(right_bgr, map2x_c, map2y_c, cv2.INTER_LINEAR)

    # Optional brightness bump for cross frames
    left_cross = apply_lr_gradient_gain(
        left_cross,
        bright_side_gain=(cfg.brighten_alpha - 2 , cfg.brighten_beta - 5 ),
        dark_side_gain=(cfg.brighten_alpha + 2.0, cfg.brighten_beta + 5),
        bands=1,
        image = 'left',
        smooth=True
    )

    right_cross = apply_lr_gradient_gain(
        right_cross,
        bright_side_gain=(cfg.brighten_alpha - 2.0, cfg.brighten_beta - 5 ),
        dark_side_gain=(cfg.brighten_alpha +2, cfg.brighten_beta +5),
        bands=1,
        image = 'right',
        smooth=True
    )

    if cfg.show_bright_frames:
        # Quick preview of (optionally flipped) cross frames
        Ld = cv2.rotate(cv2.convertScaleAbs(left_cross,  alpha=1.0, beta=0.0), cv2.ROTATE_180) if cfg.debug_flip_180 else left_cross
        Rd = cv2.rotate(cv2.convertScaleAbs(right_cross, alpha=1.0, beta=0.0), cv2.ROTATE_180) if cfg.debug_flip_180 else right_cross
        hL, wL = Ld.shape[:2]; hR, wR = Rd.shape[:2]
        if hL != hR:
            # Simple height harmonization for hstack
            target_h = min(hL, hR)
            Ld = cv2.resize(Ld, (int(wL * target_h / hL), target_h))
            Rd = cv2.resize(Rd, (int(wR * target_h / hR), target_h))
        preview = np.hstack([Ld, Rd])
        s = cfg.debug_display_scale if 0 < cfg.debug_display_scale <= 1.5 else 0.6
        preview = cv2.resize(preview, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow("Brightened Cross Frames", preview)
        cv2.waitKey(1)

    gray_left  = cv2.cvtColor(left_cross,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_cross, cv2.COLOR_BGR2GRAY)

    # === 1) ArUco detection (4×4 + 7×7) with FULLY CONFIGURABLE inputs ===
    aruco4 = _get_aruco_dict(cfg.aruco_4x4_dict)
    aruco7 = _get_aruco_dict(cfg.aruco_7x7_dict)

    params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMin = cfg.aruco_adapt_win_min
    params.adaptiveThreshWinSizeMax = cfg.aruco_adapt_win_max
    params.adaptiveThreshWinSizeStep = cfg.aruco_adapt_win_step
    params.cornerRefinementMethod = cfg.aruco_corner_refine_method
    params.minMarkerPerimeterRate = cfg.aruco_min_perimeter_rate
    params.maxMarkerPerimeterRate = cfg.aruco_max_perimeter_rate
    params.minDistanceToBorder = cfg.aruco_min_distance_to_border

    L4, _ = _aruco_centers(left_aruco,  aruco4, params)
    R4, _ = _aruco_centers(right_aruco, aruco4, params)
    L7, c7L = _aruco_centers(left_aruco,  aruco7, params)
    R7, _ = _aruco_centers(right_aruco, aruco7, params)

    aruco_centers_L_4x4, aruco_centers_R_4x4, aruco_ids_4x4 = _common_id_centers(L4, R4)
    aruco_centers_L_7x7, aruco_centers_R_7x7, aruco_ids_7x7 = _common_id_centers(L7, R7)

    # Pose (LEFT) for 7×7 — markerLength configurable via cfg
    aruco_rotations_7x7: List[np.ndarray] = []
    aruco_translations_7x7: List[np.ndarray] = []
    if c7L:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c7L, cfg.aruco_marker_length_7x7_m, K1, D1)
        aruco_rotations_7x7.append(rvec)
        aruco_translations_7x7.append(tvec)

    # === 2) Cross detection & labeling (LEFT always re-labeled) ===
    cross_left_raw, _ = detect_crosses(left_cross)
    cross_right_raw, _ = detect_crosses(right_cross)
    cross_left_labeled = separate_fn(_coords_only(cross_left_raw), gray_left)
    print(f"crosses detected in left frame: {len(cross_left_raw)}", f"crosses detected in right frame: {len(cross_right_raw)}")
    # === 3) Init/update stereo matches of crosses (refresh rule uses cfg.need_fresh_min_pairs) ===
    need_fresh = (state.n_pairs < cfg.need_fresh_min_pairs) or (state.n_pairs < state.len_initial_matches) or (state.n_pairs == 0)
    if need_fresh:
        # Fresh stereo matching (no KLT yet)
        left_m, right_m = match_stereo_points(
            cross_left_labeled, cross_right_raw,
            max_vertical_disparity=cfg.max_vertical_disparity,
            max_total_distance=cfg.max_total_distance,
            min_horizontal_disparity=cfg.min_horizontal_disparity,
        )
        if not left_m or not right_m:
            return None, [], [], [], [], [], [], [], frame_counter + 1, state
        state.tracked_cross_left_klt  = left_m
        state.tracked_cross_right_klt = right_m
        state.len_initial_matches = len(left_m)
        state.prev_gray_left, state.prev_gray_right = gray_left, gray_right
    else:
        # Update LEFT via KLT, re-label from current frame, then re-match RIGHT by stereo
        if state.prev_gray_left is not None and state.tracked_cross_left_klt:
            state.tracked_cross_left_klt = klt_track(state.prev_gray_left, gray_left, state.tracked_cross_left_klt, cfg)
        state.tracked_cross_left_klt = separate_fn(_coords_only(state.tracked_cross_left_klt), gray_left)
        left_m, right_m = match_stereo_points(
            state.tracked_cross_left_klt, cross_right_raw,
            max_vertical_disparity=cfg.max_vertical_disparity,
            max_total_distance=cfg.max_total_distance,
            min_horizontal_disparity=cfg.min_horizontal_disparity,
        )
        state.tracked_cross_left_klt, state.tracked_cross_right_klt = left_m, right_m
        state.prev_gray_left, state.prev_gray_right = gray_left, gray_right

    state.n_pairs = min(len(state.tracked_cross_left_klt or []), len(state.tracked_cross_right_klt or []))

    # Optional epipolar lines overlay (debug)
    if cfg.epipolar_lines:
        for pt in state.tracked_cross_left_klt or []:
            y = int(pt[1])
            cv2.line(left_cross, (0, y), (left_cross.shape[1], y), (255, 0, 255), 1)
        for pt in state.tracked_cross_right_klt or []:
            y = int(pt[1])
            cv2.line(right_cross, (0, y), (right_cross.shape[1], y), (255, 0, 255), 1)

    # === Debug overlay every N frames (fully driven by cfg) ===
    if cfg.debug_every_n and frame_counter % cfg.debug_every_n == 0 and cfg.show_debug_frame:
        L_ann = _annotate_frame(left_cross,  state.tracked_cross_left_klt,
                                aruco4=aruco_centers_L_4x4, aruco7=aruco_centers_L_7x7,
                                flip=cfg.debug_flip_180, font_scale=1.6, thickness=2)
        R_ann = _annotate_frame(right_cross, state.tracked_cross_right_klt,
                                aruco4=aruco_centers_R_4x4, aruco7=aruco_centers_R_7x7,
                                flip=cfg.debug_flip_180, font_scale=1.6, thickness=2)
        hL, wL = L_ann.shape[:2]; hR, wR = R_ann.shape[:2]
        if hL != hR:
            target_h = min(hL, hR)

            def _resize_to_h(img, target_h):
                h, w = img.shape[:2]
                new_w = max(1, int(round(w * (target_h / h))))
                return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

            L_ann = _resize_to_h(L_ann, target_h)
            R_ann = _resize_to_h(R_ann, target_h)
        combined = np.hstack([L_ann, R_ann])
        s = cfg.debug_display_scale if 0 < cfg.debug_display_scale <= 1.5 else 0.6
        combined = cv2.resize(combined, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        win = f"{cfg.debug_window_title_prefix} {frame_counter}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, combined.shape[1], combined.shape[0])
        cv2.imshow(win, combined)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q'), 13) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.waitKey(1)

    # === 4) Triangulation ===
    if not state.tracked_cross_left_klt or not state.tracked_cross_right_klt:
        return None, [], [], [], [], [], [], [], frame_counter + 1, state

    n_cross = min(len(state.tracked_cross_left_klt), len(state.tracked_cross_right_klt))
    cross_L = [tuple(map(float, p[:2])) for p in state.tracked_cross_left_klt[:n_cross]]
    cross_R = [tuple(map(float, p[:2])) for p in state.tracked_cross_right_klt[:n_cross]]
    cross_3d = triangulate_points(cross_L, cross_R, P1c, P2c)

    aruco_3d_4x4: List[np.ndarray] = []
    if aruco_centers_L_4x4 and aruco_centers_R_4x4:
        aruco_3d_4x4 = triangulate_points(aruco_centers_L_4x4, aruco_centers_R_4x4, P1a, P2a)

    aruco_3d_7x7: List[np.ndarray] = []
    if aruco_centers_L_7x7 and aruco_centers_R_7x7:
        aruco_3d_7x7 = triangulate_points(aruco_centers_L_7x7, aruco_centers_R_7x7, P1a, P2a)

    labels: List[Union[int, str]] = [(p[2] if len(p) == 3 else "") for p in state.tracked_cross_left_klt[:n_cross]]

    # Ensure the returned cross_3d is a clean (N,3) array even if triangulate_points preserved labels
    cross_3d_array = np.array([c if isinstance(c, np.ndarray) else c[0] for c in cross_3d])

    return (
        cross_3d_array,
        labels,
        aruco_3d_4x4, aruco_ids_4x4,
        aruco_3d_7x7, aruco_ids_7x7,
        aruco_rotations_7x7, aruco_translations_7x7,
        frame_counter + 1,
        state,
    )


# ------------------------------- module main ---------------------------------
if __name__ == "__main__":
    # This is a utilities module; no runnable demo here.
    # Import this file and call `process_stereo_pair(...)` from your main loop.
    pass


# ============================================================================ #
#                               GEOMETRY UTILITIES                             #
# ============================================================================ #

# ------------------------------- Triangulation -------------------------------

def triangulate_points(
    pts1: List[Point2D],
    pts2: List[Tuple[float, float]],
    P1: np.ndarray,
    P2: np.ndarray,
) -> List[Union[np.ndarray, Tuple[np.ndarray, Union[int, str]]]]:
    """RELEVANT FUNCTION INPUTS:
    - pts1: list of left-image points as (x, y) OR (x, y, label). If a label is present, it is preserved.
    - pts2: list of right-image points as (x, y). Must correspond 1:1 with pts1 (same length / order).
    - P1, P2: 3×4 projection matrices for the left and right cameras (e.g., from cv2.stereoRectify).

    RETURNS:
    - A list with length N. Each item is either:
        * np.ndarray of shape (3,) if no labels were provided, or
        * (np.ndarray(3,), label) if labels were provided in pts1).

    Notes:
    - Uses cv2.triangulatePoints on vectorized 2×N coordinate arrays.
    """
    if len(pts1) != len(pts2):
        raise ValueError(f"pts1 and pts2 must have same length; got {len(pts1)} vs {len(pts2)}")

    # Extract coordinates
    left_xy = np.array([p[:2] for p in pts1], dtype=np.float32).T  # 2×N
    right_xy = np.array([p[:2] for p in pts2], dtype=np.float32).T  # 2×N

    # Triangulate (homogeneous) -> Euclidean
    pts4 = cv2.triangulatePoints(P1, P2, left_xy, right_xy)  # 4×N
    pts3 = (pts4[:3, :] / pts4[3, :]).T  # N×3

    # Reattach labels if the left points had them
    have_labels = any(len(p) == 3 for p in pts1)
    if have_labels:
        labels = [p[2] for p in pts1]
        return [(pts3[i], labels[i]) for i in range(len(pts1))]
    else:
        return [pts3[i] for i in range(len(pts1))]


# ------------------------------- Grid sorting -------------------------------

def sort_markers_gridwise(
    points: Iterable[Tuple[float, float]],
    n_rows: int = 2,
) -> List[Tuple[float, float]]:
    """RELEVANT FUNCTION INPUTS:
    - points: iterable of (x, y) pixel coordinates.
    - n_rows: expected number of horizontal rows in the grid (default 2).
              For n_rows=2, this matches your original median split.

    RETURNS:
    - Points sorted row-wise from top row to bottom row, and within each row from left to right.

    Notes:
    - Uses a y-binning into n_rows bands; sorts each band by x.
    """
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return []

    if n_rows <= 1:
        # Single row: just sort by x
        order = np.argsort(pts[:, 0])
        return [tuple(p) for p in pts[order]]

    y_vals = pts[:, 1]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

    # Row edges; tiny epsilon to include max in last bin
    edges = np.linspace(y_min, y_max + 1e-9, n_rows + 1)
    row_idx = np.clip(np.digitize(y_vals, edges) - 1, 0, n_rows - 1)

    sorted_points: List[Tuple[float, float]] = []
    # Top row (smallest y) first
    for r in range(n_rows):
        row_pts = pts[row_idx == r]
        if row_pts.size == 0:
            continue
        order_x = np.argsort(row_pts[:, 0])
        sorted_points.extend([tuple(p) for p in row_pts[order_x]])

    return sorted_points


# ------------------------------- Stereo matching -------------------------------

def match_stereo_points(
    left_points: List[Point2D],
    right_points: List[Tuple[float, float]],
    max_vertical_disparity: float = 2.0,
    max_total_distance: float = 30.0,
    min_horizontal_disparity: float = 100.0,
) -> Tuple[List[Point2D], List[Tuple[float, float]]]:
    """RELEVANT FUNCTION INPUTS:
    - left_points: list of (x, y) OR (x, y, label) from the LEFT image (rectified).
    - right_points: list of (x, y) from the RIGHT image (rectified).
    - max_vertical_disparity: max allowed |y_left - y_right| (px) — should be small when rectified.
    - max_total_distance:    max allowed Euclidean distance between candidate pairs (px).
    - min_horizontal_disparity: require x_left - x_right >= this (px) (avoid reversed matches).

    RETURNS:
    - matched_left: list of matched LEFT points (preserves label if present).
    - matched_right: list of matched RIGHT points in the same order.

    Method:
    - Build a cost matrix with invalid pairs set to a large penalty.
    - Solve with Hungarian algorithm for a globally optimal assignment.
    """
    if not left_points or not right_points:
        return [], []

    left_xy = np.array([p[:2] for p in left_points], dtype=float)
    left_labels = [p[2] if len(p) == 3 else None for p in left_points]
    right_xy = np.array([p[:2] for p in right_points], dtype=float)

    nL, nR = len(left_points), len(right_points)
    big = 1e6
    cost = np.full((nL, nR), fill_value=big, dtype=float)

    for i in range(nL):
        for j in range(nR):
            lpt = left_xy[i]
            rpt = right_xy[j]
            vdisp = abs(lpt[1] - rpt[1])
            hdisp = lpt[0] - rpt[0]  # expect positive if left x > right x in standard rectified setup
            if vdisp <= max_vertical_disparity and hdisp >= min_horizontal_disparity:
                d = float(np.linalg.norm(lpt - rpt))
                if d <= max_total_distance:
                    cost[i, j] = d

    li, rj = linear_sum_assignment(cost)

    matched_left: List[Point2D] = []
    matched_right: List[Tuple[float, float]] = []
    for i, j in zip(li, rj):
        if cost[i, j] < big:
            lx, ly = left_xy[i]
            label = left_labels[i]
            matched_left.append((lx, ly) if label is None else (lx, ly, label))
            matched_right.append((right_xy[j, 0], right_xy[j, 1]))

    return matched_left, matched_right
