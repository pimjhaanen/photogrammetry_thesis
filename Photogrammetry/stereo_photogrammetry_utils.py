"""
Processing utilities for the kite stereo photogrammetry pipeline.

- NO ArUco code.
- Accepts optional per-frame R_current to override calib['R'] (for rotation corrections).
- If cfg.gradient_brightness_contrast is None -> apply uniform alpha/beta to whole frame.
  Else, apply your left/right gradient gain (good for turn segments).
- Keeps: cross detection, KLT tracking on LEFT, stereo matching with epipolar + Hungarian,
  and triangulation to 3D. Optional debug overlay with interactive quit (Esc/q/Enter).

Return signature:
    cross_3d (N,3) or None, labels (list), next_frame_counter, updated_state
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
    """
    Tunables for detection, tracking, rectification, matching, and debug.

    - max_vertical_disparity: allowed |y_L - y_R| in rectified pixels (epipolar constraint).
    - max_total_distance:     max 2D distance between candidate pair (px).
    - min_horizontal_disparity: require x_L - x_R >= this (px).

    - lk_*: KLT settings for tracking LEFT points frame-to-frame.

    - rectify_alpha_cross: alpha for stereoRectify (0..1) for cross stream.

    - brighten_alpha / brighten_beta: if gradient_brightness_contrast is None, apply
      cv2.convertScaleAbs with these to the whole frame; otherwise the per-side gradient
      function will be used (see code).

    - show_debug_frame / debug_every_n / debug_flip_180 / debug_display_scale:
      controls the optional overlay window.

    - need_fresh_min_pairs: trigger a “fresh” stereo match if pairs drop below this count
      or below the initial matched count.

    - epipolar_lines: overlay epipolar lines in debug.

    - The SYNC/FLASH fields are only used by the runner’s match_videos call (kept for convenience).

    - premultiply_rotation_correction (used by the runner): whether to apply R_corr @ R (True) or R @ R_corr (False).
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

    # Rectification for crosses
    rectify_alpha_cross: float = 0.0

    # Preprocessing (brightness/contrast bump on cross frames)
    brighten_alpha: float = 4.0
    brighten_beta: float = 20.0
    gradient_brightness_contrast: Optional[str] = "lr"  # "lr" or None

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

    # --- AUDIO SYNC (used by runner only) ---
    sync_start_seconds: float = 0.0
    sync_match_duration: float = 300.0
    sync_downsample_factor: int = 50
    sync_plot_audio: bool = False

    # --- FLASH DETECTION (used by runner only) ---
    flash_occurs_after: float = 0.0
    flash_occurs_before: Optional[float] = None
    flash_center_fraction: float = 1/3
    flash_min_jump: float = 20.0
    flash_slope_ratio: float = 5.0
    flash_baseline_window: int = 5
    flash_brightness_floor: float = 0.0
    flash_plot: bool = True

    # --- How to apply rotation correction relative to base R (used by runner only) ---
    premultiply_rotation_correction: bool = True


@dataclass
class TrackerState:
    """Holds tracking state across frames (replaces module-level globals)."""
    prev_gray_left: Optional[np.ndarray] = None
    prev_gray_right: Optional[np.ndarray] = None
    tracked_cross_left_klt: Optional[List[Point2D]] = None
    tracked_cross_right_klt: Optional[List[Tuple[float, float]]] = None
    len_initial_matches: int = 0
    n_pairs: int = 0

#Accuracy test helper:
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


# ================================== HELPERS ==================================

def _coords_only(pts: Iterable[Point2D]) -> List[Tuple[float, float]]:
    """Drop labels if present, keep (x, y) only."""
    return [(float(p[0]), float(p[1])) for p in (pts or [])]


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


def _annotate_frame(
    base_img: np.ndarray,
    crosses: Iterable[Point2D],
    flip: bool = True,
    font_scale: float = 1.6,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay cross indices/labels for quick inspection (optionally 180° flip)."""
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
    R_current: Optional[np.ndarray] = None,   # <<< per-frame rotation override
) -> Tuple[
    Optional[np.ndarray], List[Union[int, str]],
    int, TrackerState
]:
    """
    - left_bgr, right_bgr: BGR frames (same size).
    - calib: {'camera_matrix_1','dist_coeffs_1','camera_matrix_2','dist_coeffs_2','R','T'}.
    - R_current: if provided, overrides calib['R'] for this frame (translation T unchanged).
    - state: TrackerState to persist across frames.
    - separate_fn: function to label crosses on the LEFT image as (x,y,label).

    RETURNS:
    - cross_3d (N,3) or None if not enough pairs,
      labels (list),
      next_frame_counter,
      updated_state.
    """
    K1, D1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    K2, D2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R = calib["R"] if R_current is None else R_current
    T = calib["T"]

    h, w = left_bgr.shape[:2]
    size = (w, h)

    # --- Rectify using (possibly corrected) R for this frame ---
    (_, _, P1c, P2c, Qc), mapsC = _stereo_rectify_maps(K1, D1, K2, D2, R, T, size, cfg.rectify_alpha_cross)
    map1x_c, map1y_c, map2x_c, map2y_c = mapsC
    left_cross  = cv2.remap(left_bgr,  map1x_c, map1y_c, cv2.INTER_LINEAR)
    right_cross = cv2.remap(right_bgr, map2x_c, map2y_c, cv2.INTER_LINEAR)

    # --- Brightness/contrast bump: uniform vs gradient ---
    if cfg.gradient_brightness_contrast is None:
        left_cross  = cv2.convertScaleAbs(left_cross,  alpha=cfg.brighten_alpha, beta=cfg.brighten_beta)
        right_cross = cv2.convertScaleAbs(right_cross, alpha=cfg.brighten_alpha, beta=cfg.brighten_beta)
    else:
        # Your gradient gain helper (good for turns)
        left_cross = apply_lr_gradient_gain(
            left_cross,
            bright_side_gain=(cfg.brighten_alpha - 2.0, cfg.brighten_beta - 10.0),
            dark_side_gain=(cfg.brighten_alpha + 2.0,  cfg.brighten_beta + 5.0),
            bands=1, image='left', smooth=True
        )
        right_cross = apply_lr_gradient_gain(
            right_cross,
            bright_side_gain=(cfg.brighten_alpha - 2.0, cfg.brighten_beta - 10.0),
            dark_side_gain=(cfg.brighten_alpha + 2.0,  cfg.brighten_beta + 5.0),
            bands=1, image='right', smooth=True
        )

    if cfg.show_bright_frames:
        Ld = cv2.rotate(left_cross,  cv2.ROTATE_180) if cfg.debug_flip_180 else left_cross
        Rd = cv2.rotate(right_cross, cv2.ROTATE_180) if cfg.debug_flip_180 else right_cross
        hL, wL = Ld.shape[:2]; hR, wR = Rd.shape[:2]
        if hL != hR:
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

    # === 1) Cross detection & labeling (LEFT always re-labeled) ===
    cross_left_raw, _ = detect_crosses(left_cross)
    cross_right_raw, _ = detect_crosses(right_cross)
    cross_left_labeled = separate_fn(_coords_only(cross_left_raw), gray_left)
    print(f"crosses detected in left frame: {len(cross_left_raw)}",
          f"crosses detected in right frame: {len(cross_right_raw)}")

    # === 2) Init/update stereo matches of crosses (refresh rule uses cfg.need_fresh_min_pairs) ===
    need_fresh = (state.n_pairs < cfg.need_fresh_min_pairs) or (state.n_pairs < state.len_initial_matches) or (state.n_pairs == 0)
    if need_fresh:
        left_m, right_m = match_stereo_points(
            cross_left_labeled, cross_right_raw,
            max_vertical_disparity=cfg.max_vertical_disparity,
            max_total_distance=cfg.max_total_distance,
            min_horizontal_disparity=cfg.min_horizontal_disparity,
        )
        if not left_m or not right_m:
            return None, [], frame_counter + 1, state
        state.tracked_cross_left_klt  = left_m
        state.tracked_cross_right_klt = right_m
        state.len_initial_matches = len(left_m)
        state.prev_gray_left, state.prev_gray_right = gray_left, gray_right
    else:
        # Update LEFT via KLT, re-label on current frame, then re-match RIGHT by stereo
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

    # --- NEW: quick epipolar/2D distance diagnostic print every 30 frames ---
    if state.tracked_cross_left_klt and state.tracked_cross_right_klt and (frame_counter % 30 == 0):
        Lp = np.array([p[:2] for p in state.tracked_cross_left_klt], float)
        Rp = np.array([p for p in state.tracked_cross_right_klt], float)
        vdisp = np.abs(Lp[:, 1] - Rp[:, 1])
        d2 = np.linalg.norm(Lp - Rp, axis=1)
        print(
            f"[diag] pairs={len(Lp)} mean|Δy|={np.mean(vdisp):.2f} px  p95|Δy|={np.percentile(vdisp, 95):.2f}  mean d2={np.mean(d2):.1f} px")

    # Optional epipolar lines overlay (debug)
    if cfg.epipolar_lines:
        for pt in state.tracked_cross_left_klt or []:
            y = int(pt[1])
            cv2.line(left_cross, (0, y), (left_cross.shape[1], y), (255, 0, 255), 1)
        for pt in state.tracked_cross_right_klt or []:
            y = int(pt[1])
            cv2.line(right_cross, (0, y), (right_cross.shape[1], y), (255, 0, 255), 1)

    # === Debug overlay every N frames ===
    if cfg.debug_every_n and frame_counter % cfg.debug_every_n == 0 and cfg.show_debug_frame:
        L_ann = _annotate_frame(left_cross,  state.tracked_cross_left_klt,
                                flip=cfg.debug_flip_180, font_scale=1.6, thickness=2)
        R_ann = _annotate_frame(right_cross, state.tracked_cross_right_klt,
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
            key = cv2.waitKey(20) & 0xFF  # mask to 8-bit, fixes the previous 0.FF typo
            if key in (27, ord('q'), ord('Q'), 13) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.waitKey(1)

    # === 3) Triangulation ===
    if not state.tracked_cross_left_klt or not state.tracked_cross_right_klt:
        return None, [], frame_counter + 1, state

    n_cross = min(len(state.tracked_cross_left_klt), len(state.tracked_cross_right_klt))
    cross_L = [tuple(map(float, p[:2])) for p in state.tracked_cross_left_klt[:n_cross]]
    cross_R = [tuple(map(float, p[:2])) for p in state.tracked_cross_right_klt[:n_cross]]
    cross_3d = triangulate_points(cross_L, cross_R, P1c, P2c)

    labels: List[Union[int, str]] = [(p[2] if len(p) == 3 else "") for p in state.tracked_cross_left_klt[:n_cross]]

    # Ensure (N,3) array
    cross_3d_array = np.array([c if isinstance(c, np.ndarray) else c[0] for c in cross_3d])

    return cross_3d_array, labels, frame_counter + 1, state


# ============================================================================ #
#                               GEOMETRY UTILITIES                             #
# ============================================================================ #

def triangulate_points(
    pts1: List[Point2D],
    pts2: List[Tuple[float, float]],
    P1: np.ndarray,
    P2: np.ndarray,
) -> List[Union[np.ndarray, Tuple[np.ndarray, Union[int, str]]]]:
    """
    Triangulate N corresponding points from rectified cameras with projection matrices P1/P2.
    If pts1 elements carry labels as (x, y, label), the label is preserved in the output tuples.
    """
    if len(pts1) != len(pts2):
        raise ValueError(f"pts1 and pts2 must have same length; got {len(pts1)} vs {len(pts2)}")

    left_xy = np.array([p[:2] for p in pts1], dtype=np.float32).T  # 2×N
    right_xy = np.array([p[:2] for p in pts2], dtype=np.float32).T  # 2×N

    pts4 = cv2.triangulatePoints(P1, P2, left_xy, right_xy)  # 4×N
    pts3 = (pts4[:3, :] / pts4[3, :]).T                      # N×3

    have_labels = any(len(p) == 3 for p in pts1)
    if have_labels:
        labels = [p[2] for p in pts1]
        return [(pts3[i], labels[i]) for i in range(len(pts1))]
    else:
        return [pts3[i] for i in range(len(pts1))]


def match_stereo_points(
    left_points: List[Point2D],
    right_points: List[Tuple[float, float]],
    max_vertical_disparity: float = 2.0,
    max_total_distance: float = 30.0,
    min_horizontal_disparity: float = 100.0,
) -> Tuple[List[Point2D], List[Tuple[float, float]]]:
    """
    Stereo match points via epipolar + distance constraints, solved with Hungarian assignment.
    Keeps labels if present on LEFT points.
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
            hdisp = lpt[0] - rpt[0]  # expect positive if left x > right x in rectified setup
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
