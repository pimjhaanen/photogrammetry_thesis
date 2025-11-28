"""This code can be used to label cross markers and build/maintain an ArUco reference frame.
It provides: (1) LE/strut labeling by local brightness, (2) an ArUco reference table with
gap-filling, (3) a per-frame ArUco pose with optional SLERP/EMA smoothing, and (4) geometric
utilities (3D line fit, shortest-path ordering, and smoothing-by-matching).

You can adapt brightness thresholds, interpolation/smoothing options, and path parameters to
your dataset.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cv2

import math

# =============================== Module settings ===============================

# Master low-pass toggle and sub-switches for smoothing
LOWPASS: bool = False              # global master switch
LPF_ARUCO_TRACKS: bool = False     # low-pass raw ArUco tracks (per-id x/y/z)
LPF_POSE: bool = False             # low-pass pose (R, origin) via SLERP/EMA
ALPHA: float = 0.9                 # EMA factor for origins; SLERP uses beta = 1 - ALPHA

# Sampling for fitted 3D line visualization
LINE_SAMPLES: int = 50

# Filled by the caller (e.g., from build_aruco_reference_table). Do NOT auto-populate here.
aruco_ref_table: Optional[pd.DataFrame] = None

# Cache for smoothed poses across frames: frame_idx -> (R_axes, origin)
filtered_axes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


# ============================== Cross LE/struts ===============================

def _annulus_stats_fast(bgr_img: np.ndarray, cx: int, cy: int, r_in: int, r_out: int):
    """Return (V_med, S_med) in an annulus around (cx,cy) using a small ROI for speed."""
    h, w = bgr_img.shape[:2]
    x1 = max(cx - r_out, 0); x2 = min(cx + r_out + 1, w)
    y1 = max(cy - r_out, 0); y2 = min(cy + r_out + 1, h)
    if x2 <= x1 or y2 <= y1:
        return np.nan, np.nan

    roi_bgr = bgr_img[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    yy, xx = np.ogrid[y1:y2, x1:x2]
    rr2 = (yy - cy) * (yy - cy) + (xx - cx) * (xx - cx)
    mask = (rr2 >= r_in * r_in) & (rr2 <= r_out * r_out)
    if not np.any(mask):
        return np.nan, np.nan

    V = roi_hsv[..., 2][mask]
    S = roi_hsv[..., 1][mask]
    return float(np.median(V)), float(np.median(S))


def separate_LE_and_struts(
    cross_centres: Iterable[Tuple[float, float]],
    gray_img: np.ndarray,
    *,
    bgr_img: Optional[np.ndarray] = None,     # pass original BGR frame for color stats
    horizontal_spacing: float = 130.0,
    r_in: int = 6, r_out: int = 18,          # annulus radii for local background
    # --- New brightness-only "slider" threshold ---
    bg_threshold: float = 0.4,               # 0.0..1.0 (fraction) OR 0..255 (absolute)
    # Keep for compatibility; no longer used in brightness-only mode:
    min_points_per_strut: int = 2,           # require at least this many points per strut column
) -> List[Tuple[float, float, Union[int, str]]]:
    """
    Labels crosses as 'LE' (bright/white cloth) or strut ids '0'..'7' (dark/black background),
    using *only* the local background brightness (median V in HSV) around each point.

    Slider:
      - bg_threshold in [0.0..1.0] is treated as a fraction of 255 (e.g., 0.5 -> 127.5).
      - bg_threshold in [1..255] is used as an absolute V threshold.
      - V_med >= threshold  => 'LE'
      - V_med <  threshold  => 'strut' candidate

    Remaining logic:
      - Strut candidates are clustered by X (horizontal_spacing) to form columns.
      - Columns with < min_points_per_strut are reclassified as 'LE' to avoid spurious singletons.
    """
    # Convert threshold to absolute [0..255]
    thr = float(bg_threshold)
    if 0.0 <= thr <= 1.0:
        thr = thr * 255.0
    thr = float(np.clip(thr, 0.0, 255.0))

    pts = [(float(x), float(y)) for (x, y) in cross_centres]
    if not pts:
        return []

    if bgr_img is None:
        # If not provided, build a quick BGR from gray (S will be ~0 everywhere; we only use V)
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # --- 1) Collect per-point local background brightness (median V) ---
    stats = []
    for (cx, cy) in pts:
        # _annulus_stats_fast must return (V_med, S_med); we only use V_med here
        V_med, _S_med = _annulus_stats_fast(bgr_img, int(round(cx)), int(round(cy)), r_in, r_out)
        if np.isfinite(V_med):
            stats.append((cx, cy, float(V_med)))
    if not stats:
        return []

    # --- 2) Brightness-only split: bright => LE, dark => strut candidate ---
    LE_points: List[Tuple[float, float, str]] = []
    strut_candidates: List[Tuple[float, float]] = []
    for (cx, cy, V_med) in stats:
        if V_med >= thr:
            LE_points.append((cx, cy, "LE"))
        else:
            strut_candidates.append((cx, cy))

    # --- 3) Cluster strut candidates by X and assign ids 0..7 ---
    strut_candidates.sort(key=lambda p: p[0])
    clusters: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for pt in strut_candidates:
        if not cur or abs(pt[0] - cur[0][0]) < horizontal_spacing:
            cur.append(pt)
        else:
            clusters.append(cur)
            cur = [pt]
    if cur:
        clusters.append(cur)

    # Drop too-small columns (likely noise) and reclassify them as LE
    filtered_clusters: List[List[Tuple[float, float]]] = []
    at_least = max(1, int(min_points_per_strut))
    for col in clusters:
        if len(col) < at_least:
            for (cx, cy) in col:
                LE_points.append((cx, cy, "LE"))
        else:
            filtered_clusters.append(col)

    labeled_struts: List[Tuple[float, float, str]] = []
    for i, col in enumerate(filtered_clusters):
        sid = str(min(i, 7))
        for (cx, cy) in col:
            labeled_struts.append((cx, cy, sid))

    return LE_points + labeled_struts



# ============================ ArUco reference table ============================

def _pick_marker_ids(df: pd.DataFrame) -> Tuple[int, int, int]:
    """RELEVANT FUNCTION INPUTS:
    - df: DataFrame with columns ['type' in {'4x4','7x7'}, 'aruco_id', ...]

    RETURNS:
    - (id_a, id_b, id7): two 4×4 ids (prefer 0 & 1) and one 7×7 id (prefer 1)

    Raises:
    - ValueError if fewer than two distinct 4×4 ids or no 7×7 id exist.
    """
    ids_4x4 = sorted(df.loc[df["type"] == "4x4", "aruco_id"].unique())
    if len(ids_4x4) < 2:
        raise ValueError("Need at least two distinct 4x4 aruco ids.")
    id_a, id_b = (0, 1) if (0 in ids_4x4 and 1 in ids_4x4) else (ids_4x4[0], ids_4x4[1])

    ids_7x7 = sorted(df.loc[df["type"] == "7x7", "aruco_id"].unique())
    if len(ids_7x7) == 0:
        raise ValueError("Need at least one 7x7 aruco id.")
    id7 = 1 if 1 in ids_7x7 else ids_7x7[0]
    return int(id_a), int(id_b), int(id7)


def _prep_track(
    df: pd.DataFrame,
    marker_type: str,
    marker_id: int,
    suffix: str,
    full_index: np.ndarray,
) -> pd.DataFrame:
    """RELEVANT FUNCTION INPUTS:
    - df: ArUco detections with ['type','aruco_id','frame_idx','x','y','z'] columns
    - marker_type: '4x4' or '7x7'
    - marker_id: specific ArUco id to extract
    - suffix: string to append to columns (e.g. '4x4a')
    - full_index: array of frame indices to reindex/interpolate to

    RETURNS:
    - DataFrame indexed by full_index with columns [x_suffix, y_suffix, z_suffix] (interpolated & filled).
    """
    track = (
        df[(df["type"] == marker_type) & (df["aruco_id"] == marker_id)]
        .sort_values("frame_idx")
        .drop_duplicates("frame_idx")[["frame_idx", "x", "y", "z"]]
        .set_index("frame_idx")
    )
    track = track.reindex(full_index).interpolate(method="linear", limit_direction="both").ffill().bfill()
    track.columns = [f"{c}_{suffix}" for c in track.columns]
    return track


def build_aruco_reference_table(aruco_df: pd.DataFrame, frames_to_cover: Iterable[int]) -> pd.DataFrame:
    """RELEVANT FUNCTION INPUTS:
    - aruco_df: DataFrame of ArUco 3D detections (columns: type, aruco_id, frame_idx, x, y, z)
    - frames_to_cover: iterable of frame indices to cover (e.g., union of detected frames)

    RETURNS:
    - DataFrame indexed by frame_idx with columns:
        ['x_4x4a','y_4x4a','z_4x4a', 'x_4x4b','y_4x4b','z_4x4b', 'x_7x7','y_7x7','z_7x7']
      Optionally low-pass filtered if LOWPASS and LPF_ARUCO_TRACKS are True.

    Notes:
    - Assign its result to the module-global `aruco_ref_table` in your pipeline if you want
      `get_aruco_transform` to use it.
    """
    id_a, id_b, id7 = _pick_marker_ids(aruco_df)
    frames = np.asarray(list(frames_to_cover), dtype=int)
    if frames.size == 0:
        raise ValueError("frames_to_cover is empty.")
    full_index = np.arange(int(frames.min()), int(frames.max()) + 1, dtype=int)

    t_a = _prep_track(aruco_df, "4x4", id_a, "4x4a", full_index)
    t_b = _prep_track(aruco_df, "4x4", id_b, "4x4b", full_index)
    t7  = _prep_track(aruco_df, "7x7", id7, "7x7",  full_index)

    wide = t_a.join(t_b, how="outer").join(t7, how="outer")
    wide = wide.interpolate(method="linear", limit_direction="both").ffill().bfill()

    if LOWPASS and LPF_ARUCO_TRACKS:
        cols = [c for c in wide.columns if c.startswith(("x_", "y_", "z_"))]
        # zero-phase-like smoothing via centered rolling mean
        wide[cols] = wide[cols].rolling(window=7, center=True, min_periods=1).mean()

    wide.index.name = "frame_idx"
    return wide


# ================================ Pose helpers ================================

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - q: quaternion [w, x, y, z]

    RETURNS:
    - unit quaternion with non-negative scalar component (to avoid sign flips)
    """
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    return q


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - R: 3×3 rotation matrix

    RETURNS:
    - quaternion [w, x, y, z]
    """
    m = R
    t = float(np.trace(m))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return _quat_normalize(np.array([w, x, y, z], dtype=float))


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - q: quaternion [w, x, y, z]

    RETURNS:
    - 3×3 rotation matrix
    """
    w, x, y, z = _quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - q0, q1: unit quaternions [w, x, y, z]
    - t: interpolation parameter in [0, 1]

    RETURNS:
    - unit quaternion slerp(q0, q1, t)
    """
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize((1.0 - t) * q0 + t * q1)
    theta = np.arccos(dot)
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return _quat_normalize(s0 * q0 + s1 * q1)


def get_aruco_transform(
    frame_idx: int,
    lowpass: Optional[bool] = None,
    alpha: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """RELEVANT FUNCTION INPUTS:
    - frame_idx: row index into the global `aruco_ref_table`
    - lowpass: if True, smooth pose vs. previous frames (defaults to LOWPASS & LPF_POSE)
    - alpha: EMA factor for origin; pose SLERP uses beta = 1 - alpha (defaults to ALPHA)

    RETURNS:
    - (R_axes, origin), or (None, None) if frame_idx not in `aruco_ref_table`
      * R_axes: 3×3 rotation whose columns are [x_hat, y_hat, z_hat]
      * origin: 3-vector: chosen origin (center of the 7×7 marker)

    Behavior:
    - Origin O := center of 7×7 marker.
    - x_hat := normalize( p7 - midpoint(p4a, p4b) )  (keeps your previous direction)
    - y_raw := p4b - p4a;  z_hat := normalize( x_hat × y_raw )
    - y_hat := normalize( z_hat × x_hat )  (right-handed; re-orthonormalized).
    - Optional smoothing: SLERP between previous & current R, EMA for origin.
    """
    global filtered_axes

    # Defaults to module-level switches
    if lowpass is None:
        lowpass = LOWPASS and LPF_POSE
    if alpha is None:
        alpha = ALPHA

    if aruco_ref_table is None or frame_idx not in aruco_ref_table.index:
        return None, None

    r = aruco_ref_table.loc[frame_idx]
    p4a = np.array([r["x_4x4a"], r["y_4x4a"], r["z_4x4a"]], float)
    p4b = np.array([r["x_4x4b"], r["y_4x4b"], r["z_4x4b"]], float)
    p7  = np.array([r["x_7x7"],  r["y_7x7"],  r["z_7x7"] ], float)

    origin = p7
    mid_4x4 = 0.5 * (p4a + p4b)
    x_vec = p7 - mid_4x4
    y_raw = p4b - p4a

    def _safe_norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    x_hat = _safe_norm(x_vec)
    y_raw_hat = _safe_norm(y_raw)
    z_hat = _safe_norm(np.cross(x_hat, y_raw_hat))
    y_hat = _safe_norm(np.cross(z_hat, x_hat))

    # Re-orthonormalize (numerical cleanliness)
    z_hat = _safe_norm(np.cross(x_hat, y_hat))
    y_hat = _safe_norm(np.cross(z_hat, x_hat))
    R_axes = np.column_stack([x_hat, y_hat, z_hat])

    # Smoothing vs. most recent previous pose
    if lowpass and filtered_axes:
        prev_keys = [k for k in filtered_axes.keys() if k < frame_idx]
        if prev_keys:
            prev_key = max(prev_keys)
            R_prev, O_prev = filtered_axes[prev_key]
            beta = 1.0 - alpha
            q_prev = _mat_to_quat(R_prev)
            q_curr = _mat_to_quat(R_axes)
            q_blend = _quat_slerp(q_prev, q_curr, beta)
            R_axes = _quat_to_mat(q_blend)
            origin = alpha * O_prev + (1.0 - alpha) * origin

    filtered_axes[frame_idx] = (R_axes, origin)
    return R_axes, origin


def transform_points_to_aruco(points_xyz: np.ndarray, frame_idx: int, lowpass: bool = False) -> Optional[np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - points_xyz: array (N,3) of points in the *camera/world* frame used when the ArUco coords were computed
    - frame_idx: frame id to fetch the ArUco transform from `aruco_ref_table`
    - lowpass: if True, use the smoothed pose from get_aruco_transform(lowpass=True)

    RETURNS:
    - (N,3) points expressed in the per-frame ArUco axes, or None if transform not available.
    """
    R_axes, origin = get_aruco_transform(frame_idx, lowpass=lowpass)
    if R_axes is None:
        return None
    return (points_xyz - origin) @ R_axes


# ================================ 3D utilities ================================

def fit_line_3d(points: np.ndarray) -> Optional[np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - points: array (N,3) of 3D points

    RETURNS:
    - (LINE_SAMPLES, 3) segment representing the orthogonal least-squares line through points,
      or None if fewer than 2 points.

    Implementation:
    - Mean-center, SVD for principal direction, then sample between min/max projections.
    """
    if points is None or points.shape[0] < 2:
        return None
    p0 = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - p0, full_matrices=False)
    direction = Vt[0]
    t = (points - p0) @ direction
    tmin, tmax = float(t.min()), float(t.max())
    if np.isclose(tmax - tmin, 0.0):
        tmin, tmax = -0.5, 0.5
    ts = np.linspace(tmin, tmax, LINE_SAMPLES)
    return p0 + np.outer(ts, direction)



def _pairwise_dist(P: np.ndarray) -> np.ndarray:
    diffs = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diffs, axis=2)

def _path_len(P: np.ndarray) -> float:
    if P is None or P.shape[0] < 2:
        return 0.0
    d = P[1:] - P[:-1]
    return float(np.linalg.norm(d, axis=1).sum())

def _two_opt_locked(order: List[int], D: np.ndarray, s: int, t: int) -> List[int]:
    """2-opt improvement while keeping endpoints s and t fixed (open path)."""
    n = len(order)
    if n < 4:
        return order
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 3):
            for j in range(i + 2, n - 1):
                a, b = order[i], order[i + 1]
                c, d = order[j], order[j + 1]
                # keep endpoints locked
                if (a == s) or (d == t):
                    pass
                old = D[a, b] + D[c, d]
                new = D[a, c] + D[b, d]
                if new + 1e-12 < old:
                    order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                    improved = True
    return order


# --------------------------- geometry ops ---------------------------

def fit_line_3d(points: np.ndarray, samples: int = LINE_SAMPLES) -> Optional[np.ndarray]:
    """Orthogonal least-squares 3D line through points. Returns sampled segment."""
    if points is None or points.shape[0] < 2:
        return None
    P = np.asarray(points, float)
    p0 = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - p0, full_matrices=False)
    d = Vt[0]
    t = (P - p0) @ d
    tmin, tmax = float(t.min()), float(t.max())
    if np.isclose(tmax - tmin, 0.0):
        tmin, tmax = -0.5, 0.5
    ts = np.linspace(tmin, tmax, samples)
    return p0 + np.outer(ts, d)

def _pca_dir(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or points.shape[0] < 2:
        return None
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    n = np.linalg.norm(d)
    return d / (n + 1e-12)

def _best_fit_plane_normal(P: np.ndarray) -> np.ndarray:
    Q = np.asarray(P, float)
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[-1]
    return n / (np.linalg.norm(n) + 1e-12)


# --------------------------- LE path ---------------------------

def le_shortest_path_3d(points_3d: np.ndarray,
                        endpoints_k: int = 10,
                        jump_factor: float = 2.5) -> Optional[np.ndarray]:
    """
    Reconstruct LE as an open path with endpoints chosen from the K lowest-Z points.

    - Endpoints: pick K points with the smallest Z (closest to camera); try all pairs.
    - Path: greedy nearest extension with last endpoint fixed, followed by 2-opt (locked).
    - Orientation: ensure path goes from lower-Z to higher-Z.

    Args:
        points_3d: (N,3) LE points in world/camera coordinates.
        endpoints_k: how many low-Z candidates to consider for endpoints.
        jump_factor: (kept for API symmetry; not used here but kept for future constraints)

    Returns:
        (M,3) ordered LE polyline, or None if <2 points.
    """
    if points_3d is None or points_3d.shape[0] < 2:
        return None

    P = np.asarray(points_3d, float)
    n = P.shape[0]
    D = _pairwise_dist(P)

    k = max(2, min(endpoints_k, n))
    cand = np.argsort(P[:, 2])[:k]  # lower Z ~ closer to camera
    best_cost, best_path = np.inf, None

    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            s, t = int(cand[a]), int(cand[b])
            used = np.zeros(n, dtype=bool)
            used[s] = True
            order = [s]
            while len(order) < n - 1:
                last = order[-1]
                rem = np.where(~used)[0]
                rem = rem[rem != t]
                if rem.size == 0:
                    break
                nxt = int(rem[np.argmin(D[last, rem])])
                order.append(nxt)
                used[nxt] = True
            order.append(t)
            order = _two_opt_locked(order, D, s, t)
            path = P[np.array(order)]
            cost = _path_len(path)
            if cost < best_cost:
                best_cost, best_path = cost, path

    if best_path is not None and best_path[0, 2] > best_path[-1, 2]:
        best_path = best_path[::-1]
    return best_path


# --------------------------- reference frame from struts ---------------------------

def compute_ref_frame_from_center_struts(
    LE_path_world: Optional[np.ndarray],
    P3_world: Optional[np.ndarray],
    P4_world: Optional[np.ndarray],
    *,
    global_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    force_flip_x: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build a right-handed (x,y,z) frame from center struts 3 & 4.

    Steps:
      1) z_hat := normal of best-fit plane through all points of strut3 ∪ strut4.
         Ensure z_hat aligns with global_up.
      2) x_hat := sum of PCA directions of strut3 and strut4, projected onto that plane.
         If LE_path is given, flip x_hat so it aligns with the LE direction in-plane.
      3) y_hat := z_hat × x_hat. Re-orthonormalize.

    Returns:
      (R_axes, origin), where columns of R_axes are [x_hat, y_hat, z_hat],
      and origin is the midpoint of the two strut centroids.
    """
    if P3_world is None or P4_world is None or P3_world.shape[0] < 2 or P4_world.shape[0] < 2:
        return None, None

    all_pts = np.vstack([P3_world, P4_world])
    z_hat = _best_fit_plane_normal(all_pts)
    global_up = np.asarray(global_up, float)
    global_up /= (np.linalg.norm(global_up) + 1e-12)
    if np.dot(z_hat, global_up) < 0.0:
        z_hat = -z_hat

    d3 = _pca_dir(P3_world)
    d4 = _pca_dir(P4_world)
    if d3 is None or d4 is None:
        return None, None
    if np.dot(d3, d4) < 0.0:
        d4 = -d4

    x_raw = d3 + d4
    x_plane = x_raw - np.dot(x_raw, z_hat) * z_hat
    if np.linalg.norm(x_plane) < 1e-12:
        # fallback axis in plane
        tmp = np.array([1.0, 0.0, 0.0], float)
        if abs(np.dot(tmp, z_hat)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], float)
        x_plane = tmp - np.dot(tmp, z_hat) * z_hat
    x_hat = x_plane / (np.linalg.norm(x_plane) + 1e-12)

    if LE_path_world is not None and LE_path_world.shape[0] >= 2:
        le_vec = LE_path_world[-1] - LE_path_world[0]
        le_proj = le_vec - np.dot(le_vec, z_hat) * z_hat
        if np.linalg.norm(le_proj) > 1e-12 and np.dot(x_hat, le_proj) < 0.0:
            x_hat = -x_hat

    if force_flip_x:
        x_hat = -x_hat

    y_hat = np.cross(z_hat, x_hat)
    y_hat /= (np.linalg.norm(y_hat) + 1e-12)
    z_hat = np.cross(x_hat, y_hat)
    z_hat /= (np.linalg.norm(z_hat) + 1e-12)
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= (np.linalg.norm(y_hat) + 1e-12)

    origin = 0.5 * (P3_world.mean(axis=0) + P4_world.mean(axis=0))
    R_axes = np.column_stack([x_hat, y_hat, z_hat])
    return R_axes, origin


# --------------------------- smoothing ---------------------------

def smooth_points_by_matching(
    curr_pts: np.ndarray,
    prev_pts: Optional[np.ndarray],
    alpha: float = 0.9,
    max_dist: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EMA smoothing by nearest-neighbor matching.
    Returns (smoothed_points, matched_mask) with shape (M,), True if blended.
    """
    if curr_pts is None or curr_pts.size == 0:
        return curr_pts, np.zeros((0,), dtype=bool)
    if prev_pts is None or prev_pts.size == 0:
        return curr_pts, np.zeros((curr_pts.shape[0],), dtype=bool)

    diffs = curr_pts[:, None, :] - prev_pts[None, :, :]
    D = np.linalg.norm(diffs, axis=2)
    idxs = np.argmin(D, axis=1)
    dmin = D[np.arange(D.shape[0]), idxs]
    matched = dmin <= max_dist

    smoothed = curr_pts.copy()
    smoothed[matched] = alpha * prev_pts[idxs[matched]] + (1.0 - alpha) * curr_pts[matched]
    return smoothed, matched
