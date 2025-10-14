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
    horizontal_spacing: float = 150.0,
    r_in: int = 6, r_out: int = 18,          # annulus radii
    le_v_bias: float = 12.0,                 # lower V threshold by this many units for LE
    le_s_max: float = 70.0,                  # if S_med <= this and V_med >= le_v_floor -> LE
    le_v_floor: float = 100.0,               # minimum V for the saturation rule to apply
    min_points_per_strut: int = 2,           # NEW: require at least this many points per strut column
) -> List[Tuple[float, float, Union[int, str]]]:
    """
    Labels crosses as 'LE' (white cloth) or strut ids '0'..'7'.
    - Otsu on per-point background V to split bright vs dark.
    - Bias V-threshold downward for LE (le_v_bias).
    - Color tie-breaker: low S + bright V ⇒ LE.
    - NEW: a strut column must contain at least `min_points_per_strut` points; singletons are reclassified as LE.
    """
    pts = [(float(x), float(y)) for (x, y) in cross_centres]
    if not pts:
        return []

    if bgr_img is None:
        # if not provided, build a quick BGR from gray (S will be ~0 everywhere)
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # --- 1) collect per-point background stats (median V, S in an annulus) ---
    stats = []
    for (cx, cy) in pts:
        V_med, S_med = _annulus_stats_fast(bgr_img, int(round(cx)), int(round(cy)), r_in, r_out)
        if np.isfinite(V_med):
            stats.append((cx, cy, V_med, (S_med if np.isfinite(S_med) else 0.0)))
    if not stats:
        return []

    V_vals = np.array([v for _, _, v, _ in stats], dtype=np.float32)

    # --- 2) Otsu on V ---
    V_u8 = np.clip(V_vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    v_thr, _ = cv2.threshold(V_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thr = float(v_thr)

    # --- 3) biased threshold + hysteresis + color tie-breaker ---
    v_thr_le = max(0.0, v_thr - le_v_bias)   # LOWER threshold for LE
    margin   = 4.0
    v_hi, v_lo = v_thr_le + margin, v_thr_le - margin

    LE_points: List[Tuple[float, float, str]] = []
    strut_candidates: List[Tuple[float, float]] = []

    for (cx, cy, V_med, S_med) in stats:
        if V_med >= v_hi:
            LE_points.append((cx, cy, "LE"))
            continue
        if V_med <= v_lo:
            strut_candidates.append((cx, cy))
            continue

        # tie-breaker with color: low S and reasonably bright V -> LE
        if (S_med <= le_s_max) and (V_med >= le_v_floor):
            LE_points.append((cx, cy, "LE"))
        else:
            if V_med >= v_thr_le:
                LE_points.append((cx, cy, "LE"))
            else:
                strut_candidates.append((cx, cy))

    # --- 4) cluster strut candidates by x and assign ids 0..7 ---
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

    # NEW: drop singleton (or too-small) clusters and reclassify them as LE
    filtered_clusters: List[List[Tuple[float, float]]] = []
    for col in clusters:
        if len(col) < max(1, int(min_points_per_strut)):
            # reclassify every point in this tiny "column" as LE to avoid shifting ids
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
    """Pairwise Euclidean distances for points P (N,3) -> (N,N)."""
    diffs = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _two_opt_improve(order: List[int], D: np.ndarray) -> List[int]:
    """2-opt local improvement for a path given a distance matrix D."""
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
                old = D[a, b] + D[c, d]
                new = D[a, c] + D[b, d]
                if new + 1e-12 < old:
                    order[i + 1 : j + 1] = reversed(order[i + 1 : j + 1])
                    improved = True
    return order


def le_shortest_path_3d(points_3d: np.ndarray, exact_limit: int = 12) -> Optional[np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - points_3d: array (N,3)
    - exact_limit: for N <= exact_limit, solve exactly (Held–Karp DP); else greedy + 2-opt

    RETURNS:
    - points_3d re-ordered to minimize total path length (open path), or None if N < 2.
    """
    n = int(points_3d.shape[0]) if points_3d is not None else 0
    if n < 2:
        return None
    if n == 2:
        return points_3d.copy()

    D = _pairwise_dist(points_3d)

    if n <= exact_limit:
        # Held–Karp dynamic programming for shortest Hamiltonian path (open)
        ALL = 1 << n
        dp = np.full((ALL, n), np.inf)
        parent = np.full((ALL, n), -1, dtype=int)
        for j in range(n):
            dp[1 << j, j] = 0.0
        for mask in range(ALL):
            js = np.nonzero([(mask >> k) & 1 for k in range(n)])[0]
            if js.size <= 1:
                continue
            for j in js:
                pmask = mask ^ (1 << j)
                iset = np.nonzero([(pmask >> k) & 1 for k in range(n)])[0]
                if iset.size == 0:
                    continue
                costs = dp[pmask, iset] + D[iset, j]
                kidx = int(np.argmin(costs))
                val = float(costs[kidx])
                if val < dp[mask, j]:
                    dp[mask, j] = val
                    parent[mask, j] = int(iset[kidx])
        full = ALL - 1
        end = int(np.argmin(dp[full, :]))
        order: List[int] = []
        mask = full
        j = end
        while j != -1:
            order.append(j)
            pj = parent[mask, j]
            mask ^= (1 << j)
            j = pj
        order = order[::-1]
    else:
        # Heuristic: farthest pair seed + greedy nearest extension, then 2-opt
        i, j = np.unravel_index(np.argmax(D), D.shape)
        start = int(i)
        used = np.zeros(n, dtype=bool)
        used[start] = True
        order = [start]
        for _ in range(n - 1):
            last = order[-1]
            candidates = np.where(~used)[0]
            nxt = int(candidates[np.argmin(D[last, candidates])])
            order.append(nxt)
            used[nxt] = True
        order = _two_opt_improve(order, D)

    return points_3d[np.array(order)]


def smooth_points_by_matching(
    curr_pts: np.ndarray,
    prev_pts: Optional[np.ndarray],
    alpha: float = 0.9,
    max_dist: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - curr_pts: (M,3) current points
    - prev_pts: (N,3) previous points (can be None or empty)
    - alpha: EMA factor; smoothed = alpha*prev + (1-alpha)*curr (when matched)
    - max_dist: max Euclidean distance for a valid match (same units as points)

    RETURNS:
    - (smoothed_pts, matched_mask) where matched_mask is (M,) boolean (True if blended)

    Notes:
    - Brute-force nearest-neighbor (fine for small M,N). Unmatched curr points are unchanged.
    """
    if curr_pts is None or curr_pts.size == 0:
        return curr_pts, np.zeros((0,), dtype=bool)
    if prev_pts is None or prev_pts.size == 0:
        return curr_pts, np.zeros((curr_pts.shape[0],), dtype=bool)

    diffs = curr_pts[:, None, :] - prev_pts[None, :, :]  # (M,N,3)
    D = np.linalg.norm(diffs, axis=2)                    # (M,N)
    idxs = np.argmin(D, axis=1)                          # (M,)
    dmin = D[np.arange(D.shape[0]), idxs]
    matched = dmin <= max_dist

    smoothed = curr_pts.copy()
    smoothed[matched] = alpha * prev_pts[idxs[matched]] + (1.0 - alpha) * curr_pts[matched]
    return smoothed, matched


# ------------------------------ usage reminder -------------------------------
# Typical usage in your pipeline:
#   aruco_ref_table = build_aruco_reference_table(aruco_coords_df, frames_to_cover=unique_frames)
#   R_axes, origin = get_aruco_transform(frame_idx, lowpass=True)
#   points_in_aruco = transform_points_to_aruco(points_xyz, frame_idx, lowpass=True)


# ------------------------------- module main ---------------------------------
if __name__ == "__main__":
    # Utilities module; nothing to run directly.
    # Keep imports side-effect free; define functions only.
    pass
