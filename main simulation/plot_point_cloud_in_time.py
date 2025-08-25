import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === CONFIG ===
INPUT_CSV = "output/3d_coordinates_25_06_test2_merged.csv"
ARUCO_COORDS_CSV = "output/aruco_coordinates_25_06_test2_merged.csv"
OUTPUT_VIDEO = "output/test_02_camera_frame_LPF.mp4"
FPS = 30
VIDEO_SIZE = (800, 800)

# ------------------- Smoothing Options (set once) -------------------
LOWPASS = True            # master toggle enabling smoothing behaviors
ALPHA   = 0.9            # previous-frame weight; higher = smoother (pose)
USE_ARUCO_FRAME = True   # transform to ArUco frame

# Choose *where* to apply smoothing (combine as you wish):
LPF_ARUCO_TRACKS = False  # pre-axes smoothing of raw ArUco coords (can cause "flyover" if True)
LPF_POSE         = True   # smooth pose (R,t) with SLERP for rotation and EMA for origin
SMOOTH_POINTS    = True   # smooth transformed points by matching to previous frame

# Post-transform point smoothing params (used iff SMOOTH_POINTS)
POINT_ALPHA  = 0.9        # previous-frame weight for point EMA
MATCH_RADIUS = 0.35       # meters: max distance to match a point to previous frame

# Back-compat alias so existing calls keep working (affects transform_points_to_aruco)
LOWPASS_AXES = (LOWPASS and LPF_POSE)

# ------------------- Plot styling -------------------
POINT_SIZE = 20
LINE_SAMPLES = 60             # for PCA lines (ids 0..7), not used for LE

# === LOAD DATA ===
cross_df = pd.read_csv(INPUT_CSV)
aruco_coords_df = pd.read_csv(ARUCO_COORDS_CSV)

required_cross_cols = {"frame_idx", "marker_id", "x", "y", "z"}
required_aruco_cols = {"frame_idx", "type", "aruco_id", "x", "y", "z"}
if not required_cross_cols.issubset(cross_df.columns):
    raise ValueError(f"INPUT_CSV missing columns: {required_cross_cols - set(cross_df.columns)}")
if not required_aruco_cols.issubset(aruco_coords_df.columns):
    raise ValueError(f"ARUCO_COORDS_CSV missing columns: {required_aruco_cols - set(aruco_coords_df.columns)}")

unique_frames = np.sort(cross_df["frame_idx"].unique())

# === VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, VIDEO_SIZE)

# --- Helpers for Aruco reference frame ---

def _pick_marker_ids(df):
    """Pick two 4x4 ids (prefer 0,1) and one 7x7 id (prefer 1)."""
    ids_4x4 = sorted(df.loc[df["type"] == "4x4", "aruco_id"].unique())
    if len(ids_4x4) < 2:
        raise ValueError("Need at least two distinct 4x4 aruco ids.")
    if 0 in ids_4x4 and 1 in ids_4x4:
        id_a, id_b = 0, 1
    else:
        id_a, id_b = ids_4x4[0], ids_4x4[1]

    ids_7x7 = sorted(df.loc[df["type"] == "7x7", "aruco_id"].unique())
    if len(ids_7x7) == 0:
        raise ValueError("Need at least one 7x7 aruco id.")
    id7 = 1 if 1 in ids_7x7 else ids_7x7[0]
    return id_a, id_b, id7

def _prep_track(df, marker_type, marker_id, suffix, full_index):
    """Interpolate one marker track to full_index; return x_suffix,y_suffix,z_suffix."""
    track = (df[(df["type"] == marker_type) & (df["aruco_id"] == marker_id)]
               .sort_values("frame_idx")
               .drop_duplicates("frame_idx")[["frame_idx", "x", "y", "z"]]
               .set_index("frame_idx"))
    track = track.reindex(full_index).interpolate(method="linear", limit_direction="both").ffill().bfill()
    track.columns = [f"{c}_{suffix}" for c in track.columns]
    return track

def build_aruco_reference_table(aruco_df, frames_to_cover):
    id_a, id_b, id7 = _pick_marker_ids(aruco_df)
    min_idx = int(frames_to_cover.min())
    max_idx = int(frames_to_cover.max())
    full_index = np.arange(min_idx, max_idx + 1)

    t_a = _prep_track(aruco_df, "4x4", id_a, "4x4a", full_index)
    t_b = _prep_track(aruco_df, "4x4", id_b, "4x4b", full_index)
    t7  = _prep_track(aruco_df, "7x7", id7, "7x7", full_index)

    wide = t_a.join(t_b, how="outer").join(t7, how="outer")
    wide = wide.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # --- optional low-pass on raw ArUco coordinates (pre-axes) ---
    if LOWPASS and LPF_ARUCO_TRACKS:
        cols = [c for c in wide.columns if c.startswith(("x_", "y_", "z_"))]
        # Zero-phase (centered) smoothing: less lag than causal, but still blends time
        wide[cols] = wide[cols].rolling(window=7, center=True, min_periods=1).mean()

    wide.index.name = "frame_idx"
    return wide

aruco_ref_table = build_aruco_reference_table(aruco_coords_df, frames_to_cover=unique_frames)

# --- Quaternion helpers for pose SLERP ---
def _quat_normalize(q):
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:  # keep scalar non-negative to avoid flips
        q = -q
    return q

def _mat_to_quat(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return _quat_normalize(np.array([w, x, y, z], dtype=float))

def _quat_to_mat(q):
    w, x, y, z = _quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

def _quat_slerp(q0, q1, t):
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot > 0.9995:
        return _quat_normalize((1-t)*q0 + t*q1)
    theta = np.arccos(dot)
    s0 = np.sin((1-t)*theta) / np.sin(theta)
    s1 = np.sin(t*theta) / np.sin(theta)
    return _quat_normalize(s0*q0 + s1*q1)

filtered_axes = {}   # cache: frame_idx -> (R_axes, origin)

def get_aruco_transform(frame_idx, lowpass=None, alpha=None):
    """
    Aruco frame per frame:
      Origin O = midpoint of two 4x4 markers.
      x_hat = normalize(7x7 - O)
      y_raw_hat = normalize(4x4b - 4x4a)
      z_hat = normalize(cross(x_hat, y_raw_hat))
      y_hat = normalize(cross(z_hat, x_hat))   # right-handed & continuous
    Returns (R_axes, origin) with columns [x_hat, y_hat, z_hat].
    """
    # Use master settings by default
    if lowpass is None:
        lowpass = LOWPASS and LPF_POSE
    if alpha is None:
        alpha = ALPHA

    if frame_idx not in aruco_ref_table.index:
        return None, None
    r = aruco_ref_table.loc[frame_idx]
    p4a = np.array([r["x_4x4a"], r["y_4x4a"], r["z_4x4a"]], float)
    p4b = np.array([r["x_4x4b"], r["y_4x4b"], r["z_4x4b"]], float)
    p7  = np.array([r["x_7x7"],  r["y_7x7"],  r["z_7x7"] ], float)

    origin = 0.5 * (p4a + p4b)
    x_vec = p7 - origin
    y_raw = p4b - p4a

    def _safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    x_hat = _safe_norm(x_vec)
    y_raw_hat = _safe_norm(y_raw)
    z_hat = _safe_norm(np.cross(x_hat, y_raw_hat))
    y_hat = _safe_norm(np.cross(z_hat, x_hat))

    # Re-orthonormalize
    z_hat = _safe_norm(np.cross(x_hat, y_hat))
    y_hat = _safe_norm(np.cross(z_hat, x_hat))
    R_axes = np.column_stack([x_hat, y_hat, z_hat])

    # --- Pose smoothing with SLERP (works with gaps) ---
    if lowpass and filtered_axes:
        prev_keys = [k for k in filtered_axes.keys() if k < frame_idx]
        if prev_keys:
            prev_key = max(prev_keys)
            R_prev, O_prev = filtered_axes[prev_key]
            beta = 1.0 - alpha  # step toward current pose
            q_prev = _mat_to_quat(R_prev)
            q_curr = _mat_to_quat(R_axes)
            q_blend = _quat_slerp(q_prev, q_curr, beta)
            R_axes = _quat_to_mat(q_blend)
            origin = alpha * O_prev + (1.0 - alpha) * origin

    filtered_axes[frame_idx] = (R_axes, origin)
    return R_axes, origin

def transform_points_to_aruco(points_xyz, frame_idx, lowpass=False):
    R_axes, origin = get_aruco_transform(frame_idx, lowpass=lowpass)
    if R_axes is None:
        return None
    return (points_xyz - origin) @ R_axes

# --- Fitting utilities ---

def fit_line_3d(points):
    """
    Orthogonal least-squares 3D line via SVD.
    Returns a (LINE_SAMPLES, 3) segment spanning projections of the input points.
    """
    if points is None or points.shape[0] < 2:
        return None
    p0 = points.mean(axis=0)
    U, S, Vt = np.linalg.svd(points - p0)
    direction = Vt[0]
    t = (points - p0) @ direction
    tmin, tmax = t.min(), t.max()
    if np.isclose(tmax - tmin, 0.0):
        tmin, tmax = -0.5, 0.5
    ts = np.linspace(tmin, tmax, LINE_SAMPLES)
    return p0 + np.outer(ts, direction)

# --- LE path construction helpers ---

def _pairwise_dist(P):
    diffs = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diffs, axis=2)

def _two_opt_improve(order, D):
    n = len(order)
    if n < 4:
        return order
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 3):
            for j in range(i + 2, n - 1):
                a, b = order[i], order[i+1]
                c, d = order[j], order[j+1]
                old = D[a, b] + D[c, d]
                new = D[a, c] + D[b, d]
                if new + 1e-12 < old:
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    improved = True
    return order

def le_shortest_path_3d(points_3d, exact_limit=12):
    """
    Return 3D points ordered to minimize total 3D path length (endpoints degree 1).
    Uses exact Held–Karp DP for n <= exact_limit, else greedy+2opt heuristic.
    """
    n = points_3d.shape[0]
    if n < 2:
        return None
    if n == 2:
        return points_3d.copy()

    D = _pairwise_dist(points_3d)

    if n <= exact_limit:
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
                kidx = np.argmin(costs)
                val = costs[kidx]
                if val < dp[mask, j]:
                    dp[mask, j] = val
                    parent[mask, j] = iset[kidx]
        full = ALL - 1
        end = int(np.argmin(dp[full, :]))
        order = []
        mask = full
        j = end
        while j != -1:
            order.append(j)
            pj = parent[mask, j]
            mask ^= (1 << j)
            j = pj
        order = order[::-1]
    else:
        i, j = np.unravel_index(np.argmax(D), D.shape)
        start = int(i)
        used = np.zeros(n, dtype=bool)
        used[start] = True
        order = [start]
        for _ in range(n - 1):
            last = order[-1]
            candidates = np.where(~used)[0]
            nxt = candidates[np.argmin(D[last, candidates])]
            order.append(int(nxt))
            used[nxt] = True
        order = _two_opt_improve(order, D)

    return points_3d[np.array(order)]

# --- Post-transform point smoothing (matching-based) ---

def smooth_points_by_matching(curr_pts, prev_pts, alpha=0.9, max_dist=0.35):
    """
    Match each current point to its nearest previous point within max_dist,
    then EMA-blend: p_smooth = alpha*prev + (1-alpha)*curr.
    Unmatched current points remain unchanged.
    """
    if prev_pts is None or curr_pts.size == 0:
        return curr_pts, np.zeros((curr_pts.shape[0],), dtype=bool)
    if prev_pts.size == 0:
        return curr_pts, np.zeros((curr_pts.shape[0],), dtype=bool)

    # brute-force NN (small n): (m,n,3) broadcast distances
    diffs = curr_pts[:, None, :] - prev_pts[None, :, :]
    D = np.linalg.norm(diffs, axis=2)  # (m, n)
    idxs = np.argmin(D, axis=1)
    dmin = D[np.arange(D.shape[0]), idxs]
    matched = dmin <= max_dist

    smoothed = curr_pts.copy()
    smoothed[matched] = alpha * prev_pts[idxs[matched]] + (1.0 - alpha) * curr_pts[matched]
    return smoothed, matched

# --- Rendering ---
# Plot limits
X_LIM = (-3, 4)
Y_LIM = (-6, 6)
Z_LIM = (-5, 2)

def render_frame(points, line_segments):
    """
    2x2 panel:
      [TL] 3D (original view)
      [TR] Front  (Y–Z)
      [BL] Side   (X–Z)
      [BR] Top    (X–Y)
    """
    fig = plt.figure(figsize=(8, 8))

    ax3d    = fig.add_subplot(2, 2, 1, projection='3d')  # 3D
    ax_front= fig.add_subplot(2, 2, 2)                   # Y-Z
    ax_side = fig.add_subplot(2, 2, 3)                   # X-Z
    ax_top  = fig.add_subplot(2, 2, 4)                   # X-Y

    # --- 3D ---
    for seg in line_segments:
        if seg is not None and len(seg) > 0:
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=2)
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=POINT_SIZE)
    ax3d.set_xlabel("X (flight dir.)")
    ax3d.set_ylabel("Y (spanwise)")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim(*X_LIM)
    ax3d.set_ylim(*Y_LIM)
    ax3d.set_zlim(*Z_LIM)
    ax3d.view_init(elev=30, azim=-45)
    ax3d.set_title("3D", fontsize=10)
    ax3d.grid(True)

    # --- Front (Y–Z) ---

    for seg in line_segments:
        if seg is not None and len(seg) > 0:
            ax_front.plot(seg[:, 1], seg[:, 2], linewidth=2)
    ax_front.scatter(points[:, 1], points[:, 2], c='red', s=POINT_SIZE)
    ax_front.set_xlabel("Y (spanwise)")
    ax_front.set_ylabel("Z")
    ax_front.set_xlim(*Y_LIM)
    ax_front.set_ylim(*Z_LIM)
    ax_front.set_aspect('equal', adjustable='box')
    ax_front.set_title("Front (Y–Z)", fontsize=10)
    ax_front.grid(True)

    # --- Side (X–Z) ---
    for seg in line_segments:
        if seg is not None and len(seg) > 0:
            ax_side.plot(seg[:, 0], seg[:, 2], linewidth=2)
    ax_side.scatter(points[:, 0], points[:, 2], c='red', s=POINT_SIZE)
    ax_side.set_xlabel("X (flight dir.)")
    ax_side.set_ylabel("Z")
    ax_side.set_xlim(*X_LIM)
    ax_side.set_ylim(*Z_LIM)
    ax_side.set_aspect('equal', adjustable='box')
    ax_side.set_title("Side (X–Z)", fontsize=10)
    ax_side.grid(True)

    # --- Top (X–Y) ---
    for seg in line_segments:
        if seg is not None and len(seg) > 0:
            ax_top.plot(seg[:, 0], seg[:, 1], linewidth=2)
    ax_top.scatter(points[:, 0], points[:, 1], c='red', s=POINT_SIZE)
    ax_top.set_xlabel("X (flight dir.)")
    ax_top.set_ylabel("Y (spanwise)")
    ax_top.set_xlim(*X_LIM)
    ax_top.set_ylim(*Y_LIM)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_title("Top (X–Y)", fontsize=10)
    ax_top.grid(True)

    fig.tight_layout()
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, VIDEO_SIZE)
    plt.close(fig)
    return img

# === MAIN LOOP ===
prev_points_smoothed = None  # cache for point-level EMA (post-transform)

frame_count = 0
for frame_idx in unique_frames:
    frame_df = cross_df[cross_df["frame_idx"] == frame_idx]
    if frame_df.shape[0] == 0:
        continue

    pts_world = frame_df[["x", "y", "z"]].values.astype(float)
    if USE_ARUCO_FRAME:
        pts = transform_points_to_aruco(pts_world, frame_idx, lowpass=LOWPASS_AXES)
        if pts is None:
            continue
    else:
        pts = pts_world

    # Use transformed coords for grouping/fitting
    frame_df = frame_df.copy()
    frame_df["marker_str"] = frame_df["marker_id"].astype(str)
    frame_df[["x", "y", "z"]] = pts

    # --- Drop isolated points (no neighbor within 2 m, Euclidean) ---
    xyz = frame_df[["x", "y", "z"]].to_numpy()
    if xyz.shape[0] >= 2:
        diffs = xyz[:, None, :] - xyz[None, :, :]
        D = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(D, np.inf)           # ignore self-distance
        min_dist = D.min(axis=1)              # nearest neighbor distance
        keep = min_dist <= 2.0
        frame_df = frame_df.loc[keep].copy()
    else:
        frame_df = frame_df.iloc[0:0].copy()

    # --- Optional: smooth points in ArUco frame by matching to previous frame ---
    points_to_plot = frame_df[["x", "y", "z"]].to_numpy()
    if SMOOTH_POINTS and LOWPASS and points_to_plot.size > 0:
        points_smoothed, _ = smooth_points_by_matching(
            points_to_plot, prev_points_smoothed, alpha=POINT_ALPHA, max_dist=MATCH_RADIUS
        )
        frame_df[["x", "y", "z"]] = points_smoothed
        points_to_plot = points_smoothed

    # Build line segments
    line_segments = []

    # 0..7: best-fit straight lines
    for k in range(8):
        grp = frame_df[frame_df["marker_str"] == str(k)]
        if grp.shape[0] >= 2:
            line_segments.append(fit_line_3d(grp[["x", "y", "z"]].values))
        else:
            line_segments.append(None)

    # LE: straight segments connecting detected points by *shortest 3D path*
    le_grp = frame_df[frame_df["marker_str"].str.upper() == "LE"]
    if le_grp.shape[0] >= 2:
        le_points = le_grp[["x", "y", "z"]].values
        le_path = le_shortest_path_3d(le_points, exact_limit=12)
        line_segments.append(le_path)
    else:
        line_segments.append(None)

    img = render_frame(points_to_plot, line_segments)
    out.write(img)
    frame_count += 1

    # cache smoothed points for next frame matching
    prev_points_smoothed = points_to_plot.copy()

out.release()
print(f"[✓] Total frames written: {frame_count}")
