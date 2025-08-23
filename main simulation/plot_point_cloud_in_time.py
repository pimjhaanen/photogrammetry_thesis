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
OUTPUT_VIDEO = "output/test_02_camera_frame.mp4"
FPS = 30
VIDEO_SIZE = (800, 800)

# Transform / filtering
USE_ARUCO_FRAME = True
LOWPASS_AXES = False          # option kept; default False as requested
ALPHA = 0.9                   # smoothing alpha if enabled

# Plot styling
POINT_SIZE = 60
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
    wide.index.name = "frame_idx"
    return wide

aruco_ref_table = build_aruco_reference_table(aruco_coords_df, frames_to_cover=unique_frames)

filtered_axes = {}   # optional lowpass cache: frame_idx -> (R_axes, origin)

def get_aruco_transform(frame_idx, lowpass=False, alpha=ALPHA):
    """
    Aruco frame per frame:
      Origin O = midpoint of two 4x4 markers.
      x_hat = normalize(7x7 - O)
      y_raw_hat = normalize(4x4b - 4x4a)
      z_hat = normalize(cross(x_hat, y_raw_hat))
      y_hat = normalize(cross(z_hat, x_hat))   # right-handed & continuous
    Returns (R_axes, origin) with columns [x_hat, y_hat, z_hat].
    """
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

    if lowpass and (frame_idx - 1) in filtered_axes:
        R_prev, O_prev = filtered_axes[frame_idx - 1]
        R_sm = alpha * R_prev + (1.0 - alpha) * R_axes
        x = _safe_norm(R_sm[:, 0])
        z = _safe_norm(np.cross(x, R_sm[:, 1]))
        y = _safe_norm(np.cross(z, x))
        R_axes = np.column_stack([x, y, z])
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

# --- LE: nearest-neighbor polyline (no interpolation) ---

def le_nearest_neighbor_path(points):
    """
    Connect LE points with straight segments in a nearest-neighbor order.
    Start from one endpoint (chosen from the farthest pair), then greedily
    attach the nearest unvisited point. Returns ordered (m,3) points.
    """
    n = points.shape[0]
    if n < 2:
        return None
    if n == 2:
        return points.copy()

    # pairwise distances
    diffs = points[:, None, :] - points[None, :, :]
    D = np.linalg.norm(diffs, axis=2)

    # choose farthest pair as (approx) endpoints; start at one of them
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start = int(i)

    order = [start]
    used = np.zeros(n, dtype=bool)
    used[start] = True

    for _ in range(n - 1):
        last = order[-1]
        # nearest unvisited
        candidates = np.where(~used)[0]
        nxt = candidates[np.argmin(D[last, candidates])]
        order.append(int(nxt))
        used[nxt] = True

    return points[np.array(order)]

# --- Rendering ---

def render_frame(points, line_segments):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # scatter all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=POINT_SIZE)

    # draw lines/curves
    for seg in line_segments:
        if seg is None or len(seg) == 0:
            continue
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=2)

    ax.set_xlabel("X (flight dir.)")
    ax.set_ylabel("Y (spanwise)")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.view_init(elev=30, azim=-45)
    fig.tight_layout()
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, VIDEO_SIZE)
    plt.close(fig)
    return img

# === MAIN LOOP ===
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

    line_segments = []

    # 0..7: best-fit straight lines (kept as requested)
    for k in range(8):
        grp = frame_df[frame_df["marker_str"] == str(k)]
        if grp.shape[0] >= 2:
            line_segments.append(fit_line_3d(grp[["x", "y", "z"]].values))
        else:
            line_segments.append(None)

    # LE: straight segments connecting the detected points in NN order (no interpolation)
    le_grp = frame_df[frame_df["marker_str"].str.upper() == "LE"]
    if le_grp.shape[0] >= 2:
        le_points = le_grp[["x", "y", "z"]].values
        le_path = le_nearest_neighbor_path(le_points)
        line_segments.append(le_path)   # plotted directly: connects raw points
    else:
        line_segments.append(None)

    # Render
    img = render_frame(pts, line_segments)
    out.write(img)
    frame_count += 1

out.release()
print(f"[✓] Total frames written: {frame_count}")
