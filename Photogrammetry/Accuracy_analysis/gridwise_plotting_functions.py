"""This code can be used to (1) transform 3D points from a camera frame to an ArUco marker frame
and (2) visualize 3D point sets, including fitting a plane, building an ideal reference pattern
on that plane (rectangular grid or 6-point trapezium), and plotting per-point distance errors
(total, in-plane, and out-of-plane)."""

from __future__ import annotations
# add near your other imports:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Tuple
import itertools
from matplotlib.ticker import FuncFormatter
# ========================= Assignment helper (Hungarian or greedy) =========================

def _greedy_assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback when SciPy isn't available.
    Greedy unique nearest-neighbour (not globally optimal, but OK for small N).
    Returns (row_indices, col_indices) of matches.
    """
    cost = cost.copy()
    n_rows, n_cols = cost.shape
    used_r = np.zeros(n_rows, dtype=bool)
    used_c = np.zeros(n_cols, dtype=bool)
    pairs = []

    # flatten and iterate by ascending cost
    flat_idx = np.argsort(cost, axis=None)
    for k in flat_idx:
        r, c = divmod(k, n_cols)
        if not used_r[r] and not used_c[c]:
            pairs.append((r, c))
            used_r[r] = True
            used_c[c] = True
        if used_r.all() or used_c.all():
            break

    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    rr, cc = zip(*pairs)
    return np.asarray(rr, dtype=int), np.asarray(cc, dtype=int)


def match_points_to_grid(pts: np.ndarray,
                         grid_pts: np.ndarray,
                         max_assign_dist: float | None = None,
                         prefer_scipy: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign measured points (pts) to ideal reference points (grid_pts) by minimizing total Euclidean distance.
    Returns:
      pts_assigned      : (K,3) measured points reordered to match reference order
      grid_assigned     : (K,3) corresponding reference points
      idx_pts_selected  : (K,) indices of selected measured points
      idx_grid_selected : (K,) indices of selected reference points (order defines the 'reference order')

    If sizes differ, only min(N_pts, N_ref) best matches are kept.
    If max_assign_dist is set, discards pairs farther than that (in meters).
    """
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert grid_pts.ndim == 2 and grid_pts.shape[1] == 3

    # Cost matrix
    # shape (N_pts, N_ref)
    diff = pts[:, None, :] - grid_pts[None, :, :]
    cost = np.linalg.norm(diff, axis=2)

    rr = cc = None
    if prefer_scipy:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            rr, cc = linear_sum_assignment(cost)
        except Exception:
            rr, cc = _greedy_assign(cost)
    else:
        rr, cc = _greedy_assign(cost)

    # Optionally reject bad matches by distance
    if max_assign_dist is not None:
        keep = cost[rr, cc] <= max_assign_dist
        rr = rr[keep]
        cc = cc[keep]

    # Reorder
    pts_sel = pts[rr]
    ref_sel = grid_pts[cc]
    return pts_sel, ref_sel, rr, cc


# ========================= Rigid transform: camera -> ArUco =========================

def transform_to_aruco_frame(points_3d: Iterable[Iterable[float]],
                             rvec: np.ndarray,
                             tvec: np.ndarray,
                             debug: bool = False) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - points_3d: iterable of 3D points in the camera frame, shape (N, 3)
    - rvec: Rodrigues rotation vector from cv2.aruco.estimatePoseSingleMarkers (marker in camera)
    - tvec: translation vector from cv2.aruco.estimatePoseSingleMarkers (marker in camera)
    - debug: if True, print intermediate matrices/vectors

    RETURNS:
    - points expressed in the ArUco marker frame, shape (N, 3)

    Notes:
    The ArUco pose gives the marker expressed in the camera frame:  p_cam = R * p_marker + t.
    To express camera-frame points in the marker frame, invert the transform:
      p_marker = R^T * (p_cam - t).
    """
    pts = np.asarray(points_3d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`points_3d` must have shape (N, 3); got {pts.shape}")

    # R: marker -> camera
    R_marker, _ = cv2.Rodrigues(rvec)
    # Inverse: camera -> marker
    R_inv = R_marker.T
    t_inv = -R_inv @ tvec.reshape(3, 1)

    if debug:
        print("Rotation Matrix (marker→camera) R:\n", R_marker)
        print("Inverse Rotation Matrix R.T (camera→marker):\n", R_inv)
        print("Translation t (marker→camera):", tvec.ravel())
        print("Inverse translation -R.T @ t (camera→marker):", t_inv.ravel())

    # Vectorized transform: (N,3) -> (N,3)
    pts_marker = (R_inv @ pts.T + t_inv).T
    return pts_marker


# ========================= Simple 3D scatter with labels =========================

def plot_3d(points: Iterable[Iterable[float]],
            title: str = "3D Plot",
            annotate: bool = True) -> None:
    """RELEVANT FUNCTION INPUTS:
    - points: iterable of 3D points, shape (N, 3)
    - title: figure title
    - annotate: if True, draw point indices next to the markers
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        print("⚠️ No points to plot.")
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(f"⚠️ Skipping plot for {title}: unexpected shape {pts.shape}")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    if annotate:
        for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, f"{i+1}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


# ========================= Plane fit helpers =========================

def _fit_plane_lstsq(points_3d: np.ndarray) -> Tuple[float, float, float]:
    """Fit z = A x + B y + C (least squares) and return (A, B, C)."""
    A_mat = np.c_[points_3d[:, 0], points_3d[:, 1], np.ones(points_3d.shape[0])]
    b_vec = points_3d[:, 2]
    plane_params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    return tuple(plane_params)  # (A, B, C)


def _plane_frame_axes(pts: np.ndarray,
                      normal: np.ndarray,
                      x_hint: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal frame (x_axis, y_axis, normal) on the plane.
    - If x_hint is provided, it is projected onto the plane and used as x_axis.
    - y_axis = normal × x_axis, both normalized.
    """
    if x_hint is None:
        # Use first-to-last as a weak hint if available
        x_hint = (pts[-1] - pts[0]) if len(pts) > 1 else np.array([1.0, 0.0, 0.0])
    # Remove normal component
    x_axis = x_hint - np.dot(x_hint, normal) * normal
    if np.linalg.norm(x_axis) < 1e-12:
        # Fallback axis if x_hint was parallel to normal
        x_axis = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis, normal


# ========================= Rectangular grid visualizer (original kept) =========================

def plot_3d_with_distances(points: Iterable[Iterable[float]],
                           title: str = "3D Plot with Distances",
                           actual_plane: bool = True,
                           remap_onto_origin: bool = True,
                           grid_shape: Tuple[int, int] = (2, 4),
                           grid_size_m: Tuple[float, float] = (2.7, 8.543)) -> None:
    """RELEVANT FUNCTION INPUTS:
    - points: iterable of 3D points, shape (N, 3); expected order matches `grid_shape` row-major
    - title: figure title
    - actual_plane: if True, use `grid_size_m` as the physical (height, width); if False, infer from points
    - remap_onto_origin: if True, translate/rotate so the grid origin ≈ (0,0,0) and axes align with plane
    - grid_shape: (rows, cols) for the reference grid; default (2,4) expects 8 points
    - grid_size_m: (height_m, width_m) physical size of the ideal grid if `actual_plane=True`

    This function:
    1) fits a plane to the input points;
    2) builds an orthonormal frame on the plane;
    3) constructs an ideal rectangular grid on that plane;
    4) optionally remaps both sets so the grid sits at the origin with orthonormal axes;
    5) plots both sets, plus per-point error vectors and summaries.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        print("⚠️ No points to plot.")
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(f"⚠️ Skipping plot for {title}: unexpected shape {pts.shape}")
        return

    rows, cols = grid_shape
    N_expected = rows * cols
    if pts.shape[0] != N_expected:
        print(f"⚠️ Expected {N_expected} points for grid_shape={grid_shape}, got {pts.shape[0]}. Proceeding anyway.")

    # ---- 1) Fit plane: z = A x + B y + C  -> normal = (-A, -B, 1)
    A, B, C = _fit_plane_lstsq(pts)
    normal = np.array([-A, -B, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)

    # ---- 2) Plane frame
    origin = np.mean(pts, axis=0)
    x_hint = pts[min(3, len(pts)-1)] - pts[0] if len(pts) >= 4 else None
    x_axis, y_axis, n_axis = _plane_frame_axes(pts, normal, x_hint=x_hint)

    # ---- 3) Grid sizing
    if actual_plane:
        height_m, width_m = grid_size_m
    else:
        width_m = np.linalg.norm(pts[min(cols-1, len(pts)-1)] - pts[0]) if len(pts) >= cols else 1.0
        height_m = np.linalg.norm(pts[min(cols+0, len(pts)-1)] - pts[0]) if len(pts) >= cols+1 else 1.0

    # Generate ideal grid points on the plane, centered at origin
    x_offsets = np.linspace(-0.5 * width_m, 0.5 * width_m, cols)
    y_offsets = np.linspace(-0.5 * height_m, 0.5 * height_m, rows)
    grid_points = []
    for y in y_offsets[::-1]:
        for x in x_offsets:
            p_plane = origin + x * x_axis + y * y_axis
            grid_points.append(p_plane)
    grid_points = np.asarray(grid_points)

    # --- Assign measured points to nearest reference nodes (min-sum)
    pts_assigned, ref_assigned, idx_pts, idx_ref = match_points_to_grid(
        pts, grid_points, max_assign_dist=None, prefer_scipy=True
    )

    if pts_assigned.shape[0] == 0:
        print("⚠️ No valid point↔reference assignments found.")
        return

    # From here on, use the assigned/reordered arrays
    pts = pts_assigned
    grid_points = ref_assigned

    # ---- 4) Optional remap to origin-aligned plane frame
    if remap_onto_origin:
        R = np.column_stack((x_axis, y_axis, n_axis))  # columns are plane axes in world coords
        origin_plane = origin  # use plane mean as origin
        pts = (pts - origin_plane) @ R.T
        grid_points = (grid_points - origin_plane) @ R.T

    # ---- 5) Plot both sets and error vectors
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    # Try wireframe for rectangular grids
    try:
        gp = grid_points.reshape(rows, cols, 3)
        # Horizontal lines
        for r in range(rows):
            for c in range(cols - 1):
                ax1.plot(
                    [gp[r, c, 0], gp[r, c + 1, 0]],
                    [gp[r, c, 1], gp[r, c + 1, 1]],
                    [gp[r, c, 2], gp[r, c + 1, 2]],
                    color='r', linewidth=1, alpha=0.5
                )
        # Vertical lines
        for c in range(cols):
            for r in range(rows - 1):
                ax1.plot(
                    [gp[r, c, 0], gp[r + 1, c, 0]],
                    [gp[r, c, 1], gp[r + 1, c, 1]],
                    [gp[r, c, 2], gp[r + 1, c, 2]],
                    color='r', linewidth=1, alpha=0.5
                )
    except ValueError:
        pass

    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', label="Measured Points", s=20)
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c='r', label="Ideal Grid", s=15)

    ax1.set_title(title)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc="best")
    ax1.set_box_aspect([1, 1, 1])

    # Error vectors and metrics
    deltas = pts - grid_points
    n_remap = np.array([0.0, 0.0, 1.0]) if remap_onto_origin else n_axis
    out_of_plane = deltas @ n_remap
    in_plane = np.linalg.norm(deltas - np.outer(out_of_plane, n_remap), axis=1)
    euclidean = np.linalg.norm(deltas, axis=1)

    # Draw green error segments and annotate length
    for p_meas, p_ref, d_e in zip(pts, grid_points, euclidean):
        ax1.plot([p_meas[0], p_ref[0]], [p_meas[1], p_ref[1]], [p_meas[2], p_ref[2]], 'g-', linewidth=2)
        mid = 0.5 * (p_meas + p_ref)
        ax1.text(mid[0], mid[1], mid[2], f"{d_e:.3f} m", color='black', fontsize=8)

    # Print summary stats
    print(f"[{title}] Max Euclidean error: {euclidean.max():.4f} m | Mean: {euclidean.mean():.4f} m")
    print(f"[{title}] Max In-plane error: {in_plane.max():.4f} m | Mean: {in_plane.mean():.4f} m")
    print(f"[{title}] Max Out-of-plane error: {np.abs(out_of_plane).max():.4f} m | Mean: {np.abs(out_of_plane).mean():.4f} m")

    plt.tight_layout()
    plt.show()

    # Optional: second figure with pairwise distances for the (2,4) case
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax2.set_title(title + " — Pairwise Distances")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_box_aspect([1, 1, 1])

    if rows == 2 and cols == 4 and pts.shape[0] >= 8:
        y_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        x_pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]

        def _dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        for i, j in y_pairs:
            p1, p2 = pts[i], pts[j]
            d = _dist(p1, p2)
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)
            mid = 0.5 * (p1 + p2)
            ax2.text(mid[0], mid[1], mid[2], f"{d:.3f} m", color='black', fontsize=8)

        for i, j in x_pairs:
            p1, p2 = pts[i], pts[j]
            d = _dist(p1, p2)
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', linewidth=2)
            mid = 0.5 * (p1 + p2)
            ax2.text(mid[0], mid[1], mid[2], f"{d:.3f} m", color='black', fontsize=8)

        for i, (x, y, z) in enumerate(pts):
            ax2.text(x, y, z, f"{i}", fontsize=8)

    plt.tight_layout()
    plt.show()


# ========================= Trapezium generator + visualizer (NEW) =========================

def _build_trapezium_points_on_plane(origin: np.ndarray,
                                     x_axis: np.ndarray,
                                     y_axis: np.ndarray,
                                     dx: float = 2.4,
                                     dy: float = 2.2) -> np.ndarray:
    """Generate a 6-point trapezium on the plane defined by (origin, x_axis, y_axis).

    Layout (indices and local-plane coordinates):
      Bottom row (y = -dy/2): 4 points at x = [-1.5*dx, -0.5*dx, 0.5*dx,  1.5*dx]  -> indices 0..3
      Top row    (y = +dy/2): 2 points at x = [ -0.5*dx,  0.5*dx]                   -> indices 4..5
      The top points are vertically above the middle two bottom points.

    Returns:
      (6,3) array of 3D points on the plane.
    """
    y_bottom = -0.5 * dy
    y_top    = +0.5 * dy

    x_bottoms = np.array([-1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx], dtype=float)
    x_tops    = np.array([-0.5*dx,  0.5*dx], dtype=float)

    pts_local = []
    # Bottom 4
    for xb in x_bottoms:
        pts_local.append((xb, y_bottom))
    # Top 2 (above the middle two bottom nodes)
    for xt in x_tops:
        pts_local.append((xt, y_top))

    pts = []
    for x_loc, y_loc in pts_local:
        p = origin + x_loc * x_axis + y_loc * y_axis
        pts.append(p)
    return np.asarray(pts, dtype=float)


"""
Fit a 6-point trapezium to measured 3D points using least-squares (rigid 2D fit on the fitted plane).
No ArUco, no automatic matching. You give 6 points in the intended order.

Trapezium definition:
  - Bottom row: 4 nodes with spacing dx at u = [-1.5dx, -0.5dx, 0.5dx, 1.5dx], v = -dy/2
  - Top row   : 2 nodes above the middle two, u = [-0.5dx, 0.5dx], v = +dy/2
Order: [b0, b1, b2, b3, t0, t1]
"""

# ---------------- Plane utilities ----------------
# -*- coding: utf-8 -*-
"""
Least-squares fit of a fixed 6-point trapezium to 3D points (no matching, no scale).
Trapezium (u,v) coordinates are EXACTLY:
  bottom: (0,0), (2.4,0), (4.8,0), (7.2,0)
  top   : (2.4,2.2), (4.8,2.2)
Order is [b0, b1, b2, b3, t0, t1].

We:
  1) fit plane to measured 3D points
  2) project to (u,v,w)
  3) rigid 2D LS fit (rotation+translation, det=+1) of model→measured (no scale)
  4) map fitted model back to 3D with w=0
  5) report errors, plot, and draw the 4 diagonals: (0,4), (2,4), (1,5), (3,5)
"""

# ---------- plane utilities ----------

def _fit_plane_lstsq(P: np.ndarray) -> Tuple[float, float, float]:
    """Fit z = A x + B y + C to 3D points. Returns (A,B,C)."""
    A_mat = np.c_[P[:, 0], P[:, 1], np.ones(P.shape[0])]
    b_vec = P[:, 2]
    A, B, C = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    return float(A), float(B), float(C)

def _plane_axes(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a plane to P and return (origin, x_axis, y_axis, n_axis) with x/y in-plane and n the unit normal.
    """
    A, B, C = _fit_plane_lstsq(P)
    n = np.array([-A, -B, 1.0], float)
    n /= np.linalg.norm(n)

    origin = P.mean(axis=0)

    x_guess = (P[min(3, len(P)-1)] - P[0]) if len(P) >= 4 else np.array([1.0, 0.0, 0.0])
    x_axis = x_guess - (x_guess @ n) * n
    if np.linalg.norm(x_axis) < 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ n) * n
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(n, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return origin, x_axis, y_axis, n

def _to_plane_coords(P: np.ndarray, origin: np.ndarray,
                     x_axis: np.ndarray, y_axis: np.ndarray, n_axis: np.ndarray) -> np.ndarray:
    """World 3D -> (u,v,w) in plane frame."""
    D = P - origin
    return np.column_stack([D @ x_axis, D @ y_axis, D @ n_axis])

def _from_plane_uv(uv: np.ndarray, origin: np.ndarray,
                   x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """(u,v) on plane -> world 3D with w=0."""
    return origin + uv[:, 0:1]*x_axis + uv[:, 1:2]*y_axis


# ---------- rigid 2D fit (Kabsch/Procrustes; B ≈ A R + t, det R = +1) ----------

def _rigid_fit_2d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimize sum ||A R + t - B||^2 over rotation R (2x2, det=+1) and translation t (1x2).
    A,B: (K,2)
    Returns (R, t) with shapes (2,2) and (2,)
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    assert A.shape == B.shape and A.shape[1] == 2

    muA = A.mean(axis=0, keepdims=True)
    muB = B.mean(axis=0, keepdims=True)
    A0 = A - muA
    B0 = B - muB

    # SVD of covariance
    U, _, Vt = np.linalg.svd(A0.T @ B0)  # A->B
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = (muB - muA @ R).ravel()
    return R, t


# ---------- exact right-angled trapezium model ----------

def _trapezium_uv(dx: float, dy: float) -> np.ndarray:
    """
    Right-angled trapezium in (u,v):
      bottom: (0,0), (dx,0), (2dx,0), (3dx,0)
      top   : (dx,dy), (2dx,dy)
    Order: [b0,b1,b2,b3,t0,t1]
    """
    return np.array([
        [0.0*dx, 0.0],   # b0
        [1.0*dx, 0.0],   # b1
        [2.0*dx, 0.0],   # b2
        [3.0*dx, 0.0],   # b3
        [1.0*dx, dy],    # t0 (above b1)
        [2.0*dx, dy],    # t1 (above b2)
    ], dtype=float)


# ---------- global-optimal assignment via brute force (6! = 720) ----------

def _best_assignment_rigid_ls(model_uv: np.ndarray, meas_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Find the assignment (permutation) and 2D rigid transform that minimize RMS error.
    Returns (R(2x2), t(2,), perm(6,), rms)
    """
    assert model_uv.shape == (6, 2) and meas_uv.shape[1] == 2 and meas_uv.shape[0] >= 6

    # If more than 6 points were provided, preselect the 6 closest to their centroid
    if meas_uv.shape[0] > 6:
        mu = meas_uv.mean(axis=0)
        keep = np.argsort(np.linalg.norm(meas_uv - mu, axis=1))[:6]
        meas_uv = meas_uv[keep]

    best = (None, None, None, np.inf)  # R, t, perm, rms

    # Try all permutations of mapping model_idx -> meas_idx
    for perm in itertools.permutations(range(6), 6):
        B = meas_uv[list(perm)]          # target in the order of model
        R, t = _rigid_fit_2d(model_uv, B)
        res = model_uv @ R + t - B       # residuals in UV
        rms = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
        if rms < best[3]:
            best = (R, t, np.array(perm, dtype=int), rms)

    return best  # type: ignore


# ---------- public API ----------

def fit_trapezium_auto(points: Iterable[Iterable[float]],
                       dx: float = 2.4,
                       dy: float = 2.2,
                       remap_onto_origin: bool = True,
                       title: str = "Euclidean distance of triangulated points to corresponding node on fitted plane") -> None:
    """
    Automatically match & LS-fit the fixed trapezium to measured 3D points.
    - Model in (u,v): bottom (0,0),(dx,0),(2dx,0),(3dx,0); top (dx,dy),(2dx,dy)
    - Global optimum over all 6! assignments; rigid 2D fit (rotation+translation, no scale).
    - Visualization (when remap_onto_origin=True): bottom-right corner at origin (0,0,0),
      bottom edge aligned with +X axis, fitted plane at Z=0, top-row Y > 0.
    """
    P = np.asarray(points, float)
    if P.shape[0] < 6 or P.shape[1] != 3:
        raise ValueError(f"Expected at least 6 points of shape (N,3); got {P.shape}")

    # 1) plane + axes from ALL provided points
    origin, x_axis, y_axis, n_axis = _plane_axes(P)  # expects unit vectors

    # 2) project all to (u,v,w)
    uvw_all = _to_plane_coords(P, origin, x_axis, y_axis, n_axis)
    uv_all = uvw_all[:, :2]

    # If more than 6, select the 6 closest to centroid
    if uv_all.shape[0] > 6:
        mu = uv_all.mean(axis=0)
        keep_idx = np.argsort(np.linalg.norm(uv_all - mu, axis=1))[:6]
    else:
        keep_idx = np.arange(6)

    uv_meas6 = uv_all[keep_idx]
    P_meas6  = P[keep_idx]

    # 3) exact model
    uv_model = _trapezium_uv(dx, dy)  # order: [b0,b1,b2,b3,t0,t1] = [0..5]

    # 4) global best assignment + rigid LS (returns R, t, perm, rms_uv)
    R2, t2, perm, _ = _best_assignment_rigid_ls(uv_model, uv_meas6)

    # Put measured 6 in model order according to the optimal perm
    uv_meas_ordered = uv_meas6[perm]
    P_meas_ordered  = P_meas6[perm]

    # Fitted model in UV
    uv_fit = uv_model @ R2 + t2

    # 5) back to 3D with w=0 (fitted ideal nodes on the plane)
    ref3d = _from_plane_uv(uv_fit, origin, x_axis, y_axis)

    # 6) errors (point-to-node, decomposed wrt plane normal)
    deltas = P_meas_ordered - ref3d
    out_of_plane = deltas @ n_axis
    in_plane = np.linalg.norm(deltas - np.outer(out_of_plane, n_axis), axis=1)
    euclidean = np.linalg.norm(deltas, axis=1)

    # 7) visualization
    if remap_onto_origin:
        # --- New display frame:
        # origin at bottom-right corner, X along bottom edge (to the left),
        # Y chosen so top row has positive Y, Z normal to plane (Z=0 on fitted plane).
        idx_br = 3  # bottom-right in the model order [0,1,2,3,4,5]
        O_disp = ref3d[idx_br]

        # X along bottom edge from bottom-right -> bottom-left
        x_dir = ref3d[0] - ref3d[idx_br]
        x_dir = x_dir / np.linalg.norm(x_dir)

        # Y from right-hand rule using plane normal; flip so top row has Y>0
        y_dir = np.cross(n_axis, x_dir)
        y_dir = y_dir / np.linalg.norm(y_dir)

        # Check sign of Y using midpoints of bottom vs top rows
        bottom_mid = 0.5 * (ref3d[1] + ref3d[2])
        top_mid    = 0.5 * (ref3d[4] + ref3d[5])
        if (top_mid - bottom_mid) @ y_dir < 0:
            x_dir = -x_dir
            y_dir = -y_dir
        # n_axis stays as returned by _plane_axes (Z=0 for fitted plane)

        # Transform measured and reference to display frame
        def to_disp(X):
            Xc = X - O_disp
            Xx = Xc @ x_dir
            Xy = Xc @ y_dir
            Xz = Xc @ n_axis
            return np.column_stack([Xx, Xy, Xz])

        P_disp   = to_disp(P_meas_ordered)
        REF_disp = to_disp(ref3d)
        # Force perfect plane for reference nodes
        REF_disp[:, 2] = 0.0

        xlab, ylab, zlab = "X (m)", "Y (m)", "Z (m)"
    else:
        P_disp = P_meas_ordered.copy()
        REF_disp = ref3d.copy()
        xlab, ylab, zlab = "X (m)", "Y (m)", "Z (m)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Edges only (no diagonals)
    edges = [(0,1),(1,2),(2,3),(1,4),(2,5),(4,5)]
    for i,j in edges:
        pi, pj = REF_disp[i], REF_disp[j]
        ax.plot([pi[0], pj[0]],[pi[1], pj[1]],[pi[2], pj[2]],
                color='r', linewidth=1.8, alpha=0.95, linestyle='-')

    # Translucent fill of the outer trapezium
    outer_order = [0, 1, 4, 5, 2, 3]  # boundary loop
    outer_poly = [REF_disp[k] for k in outer_order]
    ax.add_collection3d(Poly3DCollection([outer_poly],
                        facecolors=(1.0, 0.0, 0.0, 0.12), edgecolors='none'))

    # Fill the central rectangle (b1–t0–t1–b2)
    rect_order = [1, 4, 5, 2]
    rect_poly = [REF_disp[k] for k in rect_order]
    ax.add_collection3d(Poly3DCollection([rect_poly],
                        facecolors=(1.0, 0.0, 0.0, 0.18), edgecolors='none'))

    # Optional: keep inner triangle fills (without drawing diagonals)
    inner_tris = [
        (0, 1, 4),   # upper-left
        (0, 4, 2),   # lower-left
        (1, 2, 5),   # lower-right
        (2, 3, 5),   # upper-right
    ]
    for i,j,k in inner_tris:
        tri = [REF_disp[i], REF_disp[j], REF_disp[k]]
        ax.add_collection3d(Poly3DCollection([tri],
                            facecolors=(1.0, 0.0, 0.0, 0.10), edgecolors='none'))

    # Points + error segments
    ax.scatter(P_disp[:,0], P_disp[:,1], P_disp[:,2], c='b', s=34, label="Measured")
    ax.scatter(REF_disp[:,0], REF_disp[:,1], REF_disp[:,2], c='r', s=28, label="Ideal (fitted)")
    for pm, pr, d_e in zip(P_disp, REF_disp, euclidean):
        ax.plot([pm[0], pr[0]], [pm[1], pr[1]], [pm[2], pr[2]], 'g-', linewidth=2.2)
        mid = 0.5*(pm + pr)
        ax.text(mid[0], mid[1], mid[2], f"{d_e*100:.1f} cm", fontsize=8, color='black')

    ax.set_title(title)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.invert_xaxis()  # equivalent to flipping limits
    ax.set_zlabel("Z (cm)")
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, pos: f"{z * 100:.1f}"))
    ax.legend(loc="best")
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    # 8) summary: MAX and MEAN only
    print(f"[{title}]")
    print(f"  Max total error     : {euclidean.max():.4f} m | Mean total     : {euclidean.mean():.4f} m")
    print(f"  Max in-plane error  : {in_plane.max():.4f} m | Mean in-plane  : {in_plane.mean():.4f} m")
    print(f"  Max out-of-plane err: {np.abs(out_of_plane).max():.4f} m | Mean out-of-plane: {np.abs(out_of_plane).mean():.4f} m")

# ========================= (Optional) keep your simple 3D scatter for quick checks =========================

def plot_3d(points: Iterable[Iterable[float]],
            title: str = "3D Plot",
            annotate: bool = True) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0 or pts.ndim != 2 or pts.shape[1] != 3:
        print("⚠️ Skipping plot: unexpected shape.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    if annotate:
        for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, f"{i+1}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

# ========================= Example usage =========================

if __name__ == "__main__":
    # Minimal smoke test with fake data (comment out in production):
    # # Fake nearly-planar trapezium-ish measurements (add small noise)
    # dx, dy = 2.4, 2.2
    # true = np.array([
    #     [-1.5*dx, -0.5*dy, 0.00],
    #     [-0.5*dx, -0.5*dy, 0.00],
    #     [ 0.5*dx, -0.5*dy, 0.02],
    #     [ 1.5*dx, -0.5*dy, 0.00],
    #     [-0.5*dx,  0.5*dy, 0.01],
    #     [ 0.5*dx,  0.5*dy,-0.01],
    # ], dtype=float)
    # noise = 0.005 * np.random.randn(*true.shape)
    # pts_world = true + noise
    # plot_3d_with_trapezium(pts_world, "Errors vs trapezium (4+2)", remap_onto_origin=True, dx=dx, dy=dy)

    # # Keep your original rectangular visualizer available:
    # # plot_3d_with_distances(pts_world, "Errors vs ideal (2x4)", actual_plane=True,
    # #                        remap_onto_origin=True, grid_shape=(2,4), grid_size_m=(2.7, 8.543))
    pass
