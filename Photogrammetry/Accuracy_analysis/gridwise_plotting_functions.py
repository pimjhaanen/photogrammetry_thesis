"""This code can be used to (1) transform 3D points from a camera frame to an ArUco marker frame
and (2) visualize 3D point sets, including fitting a plane, building an ideal reference grid on
that plane, and plotting per-point distance errors (total, in-plane, and out-of-plane)."""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Optional


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


# ========================= Plane fit + error visualization =========================

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


def plot_3d_with_distances(points: Iterable[Iterable[float]],
                           title: str = "3D Plot with Distances",
                           actual_plane: bool = True,
                           remap_onto_origin: bool = True,
                           grid_shape: Tuple[int, int] = (2, 4),
                           grid_size_m: Tuple[float, float] = (1.0, 4.26)) -> None:
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
    3) constructs an ideal grid on that plane (same shape as points);
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
    # use a diagonal hint to stabilize x-axis direction
    x_hint = pts[min(3, len(pts)-1)] - pts[0] if len(pts) >= 4 else None
    x_axis, y_axis, n_axis = _plane_frame_axes(pts, normal, x_hint=x_hint)

    # ---- 3) Grid sizing
    if actual_plane:
        height_m, width_m = grid_size_m
    else:
        # infer approximate spans from corner pairs if present
        width_m = np.linalg.norm(pts[min(cols-1, len(pts)-1)] - pts[0]) if len(pts) >= cols else 1.0
        height_m = np.linalg.norm(pts[min(cols+0, len(pts)-1)] - pts[0]) if len(pts) >= cols+1 else 1.0

    # Generate ideal grid points on the plane, centered at origin
    x_offsets = np.linspace(-0.5 * width_m, 0.5 * width_m, cols)
    y_offsets = np.linspace(-0.5 * height_m, 0.5 * height_m, rows)
    grid_points = []
    # Reverse y for a top-to-bottom visual ordering similar to images
    for y in y_offsets[::-1]:
        for x in x_offsets:
            p_plane = origin + x * x_axis + y * y_axis
            grid_points.append(p_plane)
    grid_points = np.asarray(grid_points)

    # ---- 4) Optional remap to origin-aligned plane frame
    if remap_onto_origin:
        # Build rotation from (x_axis, y_axis, n_axis)
        R = np.column_stack((x_axis, y_axis, n_axis))  # world->plane axes
        R_inv = R.T
        p0 = grid_points[0]  # align first grid point to (0,0,0) for a stable origin
        pts = (pts - p0) @ R_inv.T
        grid_points = (grid_points - p0) @ R_inv.T

    # ---- 5) Plot both sets and error vectors
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot grid wireframe (rows, cols)
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
        # If reshape fails due to unexpected N, skip the wireframe
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
    # Recompute normal in the (possibly) remapped coordinates:
    # in remapped coordinates, the plane normal ≈ [0,0,1]
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

    # Optional: second figure with only pairwise distances along grid rows/cols (kept from your version)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax2.set_title(title + " — Pairwise Distances")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_box_aspect([1, 1, 1])

    # If shape matches, link neighbors (y_pairs/x_pairs derived for (2,4)); otherwise skip.
    if rows == 2 and cols == 4 and pts.shape[0] >= 8:
        # Vertical neighbors (between the two rows)
        y_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        # Horizontal neighbors within each row
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

        # Label indices
        for i, (x, y, z) in enumerate(pts):
            ax2.text(x, y, z, f"{i}", fontsize=8)

    plt.tight_layout()
    plt.show()


# ========================= Example usage =========================

if __name__ == "__main__":
    # Minimal smoke test with fake data (comment out in production):
    # pts_cam = np.array([[0,0,1],[1,0,1],[2,0,1],[3,0,1],[0,1,1],[1,1,1],[2,1,1],[3,1,1]], dtype=float)
    # rvec = np.array([[0.0],[0.0],[0.0]])  # identity rotation
    # tvec = np.array([[0.0],[0.0],[0.0]])  # zero translation
    # pts_marker = transform_to_aruco_frame(pts_cam, rvec, tvec, debug=True)
    # plot_3d(pts_cam, "Camera frame")
    # plot_3d_with_distances(pts_cam, "Errors vs ideal (2x4)", actual_plane=True,
    #                        remap_onto_origin=True, grid_shape=(2,4), grid_size_m=(1.0, 4.26))
    pass
