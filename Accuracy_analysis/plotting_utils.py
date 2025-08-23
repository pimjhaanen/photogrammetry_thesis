import numpy as np
import cv2
import matplotlib.pyplot as plt

def transform_to_aruco_frame(points_3d, rvec, tvec, debug=False):
    R_marker, _ = cv2.Rodrigues(rvec)
    R_inv = R_marker.T
    t_inv = -R_inv @ tvec.reshape(3, 1)

    if debug:
        print("Rotation Matrix (from ArUco):\n", R_marker)
        print("Inverse Rotation Matrix (R.T):\n", R_inv)
        print("Translation Vector (from ArUco):", tvec)
        print("Inverse Translation Vector (-R.T @ t):", t_inv.flatten())

    points_aruco = [ (R_inv @ pt.reshape(3,1) + t_inv).flatten() for pt in points_3d ]
    return np.array(points_aruco)

def plot_3d(points, title="3D Plot", annotate=True):
    if points is None or len(points) == 0:
        print("⚠️ No points to plot.")
        return
    pts = np.array(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(f"⚠️ Skipping plot for {title}: unexpected shape {pts.shape}")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    if annotate:
        for i, pt in enumerate(pts):
            ax.text(pt[0], pt[1], pt[2], f"{i+1}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

def plot_3d_with_distances(points, title="3D Plot with Distances", actual_plane=True, remap_onto_origin=True):
    if points is None or len(points) == 0:
        print("⚠️ No points to plot.")
        return
    pts = np.array(points)

    # Fit the plane to the points
    def fit_plane(points_3d):
        points_3d = np.array(points_3d)
        A = np.c_[points_3d[:, 0], points_3d[:, 1], np.ones(points_3d.shape[0])]
        b = points_3d[:, 2]
        plane_params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return plane_params

    plane_params = fit_plane(pts)
    A, B, C = plane_params
    normal = np.array([-A, -B, 1])
    normal /= np.linalg.norm(normal)

    # Compute plane coordinate system
    origin = np.mean(pts, axis=0)
    x_axis = pts[3] - pts[0]
    x_axis -= np.dot(x_axis, normal) * normal  # remove z component
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    # Actual physical dimensions
    width = 4.26 if actual_plane else np.linalg.norm(pts[3] - pts[0])
    height = 1.0 if actual_plane else np.linalg.norm(pts[0] - pts[4])

    # Create the ideal grid on the plane
    x_offsets = np.linspace(-0.5 * width, 0.5 * width, 4)
    y_offsets = np.linspace(-0.5 * height, 0.5 * height, 2)
    grid_points = []
    for y in reversed(y_offsets):
        for x in x_offsets:
            point_on_plane = origin + x * x_axis + y * y_axis
            grid_points.append(point_on_plane)
    grid_points = np.array(grid_points)

    # Remap to origin if enabled
    if remap_onto_origin:
        p0 = grid_points[0]
        target_origin = np.array([0, 0, 0])
        rotation_matrix = np.column_stack((x_axis, y_axis, normal))
        inv_rotation = np.linalg.inv(rotation_matrix)
        pts = (pts - p0) @ inv_rotation.T
        grid_points = (grid_points - p0) @ inv_rotation.T

    # Plotting
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    # Plot the adjusted plane surface
    plane_grid = np.array(grid_points).reshape(2, 4, 3)  # Reshape to 2 rows, 4 cols
    for row in plane_grid:
        for i in range(len(row) - 1):
            ax1.plot(
                [row[i][0], row[i + 1][0]],
                [row[i][1], row[i + 1][1]],
                [row[i][2], row[i + 1][2]],
                color='r', linewidth=1, alpha=0.5
            )
    for col in range(4):
        for i in range(1):
            ax1.plot(
                [plane_grid[i][col][0], plane_grid[i + 1][col][0]],
                [plane_grid[i][col][1], plane_grid[i + 1][col][1]],
                [plane_grid[i][col][2], plane_grid[i + 1][col][2]],
                color='r', linewidth=1, alpha=0.5
            )

    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', label="Points")
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c='r', label="Grid Points")
    ax1.set_title(title)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")

    # Error vectors and error metrics
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    for i, (p1, p2) in enumerate(zip(pts, grid_points)):
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', linewidth=2)
        mid_point = (p1 + p2) / 2
        dist = np.linalg.norm(p1 - p2)
        ax1.text(mid_point[0], mid_point[1], mid_point[2], f"{dist:.3f} m", color='black')

        delta = p1 - p2
        out_of_plane = np.dot(delta, normal)
        in_plane = np.linalg.norm(delta - out_of_plane * normal)

        print(
            f"Point {i}: Euclidean distance = {dist:.4f} m, In-plane error = {in_plane:.4f} m, Out-of-plane error = {out_of_plane:.4f} m")
    # Initialize lists
    euclidean_errors = []
    in_plane_errors = []
    out_of_plane_errors = []

    for p1, p2 in zip(pts, grid_points):
        delta = p1 - p2
        out_of_plane = np.dot(delta, normal)
        in_plane = np.linalg.norm(delta - out_of_plane * normal)
        euclidean = np.linalg.norm(delta)

        euclidean_errors.append(euclidean)
        in_plane_errors.append(in_plane)
        out_of_plane_errors.append(abs(out_of_plane))  # absolute out-of-plane error

    # Compute and print stats
    print(f"Max Euclidean distance error: {max(euclidean_errors):.4f} m")
    print(f"Avg Euclidean distance error: {sum(euclidean_errors) / len(euclidean_errors):.4f} m")

    print(f"Max In-plane error: {max(in_plane_errors):.4f} m")
    print(f"Avg In-plane error: {sum(in_plane_errors) / len(in_plane_errors):.4f} m")

    print(f"Max Out-of-plane error: {max(out_of_plane_errors):.4f} m")
    print(f"Avg Out-of-plane error: {sum(out_of_plane_errors) / len(out_of_plane_errors):.4f} m")

    plt.show()

    # Plot the distances between points in the grid
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax2.set_title(title)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    y_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    x_pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]

    for pair in y_pairs:
        p1, p2 = pts[pair[0]], pts[pair[1]]
        distance = euclidean_distance(p1, p2)
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)
        mid = (p1 + p2) / 2
        ax2.text(mid[0], mid[1], mid[2], f"{distance:.3f} m", color='black')

    for pair in x_pairs:
        p1, p2 = pts[pair[0]], pts[pair[1]]
        distance = euclidean_distance(p1, p2)
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'g-', linewidth=2)
        mid = (p1 + p2) / 2
        ax2.text(mid[0], mid[1], mid[2], f"{distance:.3f} m", color='black')

    for i, p in enumerate(pts):
        ax2.text(p[0], p[1], p[2], f"{i}")

    ax2.set_zlim(0, 6)
    plt.show()