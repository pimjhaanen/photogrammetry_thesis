import open3d as o3d
import numpy as np
import os
import cv2

# === CONFIGURATION ===
ply_folder = "C:/Users/pimha/PycharmProjects/photogrammetry_thesis/ply_frames"
output_dir = "render_ply_frames_with_labels"
output_video = "ply_point_cloud_animation_with_labels.mp4"
fps = 10
image_size = (800, 600)

os.makedirs(output_dir, exist_ok=True)
ply_files = sorted(f for f in os.listdir(ply_folder) if f.endswith(".ply"))

# === SETUP VISUALIZER ===
vis = o3d.visualization.Visualizer()
vis.create_window(width=image_size[0], height=image_size[1], visible=False)
geom_added = False

# Grid + Axes
grid_size = 100
spacing = 10
lines, points = [], []

for x in np.arange(-grid_size, grid_size + spacing, spacing):
    points.append([x, 0, -grid_size])
    points.append([x, 0, grid_size])
    lines.append([len(points)-2, len(points)-1])

for z in np.arange(-grid_size, grid_size + spacing, spacing):
    points.append([-grid_size, 0, z])
    points.append([grid_size, 0, z])
    lines.append([len(points)-2, len(points)-1])

grid = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
grid.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]] * len(lines))
vis.add_geometry(grid)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0))

# === RENDER LOOP ===
prev_pcd = None
for i, fname in enumerate(ply_files):
    pcd = o3d.io.read_point_cloud(os.path.join(ply_folder, fname))
    pcd.paint_uniform_color([0.2, 0.5, 1.0])

    if not geom_added:
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_lookat(pcd.get_center())
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        geom_added = True
    else:
        vis.remove_geometry(prev_pcd)
        vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    img = (np.asarray(img) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Label projection
    points = np.asarray(pcd.points)
    intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
    extrinsic = ctr.convert_to_pinhole_camera_parameters().extrinsic
    points_h = np.hstack((points, np.ones((len(points), 1))))
    proj = intrinsic @ (extrinsic[:3] @ points_h.T)
    proj[:2] /= proj[2]
    proj = proj[:2].T.astype(int)

    for pt, coord in zip(proj, points):
        x, y = pt
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            label = f"[{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}]"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            text_w, text_h = text_size
            cv2.putText(img, label,
                        (int(x - text_w // 2), int(y + text_h // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    #cv2.imwrite(f"{output_dir}/frame_{i:04d}.png", img)
    #prev_pcd = pcd

vis.destroy_window()

# === CREATE VIDEO ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_video, fourcc, fps, image_size)
for fname in sorted(os.listdir(output_dir)):
    if fname.endswith(".png"):
        img = cv2.imread(os.path.join(output_dir, fname))
        video_out.write(img)
video_out.release()
print(f"Video saved to: {output_video}")
