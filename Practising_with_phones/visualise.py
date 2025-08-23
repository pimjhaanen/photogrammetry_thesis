import json
import numpy as np
import open3d as o3d

# === Load triangulated markers ===
with open("triangulated_markers_frame0.json", "r") as f:
    data = json.load(f)

# Convert to point cloud array
points = np.array(list(data.values()), dtype=np.float64)
ids = list(data.keys())

print("Loaded points:\n", points)
print("Shape:", points.shape)

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = np.tile(np.array([[0.2, 0.4, 1.0]]), (len(points), 1))
pcd.colors = o3d.utility.Vector3dVector(colors)

# Coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

# === Add grid lines (XZ plane) ===
grid_size = 100
grid_spacing = 10
lines = []
points_grid = []

for x in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
    points_grid.append([x, 0, -grid_size])
    points_grid.append([x, 0, grid_size])
    lines.append([len(points_grid)-2, len(points_grid)-1])

for z in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
    points_grid.append([-grid_size, 0, z])
    points_grid.append([grid_size, 0, z])
    lines.append([len(points_grid)-2, len(points_grid)-1])

grid = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points_grid),
    lines=o3d.utility.Vector2iVector(lines)
)
grid.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for _ in lines])

# === Optional: Add text labels as tiny LineSets (workaround)
label_lines = []
label_points = []
label_colors = []
offset = np.array([0.5, 0.5, 0.5])

for i, point in enumerate(points):
    text = str(ids[i])
    anchor = point + offset
    label_points.append(point)
    label_points.append(anchor)
    label_lines.append([len(label_points)-2, len(label_points)-1])
    label_colors.append([1, 0, 0])

labels = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(label_points),
    lines=o3d.utility.Vector2iVector(label_lines)
)
labels.colors = o3d.utility.Vector3dVector(label_colors)

# === Visualise all ===
o3d.visualization.draw_geometries([pcd, frame, grid, labels])
