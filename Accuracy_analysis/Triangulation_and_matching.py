import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def triangulate_points(pts1, pts2, P1, P2):
    """
    Triangulate matched 2D points (pts1 and pts2) into 3D points.

    pts1: list of (x, y, label) or (x, y)
    pts2: list of (x, y)
    """
    points_3d = []
    for pl, pr in zip(pts1, pts2):
        x1, y1 = pl[:2]
        x2, y2 = pr[:2]
        label = pl[2] if len(pl) == 3 else None

        pl_h = np.array([[x1], [y1]], dtype=np.float32)
        pr_h = np.array([[x2], [y2]], dtype=np.float32)
        point_4d = cv2.triangulatePoints(P1, P2, pl_h, pr_h)
        point_3d = point_4d[:3] / point_4d[3]

        if label is not None:
            points_3d.append((point_3d.flatten(), label))
        else:
            points_3d.append(point_3d.flatten())
    return points_3d


def sort_markers_gridwise(points, n_rows=2):
    points = sorted(points, key=lambda pt: pt[1])
    row_height_split = np.median([pt[1] for pt in points])
    bottom = sorted([pt for pt in points if pt[1] > row_height_split], key=lambda pt: pt[0])
    top = sorted([pt for pt in points if pt[1] <= row_height_split], key=lambda pt: pt[0])
    return bottom + top

def match_stereo_points(left_points, right_points, max_vertical_disparity, max_total_distance):
    """
    Match left and right 2D points using epipolar (horizontal) constraints and the Hungarian algorithm.

    Args:
        left_points: List of (x, y) tuples from the left image.
        right_points: List of (x, y) tuples from the right image.
        max_vertical_disparity: Max vertical (y) difference allowed between a left-right pair.
        max_total_distance: Max total Euclidean distance allowed for a match to be valid.

    Returns:
        matched_left: List of matched points from the left image.
        matched_right: List of corresponding points from the right image.
    """
    if not left_points or not right_points:
        return [], []

    left_arr = np.array([pt[:2] for pt in left_points])
    labels = [pt[2] for pt in left_points]

    right_arr = np.array(right_points)
    #print("Left coordinates:", left_points)
    #print("Right coordinates:", right_points)
    cost_matrix = np.full((len(left_arr), len(right_arr)), fill_value=1e6)

    for i, lpt in enumerate(left_arr):
        for j, rpt in enumerate(right_arr):
            vertical_disparity = abs(lpt[1] - rpt[1])
            horizontal_disparity = lpt[0] - rpt[0]  # Ensure right points are to the left of the left camera points
            if vertical_disparity <= max_vertical_disparity and horizontal_disparity >= 100:
                dist = np.linalg.norm(lpt - rpt)
                cost_matrix[i, j] = dist
    #print("Cost matrix:\n", cost_matrix)

    left_indices, right_indices = linear_sum_assignment(cost_matrix)

    matched_left = []
    matched_right = []
    for li, ri in zip(left_indices, right_indices):
        if cost_matrix[li, ri] < max_total_distance:
            matched_left.append((*left_arr[li], labels[li]))  # keep label
            matched_right.append(tuple(right_arr[ri]))  # right has no label

    return matched_left, matched_right

