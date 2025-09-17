"""This code can be used to run a stereo photogrammetry pipeline on a single image pair.
It loads a stereo calibration, rectifies both views, detects blue circular blobs and red crosses,
sorts markers grid-wise, triangulates corresponding points to 3D, optionally transforms the points
to an ArUco marker frame, and visualizes/plots results."""

import cv2
import numpy as np
import pickle
from typing import List, Tuple, Optional

from Photogrammetry.marker_detection.marker_detection_utils import (
    detect_circles_blob,
    detect_aruco_pose,
    detect_crosses,
    get_subpixel_centers_gaussian,
)
from Photogrammetry.Accuracy_analysis.subpixel_accuracy_test import show_frame, zoom_in_on_circle
from Photogrammetry.stereo_photogrammetry_utils import triangulate_points, sort_markers_gridwise
from Photogrammetry.Accuracy_analysis.gridwise_plotting_functions import plot_3d_with_distances, transform_to_aruco_frame


# ============================ Small utilities ============================

def to_cv2_pts(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """RELEVANT FUNCTION INPUTS:
    - points: iterable of 2D points (x, y) or (x, y, r). Casts to (float(x), float(y)) tuples
              suitable for OpenCV triangulation and drawing functions.
    """
    return [(float(p[0]), float(p[1])) for p in points]


def filter_corresponding_pairs(
    left_pts: List[Tuple[float, float]],
    right_pts: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """RELEVANT FUNCTION INPUTS:
    - left_pts: sorted list of 2D points in the left image
    - right_pts: sorted list of 2D points in the right image

    Truncates both lists to the same length (min of both) to ensure 1:1 correspondence.
    """
    n = min(len(left_pts), len(right_pts))
    return left_pts[:n], right_pts[:n]


def show_frame_with_identified_markers(
    image: np.ndarray,
    blue_pts: List[Tuple[float, float]],
    cross_pts: List[Tuple[float, float]],
    window_title: str = "Image with ArUco, Circles, Crosses",
    scale: float = 0.5
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - image: BGR image to draw on
    - blue_pts: list of (x,y) blue circle centers to annotate
    - cross_pts: list of (x,y) cross centers to annotate
    - window_title: display window name
    - scale: resizing factor for display (e.g., 0.5 shows at 50%)
    """
    img_drawn = show_frame(image, blue_pts, cross_pts)  # uses your existing util
    disp = cv2.resize(img_drawn, None, fx=scale, fy=scale)
    cv2.imshow(window_title, disp)


def process_side(
    frame: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    refine_circles: str = "gaussian"
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray]
]:
    """RELEVANT FUNCTION INPUTS:
    - frame: rectified BGR image for this side (left or right)
    - mtx, dist: intrinsic matrix and distortion coefficients for this camera
    - refine_circles: subpixel method for blue circles ("gaussian" or "corner")

    RETURNS:
    - blue_centers: sub-pixel circle centers [(x,y), ...]
    - cross_centers: cross centers [(x,y), ...]
    - aruco_corners: corners returned by detect_aruco_pose (or None)
    - rvec: rotation vector of detected ArUco(s) (or None)
    - tvec: translation vector of detected ArUco(s) (or None)
    """
    # Blue circles
    blue_raw, _ = detect_circles_blob(frame, "blue")
    if refine_circles == "gaussian":
        blue_centers = get_subpixel_centers_gaussian(frame, blue_raw)
    else:
        # Fallback to corner sub-pixel if you later plug it in
        blue_centers = [(br[0], br[1]) for br in blue_raw]

    # Red crosses
    crosses, _ = detect_crosses(frame)

    # Sort both sets grid-wise (your utility)
    blue_sorted = sort_markers_gridwise(blue_centers)
    cross_sorted = sort_markers_gridwise(crosses)

    # Detect ArUco (optional)
    corners, rvec, tvec = detect_aruco_pose(frame, mtx, dist)

    return blue_sorted, cross_sorted, corners, rvec, tvec


def rectify_stereo_pair(
    left: np.ndarray, right: np.ndarray,
    mtx1: np.ndarray, dist1: np.ndarray,
    mtx2: np.ndarray, dist2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
    alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - left, right: raw BGR images
    - mtx1, dist1, mtx2, dist2: intrinsics + distortion for both cameras
    - R, T: stereo rotation and translation between camera 1 and camera 2
    - alpha: free scaling param for stereoRectify (0=crop to valid, 1=keep all)

    RETURNS:
    - left_rect, right_rect: undistorted & rectified images
    - P1, P2: projection matrices for left/right
    - R1, R2: rectification rotations for left/right
    - Q: disparity-to-depth re-projection matrix
    """
    h, w = left.shape[:2]
    image_size = (w, h)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T,
        flags=cv2.CALIB_FIX_INTRINSIC, alpha=alpha
    )

    map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, image_size, cv2.CV_32FC1)

    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect, P1, P2, R1, R2, Q


def triangulate_matched(
    left_pts: List[Tuple[float, float]],
    right_pts: List[Tuple[float, float]],
    P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - left_pts, right_pts: corresponding 2D points (equal length) in rectified pixel coords
    - P1, P2: 3x4 projection matrices from stereoRectify

    RETURNS:
    - (N, 3) array of 3D points in the rectified camera-1 coordinate frame.
    """
    l, r = filter_corresponding_pairs(left_pts, right_pts)
    if not l or not r:
        return np.empty((0, 3), dtype=float)

    # Call your triangulator (which returns a list of np.array OR (np.array, label))
    raw = triangulate_points(to_cv2_pts(l), to_cv2_pts(r), P1, P2)

    # Normalize to (N,3) ndarray
    pts_3d = []
    for item in raw:
        if isinstance(item, tuple):
            # (point3d, label)
            pts_3d.append(np.asarray(item[0], dtype=float).reshape(3))
        else:
            # plain point3d
            pts_3d.append(np.asarray(item, dtype=float).reshape(3))
    if len(pts_3d) == 0:
        return np.empty((0, 3), dtype=float)
    return np.vstack(pts_3d)



def transform_to_aruco_if_possible(
    pts_3d: np.ndarray,
    rvec: Optional[np.ndarray],
    tvec: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """RELEVANT FUNCTION INPUTS:
    - pts_3d: (N,3) points in camera-1 frame
    - rvec, tvec: ArUco pose vectors for transforming to marker frame (from camera-1 view)

    RETURNS:
    - (N,3) points in ArUco frame if pose is available, otherwise None.
    """
    if rvec is None or tvec is None:
        return None
    return transform_to_aruco_frame(pts_3d, rvec, tvec)


def zoom_debug_windows(
    left_src: np.ndarray,
    right_src: np.ndarray,
    left_circles: List[Tuple[float, float]],
    right_circles: List[Tuple[float, float]],
    scale: int = 10
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - left_src, right_src: original (pre-annotation) images to crop from
    - left_circles, right_circles: list of (x,y) circle centers (no radius needed here)
    - scale: integer scaling factor for the zoomed-in crops
    """
    for i, c in enumerate(left_circles):
        zoomed = zoom_in_on_circle(left_src, c)
        zoomed = cv2.resize(zoomed, (zoomed.shape[1] * scale, zoomed.shape[0] * scale))
        cv2.imshow(f"Zoomed-In Circle LEFT {i+1}", zoomed)

    for i, c in enumerate(right_circles):
        zoomed = zoom_in_on_circle(right_src, c)
        zoomed = cv2.resize(zoomed, (zoomed.shape[1] * scale, zoomed.shape[0] * scale))
        cv2.imshow(f"Zoomed-In Circle RIGHT {i+1}", zoomed)


# ============================ Main runner ============================

def run_stereo_photogrammetry(
    calib_file: str,
    left_path: str,
    right_path: str,
    visualize: bool = True,
    show_zoom: bool = True
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - calib_file: path to stereo calibration pickle with keys:
                  camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T
    - left_path, right_path: file paths to the left/right images
    - visualize: if True, shows annotated frames and plots distances
    - show_zoom: if True, opens zoom windows for the detected circles

    Runs the full pipeline: load calib -> rectify -> detect -> sort -> triangulate -> transform -> visualize.
    """
    # Load stereo calibration
    with open(calib_file, "rb") as f:
        calib = pickle.load(f)
    mtx1, dist1 = calib["camera_matrix_1"], calib["dist_coeffs_1"]
    mtx2, dist2 = calib["camera_matrix_2"], calib["dist_coeffs_2"]
    R, T = calib["R"], calib["T"]
    print(f"[CALIB] R:\n{R}\nT:\n{T}")

    # Read images
    left_raw = cv2.imread(left_path)
    right_raw = cv2.imread(right_path)
    if left_raw is None or right_raw is None:
        raise FileNotFoundError("Could not read one or both input images.")

    # Rectify
    left, right, P1, P2, R1, R2, Q = rectify_stereo_pair(left_raw, right_raw, mtx1, dist1, mtx2, dist2, R, T, alpha=0)

    # Process each side
    blue_L, cross_L, corners_L, rvec_L, tvec_L = process_side(left, mtx1, dist1, refine_circles="gaussian")
    blue_R, cross_R, corners_R, rvec_R, tvec_R = process_side(right, mtx2, dist2, refine_circles="gaussian")

    # Ensure correspondence and triangulate
    blue_L, blue_R = filter_corresponding_pairs(blue_L, blue_R)
    cross_L, cross_R = filter_corresponding_pairs(cross_L, cross_R)

    cross_3d = triangulate_matched(cross_L, cross_R, P1, P2) if len(cross_L) > 0 else np.empty((0, 3))
    cross_3d_aruco = transform_to_aruco_if_possible(cross_3d, rvec_L, tvec_L)

    if visualize:
        # Draw axes if ArUco detected
        if corners_L is not None and rvec_L is not None and tvec_L is not None:
            cv2.aruco.drawAxis(left, mtx1, dist1, rvec_L, tvec_L, 0.5)
        if corners_R is not None and rvec_R is not None and tvec_R is not None:
            cv2.aruco.drawAxis(right, mtx2, dist2, rvec_R, tvec_R, 0.5)

        # Show annotated frames (single function, no duplication)
        show_frame_with_identified_markers(left, blue_L, cross_L, window_title="LEFT: ArUco + Circles + Crosses", scale=0.5)
        show_frame_with_identified_markers(right, blue_R, cross_R, window_title="RIGHT: ArUco + Circles + Crosses", scale=0.5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optional zoom windows for circles
        if show_zoom and len(blue_L) > 0 and len(blue_R) > 0:
            zoom_debug_windows(left_raw, right_raw, blue_L, blue_R, scale=10)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Plot distances (example: crosses in camera frame)
        if cross_3d is not None and cross_3d.size > 0:
            plot_3d_with_distances(
                cross_3d,
                "Euclidean distance of markers to their respective points on fitted plane ($\\epsilon$)"
            )

        # ArUco-frame plot (only if we actually transformed)
        if cross_3d_aruco is not None and cross_3d_aruco.size > 0:
            plot_3d_with_distances(
                cross_3d_aruco,
                "Crosses in ArUco frame with distance lines"
            )


if __name__ == "__main__":
    # Example paths (same as your current script)
    calib_file = "../Calibration/stereoscopic_calibration/stereo_calibration_output/stereo_calibration_wide_84cm_filtered.pkl"
    left_path = "video_input/left camera/frame_0003_wide_4m.jpg"
    right_path = "video_input/right camera/frame_0003_wide_4m.jpg"

    run_stereo_photogrammetry(
        calib_file=calib_file,
        left_path=left_path,
        right_path=right_path,
        visualize=True,
        show_zoom=True
    )
