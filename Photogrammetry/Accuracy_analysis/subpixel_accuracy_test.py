"""This code can be used to visualize detected markers in stereo photogrammetry.
It provides utilities to:
(1) overlay detected circles and crosses on an image, with indices,
(2) zoom into a circle region with an adjustable margin for inspection.
"""

import cv2
import numpy as np
from typing import Iterable, Tuple, List


# ----------------------------- Show markers on frame -----------------------------

def show_frame(
    frame: np.ndarray,
    circles: Iterable[Tuple[float, float]],
    crosses: Iterable[Tuple[float, float]],
    default_radius: int = 10
) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - frame: input BGR image to draw markers on (modified in place and returned).
    - circles: list of (x, y) circle centers (floats or ints).
    - crosses: list of (x, y) cross centers (floats or ints).
    - default_radius: radius in pixels for drawing circle overlays.

    RETURNS:
    - The frame with markers drawn (same object as input, but returned for chaining).

    Notes:
    - Circles are drawn in green with white index labels.
    - Crosses are drawn in magenta with white index labels.
    """
    # Draw circles
    for i, (x, y) in enumerate(circles):
        x_int, y_int = int(round(x)), int(round(y))
        cv2.circle(frame, (x_int, y_int), default_radius, (0, 255, 0), 2)
        cv2.putText(
            frame, f"{i + 1}",
            (x_int + 10, y_int - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (255, 255, 255), 3
        )

    # Draw crosses
    for i, (x, y) in enumerate(crosses):
        x_int, y_int = int(round(x)), int(round(y))
        cv2.drawMarker(frame, (x_int, y_int), (255, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        cv2.putText(
            frame, f"{i + 1}",
            (x_int + 20, y_int - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 5,
            (0, 0, 0), 10
        )

    return frame


# ----------------------------- Zoom into a circle -----------------------------

def zoom_in_on_circle(
    image: np.ndarray,
    circle: Tuple[float, float],
    margin: int = 10,
    r: int = 10
) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - image: input BGR image.
    - circle: (x, y) center of the circle (floats or ints).
    - margin: extra pixels added around the cropped region.
    - r: nominal radius of the circle (default 10 px).

    RETURNS:
    - Cropped BGR sub-image around the circle, with the circle center marked in yellow.

    Notes:
    - Cropped image is clipped to image boundaries.
    - Optionally, you can uncomment the grid overlay code for pixel-level inspection.
    """
    x, y = circle

    # Define bounding box
    x1 = int(max(x - r - margin, 0))
    y1 = int(max(y - r - margin, 0))
    x2 = int(min(x + r + margin, image.shape[1]))
    y2 = int(min(y + r + margin, image.shape[0]))

    # Crop region of interest
    cropped = image[y1:y2, x1:x2].copy()

    # Draw center point relative to cropped region
    cx, cy = int(round(x - x1)), int(round(y - y1))
    cv2.circle(cropped, (cx, cy), 1, (0, 255, 255), 1)  # yellow dot

    # Uncomment for optional pixel grid overlay
    # h, w = cropped.shape[:2]
    # grid_color = (200, 200, 200)
    # for i in range(0, w, 5):
    #     cv2.line(cropped, (i, 0), (i, h), grid_color, 1)
    # for j in range(0, h, 5):
    #     cv2.line(cropped, (0, j), (w, j), grid_color, 1)

    return cropped

# ----------------------------- Zoom into a cross -----------------------------
def zoom_in_on_cross(
    image: np.ndarray,
    cross: Tuple[float, float],
    box_half: int = 12,
    margin: int = 8,
    draw_crosshair: bool = True
) -> np.ndarray:
    """
    Crop a small BGR patch around a detected cross center and annotate it.

    - image: BGR image
    - cross: (x, y) detected center (float/int)
    - box_half: half-size of the square box (in px) around the center (excluding margin)
    - margin: extra pixels added around the cropped region
    - draw_crosshair: if True, draws a fine crosshair at the provided center

    Returns a cropped annotated BGR patch (clipped to image bounds).
    """
    x, y = cross
    r = box_half

    # Bounding box with margin
    x1 = int(max(x - r - margin, 0))
    y1 = int(max(y - r - margin, 0))
    x2 = int(min(x + r + margin, image.shape[1]))
    y2 = int(min(y + r + margin, image.shape[0]))

    cropped = image[y1:y2, x1:x2].copy()

    # Mark the provided center (relative to crop)
    cx, cy = int(round(x - x1)), int(round(y - y1))
    # small center dot
    cv2.circle(cropped, (cx, cy), 1, (0, 255, 255), 1)

    if draw_crosshair:
        # thin crosshair to visualize alignment vs actual marker strokes
        cv2.drawMarker(
            cropped, (cx, cy),
            color=(0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=max(9, 2 * r // 3),
            thickness=1,
            line_type=cv2.LINE_AA
        )

    # Optional: faint grid to inspect pixel alignment
    # h, w = cropped.shape[:2]
    # for i in range(0, w, 5):
    #     cv2.line(cropped, (i, 0), (i, h), (200, 200, 200), 1)
    # for j in range(0, h, 5):
    #     cv2.line(cropped, (0, j), (w, j), (200, 200, 200), 1)

    return cropped


def zoom_debug_windows_crosses(
    left_src: np.ndarray,
    right_src: np.ndarray,
    left_crosses: List[Tuple[float, float]],
    right_crosses: List[Tuple[float, float]],
    scale: int = 10,
    box_half: int = 12
) -> None:
    """
    Open zoomed-in windows for cross centers on both views.
    - scale: integer resize factor for visibility
    - box_half: half box size around center before scaling
    """
    for i, c in enumerate(left_crosses):
        patch = zoom_in_on_cross(left_src, c, box_half=box_half)
        patch = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Zoomed-In Cross LEFT {i+1}", patch)

    for i, c in enumerate(right_crosses):
        patch = zoom_in_on_cross(right_src, c, box_half=box_half)
        patch = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Zoomed-In Cross RIGHT {i+1}", patch)

