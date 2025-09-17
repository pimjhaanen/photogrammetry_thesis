"""This code can be used to visualize detected markers in stereo photogrammetry.
It provides utilities to:
(1) overlay detected circles and crosses on an image, with indices,
(2) zoom into a circle region with an adjustable margin for inspection.
"""

import cv2
import numpy as np
from typing import Iterable, Tuple


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
            (x_int + 10, y_int - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (255, 255, 255), 3
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