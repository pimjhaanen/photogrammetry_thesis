import cv2

def show_frame(frame, circles, crosses, default_radius=10):
    for i, (x, y) in enumerate(circles):
        x_int, y_int = int(round(x)), int(round(y))
        cv2.circle(frame, (x_int, y_int), default_radius, (0, 255, 0), 2)
        cv2.putText(frame, f"{i + 1}", (x_int + 10, y_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    for i, (x, y) in enumerate(crosses):
        cv2.drawMarker(frame, (x, y), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        cv2.putText(frame, f"{i + 1}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return frame


def zoom_in_on_circle(image, circle, margin=10):
    """
    Zoom in on a circle with some margin around it.
    :param image: The input image.
    :param circle: The circle in the format (x, y, radius).
    :param margin: The margin around the circle to zoom in.
    :return: Cropped image around the circle.
    """
    r = 10
    x, y = circle  # Keep floats!

    # Define a square region
    x1 = int(max(x - r - margin, 0))
    y1 = int(max(y - r - margin, 0))
    x2 = int(min(x + r + margin, image.shape[1]))
    y2 = int(min(y + r + margin, image.shape[0]))

    # Crop
    cropped = image[y1:y2, x1:x2].copy()

    # Calculate center relative to cropped
    center_in_cropped = (int(round(x - x1)), int(round(y - y1)))

    # Draw the center point
    cv2.circle(cropped, center_in_cropped, 1, (0, 255, 255), 1)
    """
    # Add pixel grid
    h, w = cropped.shape[:2]
    grid_color = (200, 200, 200)
    for i in range(0, w, 5):
        cv2.line(cropped, (i, 0), (i, h), grid_color, 1)
    for j in range(0, h, 5):
        cv2.line(cropped, (0, j), (w, j), grid_color, 1)
    """
    return cropped