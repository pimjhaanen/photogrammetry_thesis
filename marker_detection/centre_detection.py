import cv2
import numpy as np


# Function to zoom in on detected marker
def zoom_in_on_marker(frame, center, size=50):
    x, y = center
    h, w = frame.shape[:2]

    # Make sure we don't go out of bounds
    x = max(min(x, w - size), size)
    y = max(min(y, h - size), size)

    zoomed_in = frame[y - size:y + size, x - size:x + size]
    return zoomed_in


# Detect intersection of pie sections using corner detection for a specific color pair
def detect_crash_dummy_circles_v2(frame, color_range1, color_range2, min_size=5):
    """
    Detect the intersection of two color segments using corner detection.

    :param frame: Input frame
    :param color_range1: Lower and upper bounds for the first color in HSV
    :param color_range2: Lower and upper bounds for the second color in HSV
    :param min_size: Minimum size threshold for detecting intersections (in pixels)
    :return: frame with detected intersection points marked
    """
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both colors
    mask1 = cv2.inRange(hsv, color_range1[0], color_range1[1])
    mask2 = cv2.inRange(hsv, color_range2[0], color_range2[1])

    # Find the intersection (AND operation) between the two color masks
    intersection_mask = cv2.bitwise_and(mask1, mask2)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    intersection_mask = cv2.morphologyEx(intersection_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    intersection_mask = cv2.dilate(intersection_mask, kernel, iterations=1)

    # Use Harris corner detection on the intersection mask
    corners = cv2.cornerHarris(intersection_mask, 2, 3, 0.04)

    # Threshold the corners to only keep strong corners
    corners = cv2.dilate(corners, None)

    # Only proceed if we have some corner points
    corner_points = np.argwhere(corners > 0.01 * corners.max())

    if corner_points.size == 0:
        return frame, []  # Return the frame if no corners are detected

    # Filter corner points based on a minimum size threshold (e.g., 10 pixels)
    filtered_corners = []
    for point in corner_points:
        x, y = point
        if cv2.contourArea(np.array([[[x, y]]])) > min_size:
            filtered_corners.append((x, y))

    # Convert to np.float32 for subpixel refinement
    corners_refined = np.float32(filtered_corners)

    # Mark the corners where color segments intersect
    frame_with_intersections = frame.copy()
    for corner in corners_refined:
        x, y = corner  # `corner` is now correctly unpacked as (x, y)
        cv2.drawMarker(frame_with_intersections, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    return frame_with_intersections, corners_refined


# Main function
def main(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)

    # Ask for frame number
    frame_number = int(input("Enter the frame number to check: "))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the video to the specified frame
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame.")
        return

    # Define the HSV ranges for each color pair

    # Green and Red (around [59,  90, 112] to [64, 106, 122] for Green, [ 3, 135, 164] to [ 5, 143, 178] for Red)
    green_lower = np.array([50, 80, 100])  # Lower bound for green
    green_upper = np.array([70, 130, 130])  # Upper bound for green
    red_lower = np.array([0, 100, 100])  # Lower bound for red
    red_upper = np.array([10, 180, 190])  # Upper bound for red

    # Blue and Orange (around [111, 200, 154] to [110, 207, 159] for Blue, [177, 145, 188] to [179, 137, 171] for Orange)
    blue_lower = np.array([100, 120, 100])  # Lower bound for blue
    blue_upper = np.array([130, 220, 180])  # Upper bound for blue
    orange_lower = np.array([160, 120, 140])  # Lower bound for orange
    orange_upper = np.array([180, 160, 200])  # Upper bound for orange

    # Purple and Yellow (around [122, 120, 113] to [125, 115, 106] for Purple, [29, 178, 159] to [28, 179, 168] for Yellow)
    purple_lower = np.array([120, 100, 100])  # Lower bound for purple
    purple_upper = np.array([135, 130, 140])  # Upper bound for purple
    yellow_lower = np.array([20, 170, 140])  # Lower bound for yellow
    yellow_upper = np.array([40, 200, 180])  # Upper bound for yellow

    # Detect intersections for each color pair
    frame_with_intersections_green_red, corners_green_red = detect_crash_dummy_circles_v2(frame, (green_lower, green_upper), (red_lower, red_upper))
    frame_with_intersections_blue_orange, corners_blue_orange = detect_crash_dummy_circles_v2(frame, (blue_lower, blue_upper), (orange_lower, orange_upper))
    frame_with_intersections_purple_yellow, corners_purple_yellow = detect_crash_dummy_circles_v2(frame, (purple_lower, purple_upper), (yellow_lower, yellow_upper))

    # Show the frame with detected intersection markers for each color pair
    resized_frame_green_red = cv2.resize(frame_with_intersections_green_red, (0, 0), fx=0.4, fy=0.4)
    resized_frame_blue_orange = cv2.resize(frame_with_intersections_blue_orange, (0, 0), fx=0.4, fy=0.4)
    resized_frame_purple_yellow = cv2.resize(frame_with_intersections_purple_yellow, (0, 0), fx=0.4, fy=0.4)

    cv2.imshow("Green and Red Intersections", resized_frame_green_red)
    cv2.imshow("Blue and Orange Intersections", resized_frame_blue_orange)
    cv2.imshow("Purple and Yellow Intersections", resized_frame_purple_yellow)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

# Call the main function with the video path
video_path = "GX010268.mp4"  # Replace with your video file path
main(video_path)
