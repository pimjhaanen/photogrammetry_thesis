import cv2
import numpy as np

# Global variable to store the clicked point
clicked_point = None

# Mouse callback function to capture the clicked point
import cv2

clicked_point = None  # Global clicked point

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def pick_color(frame, alpha=1.0, beta=0):
    """
    Adjusts brightness and contrast of the input BGR frame,
    displays it, and allows the user to click on pixels to get HSV values.

    Parameters:
    - frame: BGR image (numpy array)
    - alpha: Contrast control (1.0 = no change)
    - beta: Brightness control (0 = no change)
    """
    global clicked_point

    # Step 1: Adjust brightness and contrast in BGR space
    adjusted_bgr = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Step 2: Convert adjusted image to HSV for value lookup
    frame_hsv = cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2HSV)

    # Step 3: Show the adjusted BGR image
    adjusted_bgr = cv2.resize(adjusted_bgr, None, fx=0.5, fy=0.5)
    cv2.imshow("Adjusted Frame (BGR)", adjusted_bgr)
    cv2.setMouseCallback("Adjusted Frame (BGR)", mouse_callback)

    while True:
        if clicked_point is not None:
            x, y = clicked_point
            hsv_value = frame_hsv[y, x]
            print(f"HSV value at ({x}, {y}): {hsv_value}")
            clicked_point = None  # Reset click

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# Main function to load and show the frame
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

    # Pick color interactively
    pick_color(frame)

    cap.release()
# Call the main function with the video path
video_path = "../../Accuracy_analysis/video_input/left camera/frame_003.jpg"  # Replace with your video file path
main(video_path)
