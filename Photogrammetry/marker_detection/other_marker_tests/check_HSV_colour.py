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

def pick_color(frame, alpha=4.0, beta=20, scale=0.5):
    """
    Shows a brightness/contrast boosted, *resized* frame and reports HSV values
    from the *same* resized image (so clicks line up).
    """
    global clicked_point

    # 1) Brightness/contrast boost (kept)
    adjusted_bgr_full = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # 2) Resize for display
    disp_bgr = cv2.resize(adjusted_bgr_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 3) Build HSV of the *display* image so coordinates match
    disp_hsv = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2HSV)

    # 4) Show and read clicks
    win = "Adjusted Frame (BGR)"
    cv2.imshow(win, disp_bgr)
    cv2.setMouseCallback(win, mouse_callback)

    h, w = disp_hsv.shape[:2]
    while True:
        if clicked_point is not None:
            x, y = clicked_point
            # clamp into bounds
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            hsv_value = disp_hsv[y, x]
            print(f"HSV value at ({x}, {y}) [scaled view]: {hsv_value}")
            clicked_point = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
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
video_path = "../../input/right_videos/09_10_merged.MP4"  # Replace with your video file path
main(video_path)
