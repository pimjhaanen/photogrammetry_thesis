import cv2
import cv2.aruco as aruco

# Initialize the video capture (can be a video file or a camera)
video_path = 'no_distortion.MP4.mp4'  # Replace with your video file path or set to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Define the ArUco dictionary and the parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # You can change the dictionary here
parameters = aruco.DetectorParameters_create()

# Start reading the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are found, draw them and show message
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, 'ArUco Marker Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No ArUco Marker Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    # Set a fixed window size for the display
    cv2.namedWindow('ArUco Marker Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ArUco Marker Detection', 800, 600)  # Resize the window (width x height)

    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
