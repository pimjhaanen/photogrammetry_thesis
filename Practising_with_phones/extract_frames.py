import cv2
import os

def extract_frames(video_path, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(orig_fps // fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1
    cap.release()
    print(f"Extracted {frame_id} frames to {output_dir}")