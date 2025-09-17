"""This code can be used to scan videos for smallest detectable ArUco markers and black circular blobs.
It iterates frames with a step, detects sizes (in pixels), optionally visualizes detections, and saves a CSV summary.
The test for crosses was performed later, and is therefore in a different file"""

import os
import glob
import csv
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ----------------------------- Detection functions -----------------------------

def detect_aruco_in_video(
    video_path: str,
    aruco_dict_types: Dict[str, int],
    frame_step: int = 10,
    start_frame: int = 0,
    min_occurrences: int = 4,
    visualize: bool = False,
) -> Dict[str, Optional[int]]:
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the input video
    - aruco_dict_types: mapping like {"4x4": cv2.aruco.DICT_4X4_100, ...}
    - frame_step: sample every Nth frame
    - start_frame: frame index to start scanning from
    - min_occurrences: minimum repeats of a size for it to be considered stable
    - visualize: if True, shows annotated frames during scanning

    Returns: dict { "<dict_name>": size_px_or_None, ... }
    """
    results = {k: None for k in aruco_dict_types.keys()}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return results

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure ArUco parameters (slightly permissive)
    params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.001
    params.maxMarkerPerimeterRate = 4.0
    params.minDistanceToBorder = 1

    for dict_name, dict_id in aruco_dict_types.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        seen_sizes: Dict[int, int] = {}
        frame_index = start_frame

        while frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                frame_index += frame_step
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

            if ids is not None and len(corners) > 0:
                for marker_corners in corners:
                    pts = marker_corners[0]
                    w = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                    h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                    size = max(w, h)
                    seen_sizes[size] = seen_sizes.get(size, 0) + 1

                    if visualize:
                        pts_i = pts.astype(int)
                        cv2.polylines(frame, [pts_i], True, (0, 255, 0), 2)
                        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        cv2.putText(frame, f"{size}px", (cx, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if visualize:
                disp = cv2.resize(frame, None, fx=0.5, fy=0.5)
                cv2.imshow(f"ArUco {dict_name}", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_index += frame_step

        # Choose a stable size (first that reaches min_occurrences, smallest first)
        stable = [s for s, c in sorted(seen_sizes.items()) if c >= min_occurrences]
        results[dict_name] = stable[0] if len(stable) > 0 else None

        if visualize:
            cv2.destroyWindow(f"ArUco {dict_name}")

    cap.release()
    return results


def detect_black_circles_in_video(
    video_path: str,
    frame_step: int = 10,
    start_frame: int = 0,
    min_area: int = 30,
    max_area: int = 5000,
    min_circularity: float = 0.9,
    min_occurrences: int = 4,
    visualize: bool = False,
) -> Optional[int]:
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the input video
    - frame_step: sample every Nth frame
    - start_frame: frame index to start scanning from
    - min_area / max_area: blob area bounds (px^2)
    - min_circularity: 0..1, higher is more circle-like
    - min_occurrences: minimum repeats of a size for it to be considered stable
    - visualize: if True, shows annotated frames during scanning

    Returns: most frequently recurring (stable) diameter in pixels (int) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return None

    # Blob detector configuration
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0  # black
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = start_frame
    seen_sizes: Dict[int, int] = {}

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            frame_index += frame_step
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        kps = detector.detect(blur)

        if visualize:
            vis = frame.copy()

        if kps:
            for kp in kps:
                d = int(round(kp.size))  # diameter in px
                seen_sizes[d] = seen_sizes.get(d, 0) + 1
                if visualize:
                    x, y = map(int, kp.pt)
                    cv2.circle(vis, (x, y), d // 2, (0, 255, 0), 2)
                    cv2.putText(vis, f"{d}px", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if visualize:
            disp = cv2.resize(vis if kps else frame, None, fx=0.5, fy=0.5)
            cv2.imshow("Black Circle Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_index += frame_step

    cap.release()
    if visualize:
        cv2.destroyWindow("Black Circle Detection")

    stable = [s for s, c in sorted(seen_sizes.items()) if c >= min_occurrences]
    return stable[0] if len(stable) > 0 else None


# ----------------------------- Orchestration + CSV -----------------------------

def analyze_videos_and_save_csv(
    video_dir: str = "test_videos/aruco_and_circles",
    video_glob: str = "*.MP4",
    aruco_dicts: Dict[str, int] = None,
    frame_step: int = 10,
    start_frame: int = 0,
    circle_min_area: int = 30,
    circle_max_area: int = 5000,
    circle_min_circularity: float = 0.9,
    min_occurrences: int = 4,
    visualize: bool = False,
    csv_path: str = "detection_results.csv",
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - video_dir / video_glob: where to find videos (e.g., 'output', '*.MP4')
    - aruco_dicts: mapping like {"4x4": cv2.aruco.DICT_4X4_100, ...}
    - frame_step / start_frame: sampling controls
    - circle_min_area / circle_max_area / circle_min_circularity: blob detector constraints
    - min_occurrences: stability criterion for size reporting
    - visualize: if True, shows detections during processing
    - csv_path: output CSV file path
    """
    if aruco_dicts is None:
        aruco_dicts = {
            "4x4": cv2.aruco.DICT_4X4_100,
            "5x5": cv2.aruco.DICT_5X5_100,
            "6x6": cv2.aruco.DICT_6X6_100,
            "7x7": cv2.aruco.DICT_7X7_100,
        }

    videos = sorted(glob.glob(os.path.join(video_dir, video_glob)))
    if len(videos) == 0:
        print(f"❌ No videos found in {video_dir}/{video_glob}")
        return

    rows: List[List[str]] = []
    print(f"🗂 Found {len(videos)} video(s) in '{video_dir}'")

    for v in videos:
        print(f"\n▶ Processing: {os.path.basename(v)}")

        # ArUco sizes
        aruco_sizes = detect_aruco_in_video(
            v, aruco_dicts, frame_step, start_frame, min_occurrences, visualize
        )
        for dict_name, size in aruco_sizes.items():
            rows.append(["aruco_"+dict_name, os.path.basename(v), str(size) if size is not None else "Not detected"])
            print(f"   ArUco {dict_name}: {size if size is not None else 'Not detected'} px")

        # Circle sizes
        circ_size = detect_black_circles_in_video(
            v, frame_step, start_frame, circle_min_area, circle_max_area,
            circle_min_circularity, min_occurrences, visualize
        )
        rows.append(["circle", os.path.basename(v), str(circ_size) if circ_size is not None else "Not detected"])
        print(f"   Circle: {circ_size if circ_size is not None else 'Not detected'} px")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "video", "size_px"])
        w.writerows(rows)

    print(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    analyze_videos_and_save_csv(
        video_dir="test_videos/aruco_and_circles",
        video_glob="*.MP4",
        frame_step=10,
        start_frame=500,           # match your earlier setting; change if needed
        circle_min_area=30,
        circle_max_area=5000,
        circle_min_circularity=0.9,
        min_occurrences=4,
        visualize=True,            # set False to disable windows and speed up
        csv_path="detection_results.csv"
    )
