"""This code can be used to detect red cross markers in one or more videos and report the smallest
detected size (in pixels). It scans frames at a fixed step, segments red in HSV, cleans up the mask
with morphology, extracts contours within an area range, and (optionally) visualizes detections.

You can point it at a folder with videos (e.g., 'output/*.MP4') or a single file. It prints per-frame
detections and returns the minimum detected cross size per video."""

import os
import glob
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def detect_red_crosses_in_videos(
    video_dir: str = "test_videos/crosses",
    video_glob: str = "*.MP4",
    frame_step: int = 10,
    start_frame: int = 0,
    min_area_px2: int = 60,
    max_area_px2: int = 5000,
    hsv_lower1: Tuple[int, int, int] = (0, 100, 100),
    hsv_upper1: Tuple[int, int, int] = (10, 255, 255),
    hsv_lower2: Tuple[int, int, int] = (160, 100, 100),
    hsv_upper2: Tuple[int, int, int] = (179, 255, 255),
    morph_kernel_size: int = 5,
    visualize: bool = True,
) -> Dict[str, Optional[float]]:

    """RELEVANT FUNCTION INPUTS:
    - video_dir / video_glob: where to find input videos (e.g., 'output', '*.MP4')
    - frame_step: process every Nth frame
    - start_frame: starting frame index
    - min_area_px2 / max_area_px2: contour area bounds to accept as crosses (in pixel^2)
    - hsv_lower1/upper1 + hsv_lower2/upper2: two HSV ranges for red segmentation
    - morph_kernel_size: size of the morphology kernel (odd integer; e.g., 5)
    - visualize: if True, show an annotated preview while processing"""

    # Expand input set: support either a directory glob or a single explicit video path
    video_paths = []
    if os.path.isfile(video_dir):
        video_paths = [video_dir]  # user passed a single file in video_dir
    else:
        video_paths = sorted(glob.glob(os.path.join(video_dir, video_glob)))

    if not video_paths:
        print(f"❌ No videos found at: {video_dir}/{video_glob}" if not os.path.isfile(video_dir)
              else f"❌ File not found: {video_dir}")
        return {}

    results: Dict[str, Optional[float]] = {}
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    for vid in video_paths:
        base = os.path.basename(vid)
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"❌ Could not open video: {base}")
            results[base] = None
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n▶ Processing: {base} | 🎞️ Total frames: {total_frames}")

        frame_idx = start_frame
        min_detected_size = float("inf")

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(f"⚠️ Could not read frame {frame_idx}")
                frame_idx += frame_step
                continue

            print(f"🔍 Frame {frame_idx}")

            # --- Red segmentation in HSV (two ranges for wraparound) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array(hsv_lower1), np.array(hsv_upper1))
            mask2 = cv2.inRange(hsv, np.array(hsv_lower2), np.array(hsv_upper2))
            red_mask = cv2.bitwise_or(mask1, mask2)

            # --- Morphological cleanup ---
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            red_mask = cv2.dilate(red_mask, kernel, iterations=1)

            # --- Contours and filtering ---
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sizes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area_px2 or area > max_area_px2:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Simple diameter estimate from area (pixels)
                size_px = float(np.sqrt(area))
                sizes.append(int(round(size_px)))
                min_detected_size = min(min_detected_size, size_px)

                if visualize:
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(frame, f"{int(round(size_px))} px", (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if sizes:
                print(f"✅ {len(sizes)} red cross(es): {sizes}")
            else:
                print(f"❌ No red crosses")

            if visualize:
                disp = cv2.resize(frame, None, fx=0.4, fy=0.4)
                cv2.imshow(f"Red Cross Detection - {base}", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += frame_step

        cap.release()
        if visualize:
            cv2.destroyWindow(f"Red Cross Detection - {base}")

        results[base] = min_detected_size if min_detected_size < float("inf") else None
        if results[base] is not None:
            print(f"✅ Smallest detected red cross in {base}: {results[base]:.1f} px")
        else:
            print(f"❌ No red crosses detected in {base} at all.")

    if visualize:
        cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    # Example run on the 'output' folder. Adjust to your dataset as needed.
    detect_red_crosses_in_videos(
        video_dir="test_videos/crosses",
        video_glob="*.MP4",
        frame_step=10,
        start_frame=0,
        min_area_px2=60,
        max_area_px2=5000,
        visualize=True,
    )
