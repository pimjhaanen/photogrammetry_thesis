"""This code can be used to synchronise two GoPro (or similar) videos using audio cross-correlation
and an on-camera light flash. It detects the flash start in the reference video (camera 1), aligns
camera 2 to camera 1 via audio, and writes a CSV with matched frame indices **and** a time column
relative to the flash (t=0 at the first bright frame). Times before the flash are negative; after,
they increase by ~1/fps seconds per frame.

You can adapt flash detection sensitivity, the alignment window, and the audio downsampling factor.
"""

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ----------------------------- Parameters -----------------------------
fps_audio = 48000  # Unused directly; retained for clarity. Audio is extracted at 44.1 kHz below.


# ----------------------------- Helpers -----------------------------
def find_continuation_files(filepath: str):
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)
    files = [os.path.join(dirname, filename)]

    if base.startswith("GX") and len(base) >= 6:
        # GX010323 -> want GX020323, GX030323, ...  keep the part after 'GX01' (index 4)
        base_suffix = base[4:]  # <-- FIXED (was 5)
        for i in range(2, 10):
            continuation_name = f"GX0{i}{base_suffix}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break
    else:
        # filename.mp4 -> filename_2.mp4, filename_3.mp4, ...
        for i in range(2, 10):
            continuation_name = f"{base}_{i}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break

    return files



def extract_audio_signal(video_path: str,
                         downsample_factor: int = 50,
                         start_seconds: float = 0.0,
                         duration: Optional[float] = None):
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to a single video file (first file if a continuation set is used).
    - downsample_factor: integer decimation on the audio samples (e.g., 50 for ~882 Hz from 44.1 kHz).
    - start_seconds: where to start the audio extraction within the video, in seconds.
    - duration: optional extraction length in seconds (None = until end).

    Returns:
    - (np.ndarray, float): (mono audio signal as float64, effective_sample_rate_after_decimation).

    Implementation details:
    - Uses ffmpeg to extract mono 44.1 kHz WAV, then decimates by `downsample_factor`.
    """
    import tempfile
    import subprocess
    import scipy.io.wavfile as wavfile

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        tmp_audio_path = tmpfile.name

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_seconds),
    ]
    if duration is not None:
        cmd += ['-t', str(duration)]
    cmd += [
        '-i', video_path,
        '-vn',
        '-ac', '1',
        '-ar', '44100',
        '-loglevel', 'error',
        tmp_audio_path
    ]

    subprocess.run(cmd, check=True)

    sample_rate, data = wavfile.read(tmp_audio_path)
    os.remove(tmp_audio_path)

    data = data[::downsample_factor]
    effective_fps = sample_rate / downsample_factor
    return data.astype(float), effective_fps


def extract_video_frame_timestamps(video_path: str):
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the video file.

    Returns:
    - (np.ndarray, float): (timestamps_in_seconds_for_each_frame, fps_of_video).

    Notes:
    - Timestamps are computed as arange(N) / fps, i.e., first frame is t=0 in the file's own time base.
    """
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    timestamps = np.arange(frame_count, dtype=float) / max(fps, 1e-9)
    return timestamps, fps


def detect_flash_start_frame(
    video_path: str,
    *,
    occurs_after: float = 0.0,               # seconds (lower bound of search window)
    occurs_before: Optional[float] = None,  # seconds (upper bound of search window); None = to end
    center_fraction: float = 1/3,
    min_jump: float = 20.0,
    slope_ratio: float = 5.0,
    baseline_window: int = 5,
    brightness_floor: float = 0.0,
    plot: bool = True,
) -> Optional[int]:

    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the video that contains the flash to detect (typically camera 1).
    - occurs_before: search end time in seconds (None = search to end of file).
    - occurs_after: search start time in seconds.
    - center_fraction: side fraction of the central square ROI (e.g., 1/3 uses the central third).
    - min_jump: absolute minimum Δ brightness (gray levels) to trigger detection.
    - slope_ratio: required factor over the recent baseline slope (median |Δ|) to confirm the jump.
    - baseline_window: number of past Δ samples used to estimate the baseline slope.
    - brightness_floor: minimum post-jump brightness required (optional guard).
    - plot: if True, plots brightness and Δ curves with the detected frame.

    Returns:
    - int | None: index of the *first bright frame* (the frame where the step-up is first visible),
                  or None if not found.

    Heuristic:
    Trigger at the first k such that:
      Δ[k] >= min_jump AND Δ[k] >= slope_ratio * median(|Δ| over previous `baseline_window`)
      AND post_brightness >= brightness_floor
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = float(cap.get(cv2.CAP_PROP_FPS))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute frame limits from time window
    start_frame = int(max(0, occurs_after * fps))
    if occurs_before is None:
        end_frame = frame_count
    else:
        end_frame = int(min(frame_count, occurs_before * fps))

    if start_frame >= end_frame:
        cap.release()
        return None

    # Seek to start of window
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Central box ROI
    fx = fy = center_fraction
    x0, x1 = int((1 - fx) * 0.5 * width),  int((1 + fx) * 0.5 * width)
    y0, y1 = int((1 - fy) * 0.5 * height), int((1 + fy) * 0.5 * height)

    brightness = []
    i = start_frame
    while i < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = gray[y0:y1, x0:x1]
        brightness.append(float(np.mean(roi)))
        i += 1

    cap.release()

    if not brightness or len(brightness) < baseline_window + 2:
        if plot:
            plt.figure(figsize=(12, 4))
            if brightness:
                x = np.arange(start_frame, start_frame + len(brightness))
                plt.plot(x, brightness, label='Brightness')
            plt.title("Flash Start Detection (insufficient data)")
            plt.xlabel("Frame Index"); plt.ylabel("Brightness")
            plt.grid(True); plt.tight_layout(); plt.show()
        return None

    brightness = np.asarray(brightness, dtype=float)
    d = np.diff(brightness)  # Δ per frame

    flash_start = None
    eps = 1e-9
    for k in range(baseline_window, len(d)):
        recent = np.abs(d[max(0, k - baseline_window):k])
        baseline = np.median(recent) if recent.size else 0.0
        if (d[k] >= min_jump) and (d[k] >= slope_ratio * max(baseline, eps)):
            post_brightness = brightness[k + 1]
            if post_brightness >= brightness_floor:
                # First bright frame is the frame after the transition; map back to absolute frame index.
                flash_start = start_frame + k
                break

    if plot:
        x_frames = np.arange(start_frame, start_frame + len(brightness))
        plt.figure(figsize=(12, 6))
        # Brightness
        plt.subplot(2, 1, 1)
        plt.plot(x_frames, brightness, label='Brightness')
        if flash_start is not None:
            plt.axvline(flash_start, linestyle='--', label='Flash start', linewidth=1.5)
        if occurs_before is not None:
            plt.xlim(occurs_after * fps, occurs_before * fps)
        plt.ylabel("Brightness"); plt.title("Flash Start Detection (brightness and slope)")
        plt.grid(True); plt.legend()
        # Slope
        x_slope = np.arange(start_frame + 1, start_frame + len(brightness))
        plt.subplot(2, 1, 2)
        plt.plot(x_slope, d, label='Δ brightness/frame')
        if flash_start is not None:
            plt.axvline(flash_start, linestyle='--', label='Flash start', linewidth=1.5)
        if occurs_before is not None:
            plt.xlim(occurs_after * fps, occurs_before * fps)
        plt.xlabel("Frame Index"); plt.ylabel("Δ Brightness")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.show()

    return flash_start


# ----------------------------- Main synchronisation -----------------------------
def match_videos(video1_path: str,
                 video2_path: str,
                 start_seconds: float,
                 match_duration: float = 300.0,
                 downsample_factor: int = 50,
                 plot: bool = False,
                 output_dir: str = 'synchronised_frame_indices',
                 # NEW: flash controls
                 flash_occurs_after: float = 0.0,
                 flash_occurs_before: Optional[float] = None,
                 flash_center_fraction: float = 1/3,
                 flash_min_jump: float = 20.0,
                 flash_slope_ratio: float = 5.0,
                 flash_baseline_window: int = 5,
                 flash_brightness_floor: float = 0.0,
                 flash_plot: bool = True) -> str:
    """RELEVANT FUNCTION INPUTS:
    - video1_path: path to the reference video (camera 1). The flash is assumed to be visible here.
    - video2_path: path to the second video (camera 2) to be aligned to camera 1.
    - start_seconds: point in time (s) from which to extract audio for correlation.
    - match_duration: length (s) of audio to compare.
    - downsample_factor: integer decimation factor for audio (e.g., 50 → ~882 Hz).
    - plot: if True, shows a plot of audio alignment.
    - output_dir: directory where the matched CSV will be written.

    What it does:
    1) Finds any continuation files (for info). Uses the first file of each for audio & timestamps.
    2) Extracts audio and finds relative lag (audio2 vs audio1).
    3) Detects the *start of the flash* in the reference video (video 1 by default).
    4) Aligns timestamps of both videos so that t=0 equals the flash start in camera 1.
       (Camera 2 is shifted by the measured audio lag.)
    5) Matches frames by nearest time (within 1/max(fps1,fps2)).
    6) Writes CSV with columns: Time_s, Frame_Video1, Frame_Video2.

    Returns:
    - str: path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base1 = os.path.splitext(os.path.basename(video1_path))[0]
    base2 = os.path.splitext(os.path.basename(video2_path))[0]
    output_file = os.path.join(output_dir, f"{base1}_{base2}_matched_frames.csv")

    if os.path.exists(output_file):
        print(f"Matched frame file already exists: {output_file}")
        return output_file

    # Step 1: Get (possibly merged) file paths (info only; we use the first file in each).
    video1_files = find_continuation_files(video1_path)
    video2_files = find_continuation_files(video2_path)
    print(f"Found {len(video1_files)} files for left/reference camera: {video1_files}")
    print(f"Found {len(video2_files)} files for right/secondary camera: {video2_files}")

    # Step 2: Audio cross-correlation to get lag (audio2 relative to audio1).
    audio1, eff_fps = extract_audio_signal(video1_files[0], downsample_factor, start_seconds, match_duration)
    audio2, _       = extract_audio_signal(video2_files[0], downsample_factor, start_seconds, match_duration)

    match_samples = int(match_duration * eff_fps)
    audio1 = audio1[:match_samples]
    audio2 = audio2[:match_samples]

    correlation = np.correlate(audio1 - np.mean(audio1), audio2 - np.mean(audio2), mode='full')
    lag_samples = int(np.argmax(correlation) - (len(audio2) - 1))
    lag_seconds = lag_samples / eff_fps
    print(f"Matched lag: {lag_samples} samples → {lag_seconds:.3f} s (audio2 relative to audio1)")

    if plot:
        time = np.linspace(0, len(audio1) / eff_fps, len(audio1))
        plt.figure(figsize=(15, 6))
        plt.plot(time, audio1, label='Video 1 (ref)')
        if lag_samples > 0:
            shifted = np.pad(audio2, (lag_samples, 0), mode='constant')[:len(audio1)]
        else:
            shifted = audio2[-lag_samples:]
            pad_len = len(audio1) - len(shifted)
            shifted = np.pad(shifted, (0, max(0, pad_len)), mode='constant')[:len(audio1)]
        plt.plot(time, shifted, label='Video 2 (shifted by lag)')
        plt.title('Matched Audio Signals')
        plt.xlabel('Time [s]'); plt.ylabel('Amplitude')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Step 3: Detect start-of-flash in the reference video (video 1).
    # Step 3: Detect start-of-flash in the reference video (video 1).
    flash_video_path = video1_files[0]
    flash_start_frame = detect_flash_start_frame(
        flash_video_path,
        occurs_after=flash_occurs_after,
        occurs_before=flash_occurs_before,
        center_fraction=flash_center_fraction,
        min_jump=flash_min_jump,
        slope_ratio=flash_slope_ratio,
        baseline_window=flash_baseline_window,
        brightness_floor=flash_brightness_floor,
        plot=flash_plot,
    )
    use_flash_alignment = flash_start_frame is not None

    # Step 4: Frame timestamps
    timestamps1, fps1 = extract_video_frame_timestamps(video1_files[0])
    timestamps2, fps2 = extract_video_frame_timestamps(video2_files[0])

    if use_flash_alignment:
        # Establish t=0 at flash start in camera 1
        cap_tmp = cv2.VideoCapture(flash_video_path)
        flash_video_fps = float(cap_tmp.get(cv2.CAP_PROP_FPS))
        cap_tmp.release()

        flash_start_time = flash_start_frame / max(flash_video_fps, 1e-9)
        print(f"Start of flash at frame {flash_start_frame} (≈ {flash_start_time:.3f} s) in reference video")

        # Camera 1: set t=0 at its flash
        t1 = timestamps1 - flash_start_time

        # Camera 2: align by the measured audio lag (audio2 lags audio1 by +lag_seconds)
        # If the flash happened at t=0 in cam1, then cam2 timeline should be shifted so
        # that events align: subtract (flash_time_in_cam1 - lag) → subtract (-lag) → add lag
        # We want both to express time in cam1's flash-centric frame:
        t2 = timestamps2 - (flash_start_time - lag_seconds)
    else:
        print("⚠️ No flash detected — using audio-only alignment; t=0 will be cam1 t=0.")
        # Fall back: shift cam2 by lag, cam1 starts at 0 (no negative pre-roll).
        t1 = timestamps1.copy()
        t2 = timestamps2 - lag_seconds

    # Step 5: Frame matching (nearest-neighbour within 1/max fps)
    tol = 1.0 / max(fps1, fps2) if max(fps1, fps2) > 0 else 1/60.0
    matched_pairs = []
    for i, t in enumerate(t1):
        j = int(np.argmin(np.abs(t2 - t)))
        if abs(t2[j] - t) < tol:
            matched_pairs.append((t, i, j))  # keep time t (relative to flash if available)

    # Step 6: Write CSV with Time_s (relative to flash), Frame_Video1, Frame_Video2
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_s', 'Frame_Video1', 'Frame_Video2'])
        for t, i1, i2 in matched_pairs:
            writer.writerow([f"{t:.6f}", i1, i2])

    print(f"Saved {len(matched_pairs)} matched frame pairs to: {output_file}")
    if use_flash_alignment:
        print("Time column is relative to flash start (t=0 at first bright frame in camera 1).")
    else:
        print("Time column is relative to camera 1's own timeline (audio-only alignment).")

    return output_file


# ----------------------------- Example usage -----------------------------
if __name__ == "__main__":
    # Example call (edit paths):
    # The output CSV will include a Time_s column with t=0 at the flash (if detected).
    csv_path = match_videos(
        video1_path="../static_experiments/left_input_static/GX010332_merged.MP4",     # reference camera (flash visible here)
        video2_path="../static_experiments/right_input_static/GX010359_merged.MP4",     # secondary camera
        start_seconds=0.0,
        match_duration=30.0,
        downsample_factor=50,
        plot=True,
        output_dir="synchronised_frame_indices"
    )
    print(csv_path)
