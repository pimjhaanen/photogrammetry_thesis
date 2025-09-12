import cv2
import numpy as np
import os
import csv
import subprocess
import tempfile
from scipy.io import wavfile
import moviepy.editor as mp
import matplotlib.pyplot as plt

# Parameters
fps_audio = 48000


def find_continuation_files(filepath):
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)

    files = [os.path.join(dirname, filename)]

    if base.startswith("GX") and len(base) >= 6:
        # GoPro-style continuation: GX010237.mp4 -> GX020237.mp4, GX030237.mp4, ...
        base_prefix = base[:5]
        base_suffix = base[5:]
        for i in range(2, 10):
            continuation_name = f"GX0{i}{base_suffix}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break
    else:
        # Generic-style continuation: filename.mp4 -> filename_2.mp4, filename_3.mp4, ...
        for i in range(2, 10):
            continuation_name = f"{base}_{i}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break

    return files


def extract_audio_signal(video_path, downsample_factor=50, start_seconds=0, duration=None):
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

    # Downsample
    data = data[::downsample_factor]
    effective_fps = sample_rate / downsample_factor
    return data.astype(float), effective_fps


def extract_video_frame_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    timestamps = np.arange(frame_count) / fps
    return timestamps, fps

def detect_flash_start_frame(
    video_path,
    *,
    occurs_before=10,         # seconds (optional)
    occurs_after=0,          # seconds (optional)
    center_fraction=1/3,        # box size as fraction of width/height
    min_jump=20.0,              # absolute min jump in brightness (gray levels)
    slope_ratio=5.0,            # how many times larger than recent baseline slope (e.g., 5x = 500%)
    baseline_window=5,          # frames to estimate recent baseline slope (median of |Δ|)
    brightness_floor=0.0,       # optional floor for post-jump brightness
    plot=True
):
    """
    Detect the *start* of a flash using the discrete slope (Δ brightness per frame)
    in the central region of the image.

    Heuristic: trigger at the first frame k where
      Δ_brightness[k] >= min_jump  AND
      Δ_brightness[k] >= slope_ratio * median(|Δ_brightness| over previous `baseline_window`)
      AND brightness[k] >= brightness_floor (optional)

    Returns:
        int | None: Frame index of the *first bright frame of the flash*, or None if not found.
    """
    occurs_before = occurs_before *30
    occurs_after = occurs_after * 30 # Go from s to fps
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = float(cap.get(cv2.CAP_PROP_FPS))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute frame limits from time window (if provided)
    start_frame = int(max(0, (occurs_after or 0.0) * fps))
    end_frame   = int(min(frame_count, (occurs_before * fps) if occurs_before is not None else frame_count))

    if start_frame >= end_frame:
        cap.release()
        return None

    # Seek to start
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Central box (center_fraction of width/height)
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
        # Not enough data to compute a robust slope
        if plot:
            plt.figure(figsize=(12, 4))
            if brightness:
                x = np.arange(start_frame, start_frame + len(brightness))
                plt.plot(x, brightness, label='Brightness')
            plt.title("Flash Start Detection (insufficient data)")
            plt.xlabel("Frame Index")
            plt.ylabel("Brightness")
            plt.grid(True); plt.tight_layout(); plt.show()
        return None

    # Discrete slope (Δ per frame)
    brightness = np.asarray(brightness, dtype=float)
    d = np.diff(brightness)  # length N-1, corresponds to transitions k->k+1

    flash_start = None
    eps = 1e-9
    for k in range(baseline_window, len(d)):
        recent = np.abs(d[max(0, k - baseline_window):k])
        baseline = np.median(recent) if recent.size else 0.0
        # Require both a minimum absolute jump and a multiple of the baseline
        if (d[k] >= min_jump) and (d[k] >= slope_ratio * max(baseline, eps)):
            # Also enforce a minimum post-jump brightness if desired
            post_brightness = brightness[k + 1]
            if post_brightness >= brightness_floor:
                # The *first bright frame* is k+1 frames from start_frame
                flash_start = start_frame + (k)
                break

    if plot:
        x_frames = np.arange(start_frame, start_frame + len(brightness))
        plt.figure(figsize=(12, 6))

        # Top: brightness
        plt.subplot(2, 1, 1)
        plt.plot(x_frames, brightness, label='Brightness')
        plt.xlim(occurs_after,occurs_before)
        if flash_start is not None:
            plt.axvline(flash_start, linestyle='--', label='Flash start', linewidth=1.5)
        plt.ylabel("Brightness")
        plt.title("Flash Start Detection (brightness and slope)")
        plt.grid(True); plt.legend()

        # Bottom: slope
        x_slope = np.arange(start_frame + 1, start_frame + len(brightness))
        plt.subplot(2, 1, 2)
        plt.plot(x_slope, d, label='Δ brightness/frame')
        plt.xlim(occurs_after, occurs_before)
        if flash_start is not None:
            plt.axvline(flash_start, linestyle='--', label='Flash start', linewidth=1.5)
        plt.xlabel("Frame Index"); plt.ylabel("Δ Brightness")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.show()

    return flash_start


frame_idx = detect_flash_start_frame("test_led_7.MP4")


def match_videos(video1_path, video2_path, start_seconds, match_duration=300, downsample_factor=50, plot=False, output_dir='matched_sync'):
    os.makedirs(output_dir, exist_ok=True)
    base1 = os.path.splitext(os.path.basename(video1_path))[0]
    base2 = os.path.splitext(os.path.basename(video2_path))[0]
    output_file = os.path.join(output_dir, f"{base1}_{base2}_matched_frames.csv")

    if os.path.exists(output_file):
        print(f"Matched frame file already exists: {output_file}")
        return output_file

    # Step 1: Get (possibly merged) file paths
    video1_files = find_continuation_files(video1_path)
    video2_files = find_continuation_files(video2_path)
    print(f"Found {len(video1_files)} files for left camera: {video1_files}")
    print(f"Found {len(video2_files)} files for right camera: {video2_files}")

    # Step 2: Extract audio from both files starting at 'start_seconds'
    audio1, eff_fps = extract_audio_signal(video1_files[0], downsample_factor, start_seconds, match_duration)
    audio2, _       = extract_audio_signal(video2_files[0], downsample_factor, start_seconds, match_duration)

    match_samples = int(match_duration * eff_fps)
    audio1 = audio1[:match_samples]
    audio2 = audio2[:match_samples]

    # Step 3: Cross-correlate to find lag (audio2 relative to audio1)
    correlation = np.correlate(audio1 - np.mean(audio1), audio2 - np.mean(audio2), mode='full')
    lag_samples = int(np.argmax(correlation) - (len(audio2) - 1))
    lag_seconds = lag_samples / eff_fps
    print(f"Matched lag: {lag_samples} samples → {lag_seconds:.3f} s")

    # Optional plot
    if plot:
        time = np.linspace(0, len(audio1) / eff_fps, len(audio1))
        plt.figure(figsize=(15, 6))
        plt.plot(time, audio1, label='Video 1')
        if lag_samples > 0:
            shifted = np.pad(audio2, (lag_samples, 0), mode='constant')[:len(audio1)]
        else:
            shifted = audio2[-lag_samples:]
            pad_len = len(audio1) - len(shifted)
            shifted = np.pad(shifted, (0, max(0, pad_len)), mode='constant')[:len(audio1)]
        plt.plot(time, shifted, label='Video 2 (shifted)')
        plt.title('Matched Audio Signals')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Step 4.5: Try to find start-of-flash frame
    flash_video_path = video1_files[0]  # change if the flash is in video2
    flash_start_frame = detect_flash_start_frame(flash_video_path)
    use_flash_alignment = flash_start_frame is not None

    # Step 5: Get timestamps
    timestamps1, fps1 = extract_video_frame_timestamps(video1_files[0])
    timestamps2, fps2 = extract_video_frame_timestamps(video2_files[0])

    if use_flash_alignment:
        cap_tmp = cv2.VideoCapture(flash_video_path)
        flash_video_fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        flash_start_time = flash_start_frame / max(flash_video_fps, 1e-9)
        print(f"Start of flash at frame {flash_start_frame} (≈ {flash_start_time:.3f} s) in flash video")

        # Align both streams to flash start; correct for measured audio lag
        if flash_video_path == video1_files[0]:
            # flash in video 1 → t_f2 ≈ t_f1 - lag
            timestamps1 = timestamps1 - flash_start_time
            timestamps2 = timestamps2 - (flash_start_time - lag_seconds)
        else:
            # flash in video 2 → t_f1 ≈ t_f2 + lag
            timestamps2 = timestamps2 - flash_start_time
            timestamps1 = timestamps1 - (flash_start_time + lag_seconds)
    else:
        # No flash detected → use audio-only alignment (audio2 lags audio1 by +lag_seconds)
        print("No flash detected — falling back to audio-only alignment.")
        timestamps1 = timestamps1.copy()
        timestamps2 = timestamps2 - lag_seconds

    # Step 6: Frame matching (nearest-neighbour within 1 / max fps)
    matched_pairs = []
    tol = 1.0 / max(fps1, fps2) if max(fps1, fps2) > 0 else 1/60.0
    for i, t1 in enumerate(timestamps1):
        idx2 = int(np.argmin(np.abs(timestamps2 - t1)))
        if abs(timestamps2[idx2] - t1) < tol:
            matched_pairs.append((i, idx2))

    # Step 7: Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_Video1', 'Frame_Video2'])
        writer.writerows(matched_pairs)

    print(f"Saved {len(matched_pairs)} matched frame pairs to: {output_file}")
    return output_file

