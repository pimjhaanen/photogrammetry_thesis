import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Parameters
fps_audio = 48000


def find_continuation_file(filepath):
    dirname, filename = os.path.split(filepath)
    if filename.startswith("GX01"):
        new_filename = filename.replace("GX01", "GX02", 1)
        candidate_path = os.path.join(dirname, new_filename)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def extract_audio_with_ffmpeg(video_path, wav_output):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(fps_audio), "-f", "wav", wav_output
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio_data_ffmpeg(video_path, downsample_factor=50):
    wav_path = video_path.replace(".mp4", ".wav")

    continuation = find_continuation_file(video_path)
    if continuation:
        concat_list_path = "temp_concat_list.txt"
        with open(concat_list_path, "w") as f:
            f.write(f"file '{os.path.abspath(video_path)}'\n")
            f.write(f"file '{os.path.abspath(continuation)}'\n")
        combined_path = video_path.replace(".mp4", "_combined.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-vn", "-ac", "1", "-ar", str(fps_audio), "-f", "wav", combined_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wav_path = combined_path

    else:
        extract_audio_with_ffmpeg(video_path, wav_path)

    rate, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    audio = audio[::downsample_factor]
    effective_fps = rate / downsample_factor
    audio = audio[int(10 * effective_fps):]  # Skip 10 seconds
    time = np.linspace(0, len(audio) / effective_fps, len(audio))
    return time, audio, effective_fps


def synchronise_audio_signals(signal1, signal2, eff_fps, chunk_duration=120, fps_video=30, chunked=True):
    if not chunked:
        correlation = np.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
        lag = np.argmax(correlation) - (len(signal2) - 1)
        drift_seconds = lag / eff_fps
        drift_frames = drift_seconds * fps_video
        print(f"Global lag = {lag} samples → {drift_seconds:.3f} s → {drift_frames:.4f} frames")

        if lag >= 0:
            signal2_aligned = np.pad(signal2, (lag, 0), mode='constant')[:len(signal1)]
            signal1_aligned = signal1[:len(signal2_aligned)]
        else:
            signal2_aligned = signal2[-lag:]
            signal2_aligned = np.pad(signal2_aligned, (0, len(signal1) - len(signal2_aligned)), mode='constant')
            signal1_aligned = signal1[:len(signal2_aligned)]

        min_len = min(len(signal1_aligned), len(signal2_aligned))
        return signal1_aligned[:min_len], signal2_aligned[:min_len], [0], [abs(drift_frames)]

    chunk_samples = int(chunk_duration * eff_fps)
    aligned_signal1 = []
    aligned_signal2 = []
    drift_values = []
    chunk_times = []
    reference_lag = None
    start = 0
    chunk_idx = 0

    while start < len(signal1):
        current_chunk_duration = chunk_duration if chunk_idx > 0 else 10
        chunk_samples_current = int(current_chunk_duration * eff_fps)

        chunk1 = signal1[start:start + chunk_samples_current]
        chunk2 = signal2[start:start + chunk_samples_current + 1000]

        if len(chunk1) < int(0.1 * chunk_samples_current):
            break

        offset = int(5 * eff_fps)
        if len(chunk1) <= offset or len(chunk2) <= offset:
            break

        chunk1_corr = chunk1[offset:]
        chunk2_corr = chunk2[offset:len(chunk1) + offset]

        correlation = np.correlate(chunk1_corr - np.mean(chunk1_corr), chunk2_corr - np.mean(chunk2_corr), mode='full')
        lag = np.argmax(correlation) - (len(chunk2_corr) - 1)

        if chunk_idx == 1:
            reference_lag = lag

        delta_frames = abs((lag - reference_lag) / eff_fps * fps_video) if reference_lag is not None else 0
        print(f"Chunk {chunk_idx}: lag = {lag} samples → {lag / eff_fps:.3f} s → {delta_frames:.4f} frame drift vs. reference")

        chunk_times.append(start / eff_fps)
        drift_values.append(delta_frames)

        chunk1_final = chunk1
        chunk2_start = start - lag if lag >= 0 else start + abs(lag)
        chunk2_end = chunk2_start + chunk_samples_current

        if chunk2_start < 0:
            pad_left = abs(chunk2_start)
            chunk2_extract = signal2[0:chunk2_end]
            chunk2_final = np.pad(chunk2_extract, (pad_left, 0), mode='constant')[:chunk_samples_current]
        elif chunk2_end > len(signal2):
            chunk2_extract = signal2[chunk2_start:]
            pad_right = chunk_samples_current - len(chunk2_extract)
            chunk2_final = np.pad(chunk2_extract, (0, pad_right), mode='constant')
        else:
            chunk2_final = signal2[chunk2_start:chunk2_end]

        aligned_signal1.extend(chunk1_final)
        aligned_signal2.extend(chunk2_final)
        start += chunk_samples_current
        chunk_idx += 1

    min_len = min(len(aligned_signal1), len(aligned_signal2))
    aligned_signal1 = np.array(aligned_signal1[:min_len])
    aligned_signal2 = np.array(aligned_signal2[:min_len])

    return aligned_signal1, aligned_signal2, chunk_times, drift_values


# Load and process
video_paths = ['main simulation/left_videos/25_06_test2_merged.mp4', 'main simulation/right_videos/25_06_test2_merged.mp4']
audio_data = []
for path in video_paths:
    time, audio_mono, eff_fps = extract_audio_data_ffmpeg(path)
    audio_data.append((time, audio_mono))

signal1 = audio_data[0][1]
signal2 = audio_data[1][1]
aligned_signal1, aligned_signal2, chunk_times, drift_values = synchronise_audio_signals(
    signal1, signal2, eff_fps, chunk_duration=50, fps_video=30, chunked=True)

# Plot audio
time_axis = np.linspace(0, len(aligned_signal1) / eff_fps, len(aligned_signal1))
plt.figure(figsize=(15, 8))
plt.plot(time_axis, aligned_signal1, label='Camera 1', color='blue', alpha=0.7)
plt.plot(time_axis, aligned_signal2, label='Camera 2', color='orange', alpha=0.7)
plt.title('Synchronised Audio Signals of Both Videos')
plt.xlabel('Time (s)')
plt.ylabel('Sound Level (normalised)')
plt.legend()
plt.grid(True)
plt.show()

# Plot drift
plt.figure(figsize=(10, 5))
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left y-axis: drift in frames
ax1.plot(chunk_times, drift_values, marker='o', color='tab:blue')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Drift (frames)')
ax1.tick_params(axis='y')
ax1.set_title('Drift vs. (Aligned) starting point at 30fps')
plt.grid(True)

# Right y-axis: drift in seconds
ax2 = ax1.twinx()
drift_seconds = [v / 30 for v in drift_values]
ax2.plot(chunk_times, drift_seconds, marker='o', color='tab:blue')
ax2.set_ylabel('Drift (seconds)')
ax2.tick_params(axis='y')

plt.tight_layout()
plt.show()
