"""This code can be used to measure audio-sync drift between two GoPro videos.
It extracts mono audio via FFmpeg, (optionally) concatenates the second “GX02…”
continuation file, downsamples, then aligns the two audio streams using
cross-correlation—either globally or in chunks—to estimate time/base-frames drift.

You can adapt the audio sample rate, downsample factor, skip seconds,
chunk length, and plotting options to your dataset.

Note that the videos should have little background noise to test the drift, and some constant
sounds every once in a while (beeps, claps, etc.). For noisy flight data, matching throughout
flight is inaccurate, and therefore results in weird drift results.
"""

import os
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Default audio extraction rate (GoPro audio is commonly 48 kHz)
FPS_AUDIO_DEFAULT = 48_000


# ---------------------------------------------------------------------------
# File / FFmpeg helpers
# ---------------------------------------------------------------------------

def find_continuation_file(filepath: str) -> Optional[str]:
    """RELEVANT FUNCTION INPUTS:
    - filepath: path to the first video file (e.g., '.../GX01xxxx.mp4').

    RETURNS:
    - Absolute path to the next continuation file (GX02...) if it exists in the same folder,
      otherwise None.

    Notes:
    - This follows your original logic: GX01 → GX02 only.
    """
    dirname, filename = os.path.split(filepath)
    if filename.startswith("GX01"):
        new_filename = filename.replace("GX01", "GX02", 1)
        candidate = os.path.join(dirname, new_filename)
        if os.path.exists(candidate):
            return candidate
    return None


def extract_audio_with_ffmpeg(video_path: str, wav_output: str, fps_audio: int = FPS_AUDIO_DEFAULT) -> None:
    """RELEVANT FUNCTION INPUTS:
    - video_path: input MP4 file.
    - wav_output: output WAV path (mono).
    - fps_audio: target WAV sample rate (Hz), default 48 kHz.

    SIDE EFFECTS:
    - Writes a mono .wav file at the given sample rate using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(fps_audio), "-f", "wav", wav_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def extract_audio_data_ffmpeg(
    video_path: str,
    downsample_factor: int = 50,
    skip_seconds: float = 10.0,
    fps_audio: int = FPS_AUDIO_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """RELEVANT FUNCTION INPUTS:
    - video_path: input MP4 (GX01… preferred if you expect a GX02 continuation).
    - downsample_factor: decimation factor applied to the extracted audio (int).
    - skip_seconds: initial seconds to drop (e.g., to skip startup noise).
    - fps_audio: WAV sample rate used for extraction (Hz).

    RETURNS:
    - time: time axis (seconds) after downsampling and skipping.
    - audio: mono float32 waveform after downsampling and skipping.
    - effective_fps: sample rate after decimation (= fps_audio / downsample_factor).

    Notes:
    - If a GX02 continuation is present in the same directory, both files are concatenated
      before extraction (exactly as your original code did).
    """
    wav_path = video_path.replace(".mp4", ".wav")

    continuation = find_continuation_file(video_path)
    if continuation:
        concat_list_path = "temp_concat_list.txt"
        with open(concat_list_path, "w") as f:
            f.write(f"file '{os.path.abspath(video_path)}'\n")
            f.write(f"file '{os.path.abspath(continuation)}'\n")
        combined_path = video_path.replace(".mp4", "_combined.wav")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-vn", "-ac", "1", "-ar", str(fps_audio), "-f", "wav", combined_path
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        wav_path = combined_path
    else:
        extract_audio_with_ffmpeg(video_path, wav_path, fps_audio=fps_audio)

    rate, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)

    # Downsample by simple decimation (same behavior as your snippet)
    audio = audio[::downsample_factor]
    effective_fps = rate / downsample_factor

    # Skip initial seconds (as in your code)
    start_idx = int(skip_seconds * effective_fps)
    audio = audio[start_idx:]

    time = np.linspace(0.0, len(audio) / effective_fps, len(audio), endpoint=False)
    return time, audio, effective_fps


# ---------------------------------------------------------------------------
# Core sync / drift computation
# ---------------------------------------------------------------------------

def synchronise_audio_signals(
    signal1: np.ndarray,
    signal2: np.ndarray,
    eff_fps: float,
    chunk_duration: float = 120.0,
    fps_video: float = 30.0,
    chunked: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """RELEVANT FUNCTION INPUTS:
    - signal1, signal2: mono waveforms (float32) sampled at eff_fps (after downsampling).
    - eff_fps: effective audio sample rate (Hz) of the provided signals.
    - chunk_duration: seconds per chunk for local drift estimation (used if chunked=True).
    - fps_video: nominal video FPS to convert time drift → frame drift.
    - chunked: if True, estimate drift per chunk vs. a reference; else do one global alignment.

    RETURNS:
    - aligned_signal1: signal1 cut/zero-padded to match aligned_signal2 length (global or chunk-wise).
    - aligned_signal2: signal2 cut/shifted/zero-padded to match aligned_signal1 length.
    - chunk_times: list of chunk start times (seconds) relative to start of aligned signals.
    - drift_values: list of drift magnitudes (frames) relative to the reference lag (chunked=True),
                    or a single-element list [abs(global_drift_in_frames)] if chunked=False.

    Notes:
    - This reproduces your original algorithm (mean-subtracted cross-correlation; padding as needed).
    - For chunked=True:
        * The first chunk uses 10 s, subsequent chunks use chunk_duration.
        * The “reference lag” is captured at chunk_idx == 1, then drift is measured as
          the lag difference vs. that reference, converted to frames via eff_fps & fps_video.
    """
    if not chunked:
        # Global single-shot alignment
        s1 = signal1 - np.mean(signal1)
        s2 = signal2 - np.mean(signal2)
        corr = np.correlate(s1, s2, mode="full")
        lag = int(np.argmax(corr)) - (len(s2) - 1)

        drift_seconds = lag / eff_fps
        drift_frames = drift_seconds * fps_video
        print(f"Global lag = {lag} samples → {drift_seconds:.3f} s → {drift_frames:.4f} frames")

        if lag >= 0:
            s2_aligned = np.pad(signal2, (lag, 0), mode="constant")[:len(signal1)]
            s1_aligned = signal1[:len(s2_aligned)]
        else:
            s2_aligned = signal2[-lag:]
            s2_aligned = np.pad(s2_aligned, (0, len(signal1) - len(s2_aligned)), mode="constant")
            s1_aligned = signal1[:len(s2_aligned)]

        m = min(len(s1_aligned), len(s2_aligned))
        return s1_aligned[:m], s2_aligned[:m], [0.0], [abs(drift_frames)]

    # Chunked mode (as in your code)
    chunk_samples = int(chunk_duration * eff_fps)
    aligned_signal1: List[float] = []
    aligned_signal2: List[float] = []
    drift_values: List[float] = []
    chunk_times: List[float] = []

    reference_lag: Optional[int] = None
    start = 0
    chunk_idx = 0

    while start < len(signal1):
        # First chunk uses 10 seconds, like your snippet
        current_chunk_duration = chunk_duration if chunk_idx > 0 else 10.0
        chunk_len = int(current_chunk_duration * eff_fps)

        chunk1 = signal1[start:start + chunk_len]
        # Allow some look-ahead for chunk2 to correlate
        chunk2 = signal2[start:start + chunk_len + 1000]

        if len(chunk1) < int(0.1 * chunk_len):
            break

        offset = int(5 * eff_fps)  # ignore first 5 s within each chunk, per your code
        if len(chunk1) <= offset or len(chunk2) <= offset:
            break

        c1 = chunk1[offset:]
        c2 = chunk2[offset:len(chunk1) + offset]

        # Mean-subtracted cross-correlation
        corr = np.correlate(c1 - np.mean(c1), c2 - np.mean(c2), mode="full")
        lag = int(np.argmax(corr)) - (len(c2) - 1)

        if chunk_idx == 1:
            reference_lag = lag

        if reference_lag is not None:
            delta_frames = abs((lag - reference_lag) / eff_fps * fps_video)
        else:
            delta_frames = 0.0

        print(f"Chunk {chunk_idx}: lag = {lag} samples → {lag / eff_fps:.3f} s → {delta_frames:.4f} frame drift vs. reference")

        chunk_times.append(start / eff_fps)
        drift_values.append(delta_frames)

        # Build aligned slices (pad where needed) – same logic you had
        chunk1_final = chunk1
        if lag >= 0:
            chunk2_start = max(0, start - lag)
        else:
            chunk2_start = start + abs(lag)
        chunk2_end = chunk2_start + chunk_len

        if chunk2_start < 0:
            pad_left = abs(chunk2_start)
            chunk2_extract = signal2[0:chunk2_end]
            chunk2_final = np.pad(chunk2_extract, (pad_left, 0), mode="constant")[:chunk_len]
        elif chunk2_end > len(signal2):
            chunk2_extract = signal2[chunk2_start:]
            pad_right = chunk_len - len(chunk2_extract)
            chunk2_final = np.pad(chunk2_extract, (0, pad_right), mode="constant")
        else:
            chunk2_final = signal2[chunk2_start:chunk2_end]

        aligned_signal1.extend(chunk1_final)
        aligned_signal2.extend(chunk2_final)

        start += chunk_len
        chunk_idx += 1

    # Finalize aligned arrays to equal length
    m = min(len(aligned_signal1), len(aligned_signal2))
    return np.asarray(aligned_signal1[:m]), np.asarray(aligned_signal2[:m]), chunk_times, drift_values


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_aligned_audio(
    aligned_signal1: np.ndarray,
    aligned_signal2: np.ndarray,
    eff_fps: float,
    labels: Tuple[str, str] = ("Camera 1", "Camera 2"),
    title: str = "Synchronised Audio Signals of Both Videos",
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - aligned_signal1, aligned_signal2: equal-length arrays after alignment.
    - eff_fps: sample rate (Hz) of the aligned signals.
    - labels: legend labels for the two signals.
    - title: figure title.

    SIDE EFFECTS:
    - Shows a matplotlib figure with both waveforms over a shared time axis.
    """
    t = np.linspace(0, len(aligned_signal1) / eff_fps, len(aligned_signal1), endpoint=False)
    plt.figure(figsize=(15, 6))
    plt.plot(t, aligned_signal1, label=labels[0], alpha=0.7)
    plt.plot(t, aligned_signal2, label=labels[1], alpha=0.7)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Sound Level (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drift(
    chunk_times: List[float],
    drift_values: List[float],
    fps_video: float = 30.0,
    title: str = "Drift vs. (Aligned) starting point at 30 fps",
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - chunk_times: list of start times (s) for each chunk.
    - drift_values: drift magnitudes in *frames* (relative to reference).
    - fps_video: FPS used to convert frames ↔ seconds (for secondary axis label).
    - title: plot title.

    SIDE EFFECTS:
    - Shows a matplotlib figure with drift in frames (left axis) and seconds (right axis).
    """
    if not chunk_times or not drift_values:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(chunk_times, drift_values, marker='o')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Drift (frames)")
    ax1.set_title(title)
    ax1.grid(True)

    ax2 = ax1.twinx()
    drift_seconds = [v / fps_video for v in drift_values]
    ax2.plot(chunk_times, drift_seconds, marker='o')
    ax2.set_ylabel("Drift (seconds)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def gopro_drift_test(
    left_video: str,
    right_video: str,
    downsample_factor: int = 50,
    skip_seconds: float = 10.0,
    chunk_duration: float = 50.0,
    fps_video: float = 30.0,
    chunked: bool = True,
    show_plots: bool = True,
) -> None:
    """RELEVANT FUNCTION INPUTS:
    - left_video, right_video: paths to the two GoPro MP4s to compare.
    - downsample_factor: audio decimation factor after extraction (speeds up correlation).
    - skip_seconds: seconds to discard from start of audio (startup noise).
    - chunk_duration: chunk size in seconds for local drift estimation (chunked mode).
    - fps_video: nominal video FPS for reporting drift in frames.
    - chunked: if True, per-chunk drift vs. reference; else one global alignment.
    - show_plots: whether to show the aligned audio and drift figures.

    SIDE EFFECTS:
    - Prints drift information and shows plots if requested.
    """
    # Load/prepare audio for each camera
    t1, a1, eff_fps = extract_audio_data_ffmpeg(
        left_video, downsample_factor=downsample_factor, skip_seconds=skip_seconds
    )
    t2, a2, eff_fps2 = extract_audio_data_ffmpeg(
        right_video, downsample_factor=downsample_factor, skip_seconds=skip_seconds
    )
    # Sanity: both effective rates should match (same downsample factor)
    if not np.isclose(eff_fps, eff_fps2):
        print(f"[WARN] Effective sample rates differ: {eff_fps} vs {eff_fps2}. Using {eff_fps}.")

    # Align and compute drift
    s1_al, s2_al, times, drift = synchronise_audio_signals(
        a1, a2, eff_fps, chunk_duration=chunk_duration, fps_video=fps_video, chunked=chunked
    )

    # Report quick summary
    if drift:
        if chunked:
            print(f"[INFO] Mean drift over chunks: {np.mean(drift):.4f} frames "
                  f"(±{np.std(drift):.4f}), max={np.max(drift):.4f}")
        else:
            print(f"[INFO] Global drift: {drift[0]:.4f} frames ({drift[0]/fps_video:.4f} s)")

    # Plots
    if show_plots:
        plot_aligned_audio(s1_al, s2_al, eff_fps, labels=("Camera 1", "Camera 2"))
        if chunked:
            plot_drift(times, drift, fps_video=fps_video)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example paths (same structure you shared)
    LEFT_VIDEO = "../input/left_videos/25_06_test2_merged.mp4"
    RIGHT_VIDEO = "../input/right_videos/25_06_test2_merged.mp4"

    # Run the drift test with your preferred parameters
    gopro_drift_test(
        left_video=LEFT_VIDEO,
        right_video=RIGHT_VIDEO,
        downsample_factor=50,   # same as your snippet
        skip_seconds=10.0,      # same as your snippet
        chunk_duration=150.0,    # you used 50 s in your example call
        fps_video=30.0,
        chunked=True,
        show_plots=True,
    )
