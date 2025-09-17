"""This code can be used to analyse GoPro (or any) video audio.
It (1) extracts mono audio from a video with FFmpeg, (2) applies a band-pass
filter around a tone/noise band of interest, and (3) plots the filtered waveform
and its spectrogram.

You can adapt the output sample rate, trim duration, band-pass cutoffs, and the
spectrogram settings to your dataset.
"""

import os
import subprocess
import tempfile
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


def extract_audio_wav(video_path: str, sr_out: int = 48_000, keep_temp: bool = False) -> Tuple[int, np.ndarray, str]:
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the input video file (e.g., '.mp4', '.mov').
    - sr_out:     output audio sample rate in Hz for extraction (default 48 kHz).
    - keep_temp:  if True, keep the temporary WAV on disk; else it will be removed.

    RETURNS:
    - sample_rate: integer sample rate of the extracted audio.
    - audio:       mono audio as float32 numpy array.
    - wav_path:    path to the temporary WAV (may be deleted automatically unless keep_temp=True).

    Notes:
    - Uses FFmpeg to extract mono PCM WAV at sr_out.
    - Requires FFmpeg to be available on your PATH.
    """
    # Prepare a temp wav file (do not delete automatically so we can read it first)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",            # drop video
        "-ac", "1",       # mono
        "-ar", str(sr_out),
        "-loglevel", "error",
        wav_path,
    ]
    subprocess.run(cmd, check=True)

    sample_rate, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)

    if not keep_temp:
        try:
            os.remove(wav_path)
        except OSError:
            pass  # best effort clean-up

    return sample_rate, audio, wav_path


def bandpass_filter(audio: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """RELEVANT FUNCTION INPUTS:
    - audio:   mono audio array (float32 recommended).
    - sr:      sample rate in Hz.
    - lowcut:  lower cutoff frequency (Hz).
    - highcut: higher cutoff frequency (Hz).
    - order:   IIR Butterworth filter order (default 4).

    RETURNS:
    - filtered audio (same length as input) using zero-phase filtfilt.
    """
    nyq = 0.5 * sr
    low = max(1e-6, lowcut / nyq)
    high = min(0.999999, highcut / nyq)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid band edges after normalization: low={low}, high={high}.")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, audio)


def plot_audio_analysis_filtered(
    video_path: str,
    trim_seconds: float = 10.0,
    sr_out: int = 48_000,
    lowcut: float = 700.0,
    highcut: float = 900.0,
    order: int = 4,
    nfft: int = 2048,
    noverlap: Optional[int] = 1024,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """RELEVANT FUNCTION INPUTS:
    - video_path: path to the video whose audio you want to analyse.
    - trim_seconds: seconds to discard from the start (reduce startup noise).
    - sr_out: output sample rate for extraction (Hz).
    - lowcut/highcut/order: band-pass settings (Hz; Butterworth order).
    - nfft/noverlap: spectrogram FFT size and overlap (samples).
    - show: whether to display the plots.

    RETURNS:
    - time axis (seconds) for the filtered waveform,
    - filtered waveform (float32),
    - sample rate (Hz) used for the filtered signal.
    """
    sr, audio, _ = extract_audio_wav(video_path, sr_out=sr_out, keep_temp=False)

    # Optional: trim the first seconds to skip handling noise (e.g., mounting sounds)
    if trim_seconds and trim_seconds > 0:
        start = int(trim_seconds * sr)
        audio = audio[start:]

    # Band-pass around the tone/noise band of interest (e.g., ~800 Hz in wind tunnel)
    filtered = bandpass_filter(audio, sr, lowcut=lowcut, highcut=highcut, order=order)

    # Time base
    t = np.linspace(0.0, len(filtered) / sr, num=len(filtered), endpoint=False)

    if show:
        # 1) Filtered waveform
        plt.figure(figsize=(14, 4))
        plt.plot(t, filtered)
        plt.title(f"Filtered Audio Waveform ({int(lowcut)}–{int(highcut)} Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2) Spectrogram of filtered signal
        plt.figure(figsize=(14, 5))
        Pxx, freqs, bins, im = plt.specgram(filtered, Fs=sr, NFFT=nfft, noverlap=noverlap or 0, cmap="plasma")
        plt.title(f"Filtered Audio Spectrogram ({int(lowcut)}–{int(highcut)} Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        cbar = plt.colorbar(im)
        cbar.set_label("Intensity (dB)")
        plt.tight_layout()
        plt.show()

        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {len(filtered)/sr:.2f} s")

    return t, filtered.astype(np.float32), float(sr)


# ------------------------------- module main ---------------------------------
if __name__ == "__main__":
    # Example usage — change the path and (optionally) the band:
    plot_audio_analysis_filtered(
        r"C:\Users\pimha\PycharmProjects\photogrammetry_thesis\main simulation\right_videos\25_06_test1.mp4",
        trim_seconds=10.0,
        sr_out=48_000,
        lowcut=700.0,
        highcut=900.0,
        order=4,
        nfft=2048,
        noverlap=1024,
        show=True,
    )
