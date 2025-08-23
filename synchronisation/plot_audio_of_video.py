import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def extract_audio_wav(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", "44100",  # Standard audio sample rate
        "-loglevel", "error",
        wav_path
    ]
    subprocess.run(command, check=True)
    sample_rate, audio = wavfile.read(wav_path)
    return sample_rate, audio.astype(np.float32), wav_path

from scipy.signal import butter, filtfilt

def bandpass_filter(audio, sr, lowcut=900, highcut=1100, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)

def plot_audio_analysis_filtered(video_path, trim_seconds=10):
    sr, audio, _ = extract_audio_wav(video_path)

    # Optional: Trim first few seconds to reduce noise
    if trim_seconds > 0:
        audio = audio[int(trim_seconds * sr):]

    # === Filter around 1000 Hz ===
    filtered = bandpass_filter(audio, sr, lowcut=700, highcut=900)

    time = np.linspace(0, len(filtered) / sr, len(filtered))

    # === Plot filtered waveform ===
    plt.figure(figsize=(14, 5))
    plt.plot(time, filtered, color='mediumvioletred')
    plt.title('Filtered Audio Waveform (~1000 Hz Band)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot filtered spectrogram ===
    plt.figure(figsize=(14, 6))
    plt.specgram(filtered, Fs=sr, NFFT=2048, noverlap=1024, cmap='plasma')
    plt.title('Filtered Audio Spectrogram (~1000 Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()

    print("Sample Rate:", sr)
    print("Duration:", round(len(filtered)/sr, 2), "seconds")

# === Run it ===
plot_audio_analysis_filtered(r"C:\Users\pimha\PycharmProjects\photogrammetry_thesis\main simulation\right_videos\25_06_test1.mp4")
