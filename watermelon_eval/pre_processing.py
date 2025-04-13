import json
import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from settings import *


# Step 1: Pre-emphasis
def pre_emphasis(signal, alpha=0.97):
    emphasized = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    if VERBOSE:
        plt.figure(figsize=(10, 3))
        plt.plot(signal, label='Original', alpha=0.6)
        plt.plot(emphasized, label='Pre-emphasized', alpha=0.6)
        plt.title("Step 1: Pre-emphasis")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return emphasized


# Step 2: Framing
def frame_signal(signal, frame_size, frame_step, sample_rate):
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_step * sample_rate)
    signal_length = len(signal)

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Visualize a few frames
    if VERBOSE:
        plt.figure(figsize=(10, 3))
        for i in range(3):
            plt.plot(frames[i], label=f'Frame {i + 1}')
        plt.title("Step 2: Framing (First 3 Frames)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return frames


# Step 3: Hamming Window
def apply_hamming_window(frames):
    frame_length = frames.shape[1]
    hamming = np.hamming(frame_length)
    windowed_frames = frames * hamming

    # Visualize window effect on first frame
    if VERBOSE:
        plt.figure(figsize=(10, 3))
        plt.plot(frames[0], label='Original Frame')
        plt.plot(windowed_frames[0], label='Windowed Frame')
        plt.title("Step 3: Hamming Window (Frame 1)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return windowed_frames


# Step 4–5: Mel Spectrogram
def extract_mel_spectrogram(signal, sr, n_fft=512, frame_size=0.025, frame_step=0.010, n_mels=64): #TODO check n_fft. n_mel was 40
    emphasized = pre_emphasis(signal)
    frames = frame_signal(emphasized, frame_size, frame_step, sr)
    windowed_frames = apply_hamming_window(frames)

    # Compute power spectrum
    mag_frames = np.absolute(np.fft.rfft(windowed_frames, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # Visualize raw spectrogram
    if VERBOSE:
        plt.figure(figsize=(10, 4))
        plt.imshow(10 * np.log10(pow_frames.T + 1e-10), aspect='auto', origin='lower', cmap='viridis')
        plt.title('Step 4: Power Spectrogram (Log Scale)')
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency Bin')
        plt.colorbar(label='Power (dB)')
        plt.tight_layout()
        plt.show()

    # Mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spectrogram = np.dot(pow_frames, mel_basis.T)

    # Convert to dB for better visualization
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)

    # Final Mel Spectrogram
    if VERBOSE:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=int(frame_step * sr),
                                 x_axis='time', y_axis='mel')
        plt.title('Step 5: Mel Spectrogram (dB)')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    return mel_spectrogram_db


# audio_path = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\datasets\1_10.5\chu\1\1.wav"
# signal, sr = librosa.load(audio_path, sr=16000)
#
# #run
# mel_spec_db = extract_mel_spectrogram(signal, sr)



# -------------------> creating the json data file
# Load raw label data
with open("../watermelon_dataset/ripeness_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Output directory for spectrogram .npy files
out_dir = "../watermelon_dataset/processed_spectrograms"
os.makedirs(out_dir, exist_ok=True)

for entry in data:
    wav_path = entry["audio_path"]
    signal, sr = librosa.load(wav_path, sr=16000)

    mel_spec = extract_mel_spectrogram(signal, sr)
    spec_name = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"

    spec_path = os.path.join(out_dir, entry["ripeness_label"])
    os.makedirs(spec_path, exist_ok=True)

    full_spec_path = os.path.join(spec_path, spec_name)
    np.save(full_spec_path, mel_spec)

    entry["spectrogram_path"] = full_spec_path

# Save updated entries
with open(os.path.join(out_dir, "ripeness_with_specs.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)





# -------------------> creating the json data file
# Load raw label data
with open("../watermelon_dataset/ripeness_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Output directory for spectrogram .npy files
out_dir = os.path.join(os.path.dirname(__file__), "..", "watermelon_dataset", "processed_spectrograms")
os.makedirs(out_dir, exist_ok=True)

for entry in data:
    wav_path = entry["audio_path"]
    signal, sr = librosa.load(wav_path, sr=16000)

    mel_spec = extract_mel_spectrogram(signal, sr)
    spec_name = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"

    spec_path = os.path.join(out_dir, entry["ripeness_label"])
    os.makedirs(spec_path, exist_ok=True)

    full_spec_path = os.path.join(spec_path, spec_name)
    np.save(full_spec_path, mel_spec)

    entry["spectrogram_path"] = full_spec_path

# Save updated entries
with open(os.path.join(out_dir, "ripeness_with_specs.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)