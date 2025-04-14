import os
import json
import librosa
import numpy as np
from tqdm import tqdm

def extract_mel_spectrogram(signal, sr, n_fft=512, frame_size=0.025, frame_step=0.010, n_mels=64):
    # Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Framing
    frame_len = int(frame_size * sr)
    frame_step = int(frame_step * sr)
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step)) + 1
    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.append(emphasized, np.zeros((pad_len - signal_length)))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32)]

    # Hamming window
    hamming = np.hamming(frame_len)
    windowed = frames * hamming

    # Power spectrum
    mag_frames = np.abs(np.fft.rfft(windowed, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # Mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spectrogram = np.dot(pow_frames, mel_basis.T)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)

    return mel_spectrogram_db

def process_all_to_spectrograms(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Generating Mel Spectrograms"):
        audio_path = entry['audio_path']
        label = entry['ripeness_label']
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        try:
            signal, sr = librosa.load(audio_path, sr=16000)
            mel_spec = extract_mel_spectrogram(signal, sr)

            # Save as .npy
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            out_path = os.path.join(label_dir, f"{base_name}.npy")
            np.save(out_path, mel_spec)

            entry['spectrogram_path'] = out_path
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    # Optionally update the JSON file with paths to saved spectrograms
    with open(os.path.join(output_dir, "ripeness_with_specs.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Usage
json_input = r"..\watermelon_dataset\datasets\ripeness_labels.json"
spectrogram_output_dir = r"..\watermelon_dataset\processed_spectrograms"

process_all_to_spectrograms(json_input, spectrogram_output_dir)
