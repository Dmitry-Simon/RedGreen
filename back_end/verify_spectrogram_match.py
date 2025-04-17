import numpy as np
import librosa
from predict_from_wav import extract_mel_spectrogram  # update this import!

# === Paths ===
npy_path = r"../watermelon_dataset/processed_spectrograms/low_sweet/5.npy"
wav_path = r"../watermelon_dataset/datasets/2_9.7/audios/5.wav"

# === Load training-time spectrogram
mel_train = np.load(npy_path)

# === Generate inference-time spectrogram
signal, sr = librosa.load(wav_path, sr=16000)
mel_live = extract_mel_spectrogram(signal, sr)

# === Compare
print("Train shape:", mel_train.shape)
print("Live shape :", mel_live.shape)

if mel_train.shape == mel_live.shape:
    diff = np.abs(mel_train - mel_live)
    print("Max diff   :", np.max(diff))
    print("Mean diff  :", np.mean(diff))
else:
    print("❌ Shapes don't match! The model might mispredict due to preprocessing mismatch.")
