import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display
import argparse
import sys

from ECAPA_TDNN_Full import *


# ==== CONFIG ====
SAMPLE_RATE = 16000
N_FFT = 512
FRAME_SIZE = 0.025
FRAME_STEP = 0.010
N_MELS = 64
CLASS_NAMES = ['low_sweet', 'sweet', 'un_sweet', 'very_sweet']
MODEL_PATH = "ecapa_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== STEP 1: MEL SPECTROGRAM GENERATOR ====
def extract_mel_spectrogram(signal, sr, n_fft=512, frame_size=0.025, frame_step=0.010, n_mels=64):
    # === Step 1: Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # === Step 2: Framing
    frame_len = int(frame_size * sr)
    frame_step = int(frame_step * sr)
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step)) + 1
    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.append(emphasized, np.zeros((pad_len - signal_length)))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32)]

    # === Step 3: Hamming window
    hamming = np.hamming(frame_len)
    windowed_frames = frames * hamming

    # === Step 4: Power Spectrum
    mag_frames = np.abs(np.fft.rfft(windowed_frames, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # === Step 5: Mel Filter Bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spectrogram = np.dot(pow_frames, mel_basis.T)

    # === Step 6: Convert to dB (and then transpose)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)

    return mel_spectrogram_db  # Shape: [n_mels, time]



def plot_mel_spectrogram(mel_spec_db, sr, hop_length):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

# ==== STEP 2: LOAD MODEL ====
def load_model():
    model = ECAPA_TDNN_Full(input_dim=64, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# ==== STEP 3: PREDICT ====
def predict_from_wav(wav_path):
    signal, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel_spec = extract_mel_spectrogram(signal, sr)
    plot_mel_spectrogram(mel_spec, sr, hop_length=int(FRAME_STEP * sr))

    # Shape -> [1, 1, mel_bins, time]
    input_tensor = torch.tensor(mel_spec[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        label = CLASS_NAMES[top_class.item()]
        confidence = top_prob.item()

    print(f"\n🎯 Predicted class: **{label}** ({confidence*100:.2f}%)\n")
    return label, confidence

# ==== MAIN ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to .wav file")
    args = parser.parse_args()

    assert os.path.exists(args.wav), "❌ WAV file not found!"
    predict_from_wav(args.wav)
