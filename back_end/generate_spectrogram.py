import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ==== CONFIG ====
SAMPLE_RATE = 16000
N_FFT = 512
FRAME_SIZE = 0.025
FRAME_STEP = 0.010
N_MELS = 64

# ==== MEL SPECTROGRAM EXTRACTOR ====
def extract_mel_spectrogram(signal, sr):
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    frame_len = int(FRAME_SIZE * sr)
    frame_step = int(FRAME_STEP * sr)
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step)) + 1
    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.append(emphasized, np.zeros((pad_len - signal_length)))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32)]

    hamming = np.hamming(frame_len)
    windowed = frames * hamming

    mag_frames = np.abs(np.fft.rfft(windowed, n=N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    mel_spectrogram = np.dot(pow_frames, mel_basis.T)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)

    return mel_spectrogram_db

# ==== VISUALIZATION ====
def plot_mel_spectrogram(mel_spec_db, sr, hop_length, output_path=None):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (dB)')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"🖼️  Saved spectrogram image to: {output_path}")
    plt.show()

# ==== MAIN FUNCTION ====
def convert_wav_to_npy(wav_path, out_dir, show_plot=True):
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}.npy")
    plot_path = os.path.join(out_dir, f"{base_name}.png")

    signal, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel_spec = extract_mel_spectrogram(signal, sr)

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, mel_spec)
    print(f"✅ Saved spectrogram to: {out_path}")

    if show_plot:
        plot_mel_spectrogram(mel_spec, sr=sr, hop_length=int(FRAME_STEP * sr), output_path=plot_path)

    return out_path

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to .wav file")
    parser.add_argument("--out", type=str, default="./temp_specs", help="Directory to save .npy")
    parser.add_argument("--no_plot", action="store_true", help="Disable spectrogram plot")
    args = parser.parse_args()

    convert_wav_to_npy(args.wav, args.out, show_plot=not args.no_plot)
