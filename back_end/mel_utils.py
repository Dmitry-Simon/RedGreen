# mel_utils.py
import numpy as np
import librosa

# ——— constants ———
SR        = 16000
N_FFT     = 512
FRAME_SZ  = 0.025
FRAME_HOP = 0.010
N_MELS    = 64
ALPHA     = 0.97

def pre_emphasis(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def frame_signal(signal: np.ndarray,
                 frame_size: float = FRAME_SZ,
                 frame_step: float = FRAME_HOP,
                 sr: int = SR) -> np.ndarray:
    frame_length = int(frame_size * sr)
    hop_length   = int(frame_step * sr)
    sig_len      = len(signal)
    num_frames   = int(np.ceil(abs(sig_len - frame_length) / hop_length)) + 1
    pad_len      = num_frames * hop_length + frame_length - sig_len
    pad_signal   = np.append(signal, np.zeros(pad_len))
    indices = (
        np.tile(np.arange(frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_length, hop_length),
                  (frame_length, 1)).T
    )
    return pad_signal[indices.astype(np.int32)]

def apply_hamming(frames: np.ndarray) -> np.ndarray:
    return frames * np.hamming(frames.shape[1])

def extract_mel_spectrogram(
    signal: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    frame_size: float = FRAME_SZ,
    frame_step: float = FRAME_HOP,
    n_mels: int = N_MELS
) -> np.ndarray:
    # 1) pre‑emphasis
    emphasized    = pre_emphasis(signal)
    # 2) framing
    frames        = frame_signal(emphasized, frame_size, frame_step, sr)
    # 3) window
    win_frames    = apply_hamming(frames)
    # 4) power spectrum
    mag_frames    = np.abs(np.fft.rfft(win_frames, n=n_fft))
    pow_frames    = (1.0 / n_fft) * (mag_frames ** 2)
    # 5) mel filter bank
    mel_fb        = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec      = np.dot(pow_frames, mel_fb.T)        # shape: (n_frames, n_mels)
    # 6) dB conversion
    return librosa.power_to_db(mel_spec.T, ref=np.max)  # shape: (n_mels, n_frames)
