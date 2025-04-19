# mel_utils.py

import numpy as np
import librosa

# ——— constants ———
SR          = 16000           # Sampling rate (Hz)
WIN_LENGTH  = 1024            # Window size in samples (64 ms)
HOP_LENGTH  = 320             # Hop size in samples    (20 ms)
N_MELS      = 64              # Number of mel filterbank channels
FMIN        = 50              # Lowest mel frequency (Hz)
FMAX        = SR // 2         # Highest mel frequency = Nyquist (8000 Hz)
ALPHA       = 0.97            # Pre‑emphasis coefficient

def pre_emphasis(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Apply a pre‑emphasis filter on the waveform to boost high frequencies.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def frame_signal(signal: np.ndarray,
                 win_length: int = WIN_LENGTH,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Slice a 1D signal into overlapping frames.
    Returns an array of shape (n_frames, win_length).
    """
    sig_len    = len(signal)
    num_frames = 1 + int(np.ceil((sig_len - win_length) / hop_length))
    pad_len    = max(0, num_frames * hop_length + win_length - sig_len)
    padded     = np.concatenate([signal, np.zeros(pad_len)], axis=0)

    # Build indices for framing
    indices = (
        np.tile(np.arange(win_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_length, hop_length),
                  (win_length, 1)).T
    )
    return padded[indices.astype(np.int32)]

def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    Apply a Hamming window to each frame.
    """
    return frames * np.hamming(frames.shape[1])[None, :]

def extract_mel_spectrogram(
    signal: np.ndarray,
    sr: int = SR
) -> np.ndarray:
    """
    Compute a log‑scaled Mel spectrogram with:
      1) Pre‑emphasis
      2) Framing (1024-sample windows, 320-sample hops)
      3) Hamming window
      4) Power spectrum
      5) Mel filter bank (50–Nyquist, 64 bands)
      6) Convert to decibels

    Returns:
      mel_db: np.ndarray of shape (n_mels, n_frames)
    """
    # 1) Pre‑emphasis
    emphasized = pre_emphasis(signal, ALPHA)

    # 2) Framing + windowing
    frames     = frame_signal(emphasized, WIN_LENGTH, HOP_LENGTH)
    win_frames = apply_hamming(frames)

    # 3) Power spectrum
    mag       = np.abs(np.fft.rfft(win_frames, n=WIN_LENGTH))
    pow_spec  = (1.0 / WIN_LENGTH) * (mag ** 2)

    # 4) Mel filter bank (clamp fmax to Nyquist)
    effective_fmax = min(FMAX, sr // 2)
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=effective_fmax
    )
    mel_spec = np.dot(pow_spec, mel_fb.T)  # shape: (n_frames, n_mels)

    # 5) Convert to decibel scale
    mel_db = librosa.power_to_db(mel_spec.T, ref=np.max)  # (n_mels, n_frames)
    return mel_db
