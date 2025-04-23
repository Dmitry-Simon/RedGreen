#mel_utils.py
import numpy as np
import librosa

# ——— constants ———
SR          = 16000           # Sampling rate (Hz)
WIN_LENGTH  = 1024            # Window size in samples (64 ms)
HOP_LENGTH  = 320             # Hop size in samples    (20 ms)
N_MELS      = 64              # Number of mel filterbank channels
FMIN        = 50              # Lowest mel frequency (Hz)
FMAX        = SR // 2         # Highest mel frequency = Nyquist (8000 Hz)
ALPHA       = 0.97            # Pre‑emphasis coefficient

def pre_emphasis(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Apply a pre‑emphasis filter on the waveform to boost high frequencies.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])
    # y(t) = x(t) - alpha * x(t-1), where x is the input signal and y is the output signal.
    # The first sample is left unchanged.

def frame_signal(signal: np.ndarray,
                 win_length: int = WIN_LENGTH,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Slice a 1D signal into overlapping frames.
    Returns an array of shape (n_frames, win_length).
    """
    sig_len    = len(signal) # s * t
    num_frames = 1 + int(np.ceil((sig_len - win_length) / hop_length))
    # n = ((s * t)/h) + 1, ceil(…) so that any leftover tail will still get a (padded) frame,
    # for the very first frame at sample 0.
    pad_len    = max(0, num_frames * hop_length + win_length - sig_len)
    # pad_len = max(0, n * h + w - s * t) = max(0, 0 + 1024 - s * t) If the signal is already long enough,
    # pad_len will be 0.
    padded     = np.concatenate([signal, np.zeros(pad_len)], axis=0)
    # pad the signal with zeros at the end to make it long enough for the last frame

    # Build indices for framing
    indices = (
        np.tile(np.arange(win_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * hop_length, hop_length),
                  (win_length, 1)).T
    )
    # build a 2D array of indices for the frames, each row is a frame, each column is a sample in that frame
    return padded[indices.astype(np.int32)] # convert to int32 for indexing and return the frames

def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    Apply a Hamming window to each frame.
    """
    return frames * np.hamming(frames.shape[1])[None, :]
    # return the frames multiplied by a Hamming window of the same length as the frames, with broadcasting

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

    # 3) Power spectrum FFT + magnitude
    mag       = np.abs(np.fft.rfft(win_frames, n=WIN_LENGTH))
    # compute the FFT of each frame and take the absolute value |X(k)|,
    # where X(k) is the FFT of the frame
    pow_spec  = (1.0 / WIN_LENGTH) * (mag ** 2) # compute the power spectrum P(k) = |X(k)|^2 / M,
    # where M is the number of samples in the FFT (1024), and X(k) is the FFT of the frame

    # 4) Mel filter bank (clamp fmax to Nyquist)
    effective_fmax = min(FMAX, sr // 2)
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=effective_fmax
    ) # m = 2595log10(1 + f/700), where f is the frequency in Hz, and m is the mel frequency.
    # The mel filter bank is a matrix of shape (n_mels, n_fft/2 + 1) that maps the FFT bins to mel frequencies.
    mel_spec = np.dot(pow_spec, mel_fb.T)  # shape: (n_frames, n_mels)
    # Dot product mel_spec[i, m] = sum(pow_spec[i, k] * mel_fb[m, k] for k in range(n_fft/2 + 1)),
    # where i is the frame index and m is the mel frequency index. This computes the energy in each mel frequency band for each frame.

    # 5) Convert to decibel scale
    mel_db = librosa.power_to_db(mel_spec.T, ref=np.max)  # (n_mels, n_frames)
    # Convert to dB: mel_db = 10 * log10(mel_spec/max(mel_spec)), where mel_spec is the power spectrum in Watts.
    # The reference value is the maximum value of the mel spectrum, so that the maximum value in dB is 0 dB.
    # This is done to make the values more interpretable and to compress the dynamic range.
    return mel_db
