import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import librosa
from pathlib import Path

from watermelon_eval.misc.settings import *                  # SR, FIXED_WIDTH, etc.
from back_end.mel_utils import extract_mel_spectrogram, SR


class WatermelonSpectrogramDataset(Dataset):
    """
    Dataset that loads the raw .wav file, computes the Mel spectrogram
    with *exactly* the same pipeline used at inference, then returns:
        spectrogram torch.float32, shape [1, n_mels, time_frames]
        label torch.long
    """
    def __init__(self, json_entries):
        self.entries = json_entries
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(
            [e["ripeness_label"] for e in self.entries]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 1)  Load *audio*, not a precalculated .npy
        audio_path = Path(DATASET_ROOT) / entry["audio_path"]
        signal, sr = librosa.load(str(audio_path), sr=SR)       # SR = 16,000

        # 2)  Extract Mel spectrogram  —— SINGLE SOURCE OF TRUTH
        mel_db = extract_mel_spectrogram(signal, sr)       # (n_mels, T)

        # 3)  Pad / trim in time‑dimension (axis=1) to FIXED_WIDTH frames
        if mel_db.shape[1] < FIXED_WIDTH:
            pad = FIXED_WIDTH - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
        elif mel_db.shape[1] > FIXED_WIDTH:
            mel_db = mel_db[:, :FIXED_WIDTH]

        # 4)  Convert → tensor and add dummy channel dim  [1, n_mels, T]
        spectrogram = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label       = torch.tensor(self.labels[idx], dtype=torch.long)
        return spectrogram, label

    # Helper so you can recover the original label strings elsewhere
    def get_label_encoder(self):
        return self.label_encoder
