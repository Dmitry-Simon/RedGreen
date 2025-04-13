import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os
import json
import librosa
from settings import *

class WatermelonSpectrogramDataset(Dataset):
    def __init__(self, json_data):
        self.entries = json_data
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform([e['ripeness_label'] for e in self.entries])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        spectrogram_path_full = os.path.join('../watermelon_dataset/', entry['spectrogram_path'])
        spectrogram = np.load(spectrogram_path_full)

        # Padding or trimming to fixed time length
        if spectrogram.shape[1] < FIXED_WIDTH:
            # print('padded the spectrogram')
            pad_width = FIXED_WIDTH - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        elif spectrogram.shape[1] > FIXED_WIDTH:
            # print('truncated the spectrogram')
            spectrogram = spectrogram[:, :FIXED_WIDTH]

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spectrogram, label

    def get_label_encoder(self):
        return self.label_encoder

