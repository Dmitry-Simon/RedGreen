import os
import numpy as np
import matplotlib.pyplot as plt
import random


def show_spectrogram(file_path, title=None):
    mel_spec = np.load(file_path)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title if title else os.path.basename(file_path))
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bin")
    plt.tight_layout()
    plt.show()


def visualize_samples_from_each_class(base_dir):
    labels = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for label in labels:
        label_dir = os.path.join(base_dir, label)
        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if not files:
            print(f"No spectrograms found for {label}")
            continue

        sample_file = random.choice(files)
        file_path = os.path.join(label_dir, sample_file)
        show_spectrogram(file_path, title=f"{label.upper()} - {sample_file}")


# Usage
spectrogram_dir = r"../watermelon_dataset/processed_spectrograms"
visualize_samples_from_each_class(spectrogram_dir)
