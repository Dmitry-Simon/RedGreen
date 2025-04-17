import os
import json
import random

import numpy as np
import librosa

from mel_utils import extract_mel_spectrogram, SR

# ——— Load metadata ———
with open("../watermelon_dataset/processed_spectrograms/ripeness_with_specs.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# ——— Filter only entries where both files exist ———
valid = []
for e in entries:
    # normalize paths (handles mix of slashes/backslashes)
    wav = os.path.normpath(e["audio_path"])
    spec = os.path.normpath(e["spectrogram_path"])
    if os.path.exists(wav) and os.path.exists(spec):
        valid.append({"wav": wav, "spec": spec})
if len(valid) < 3:
    raise RuntimeError(f"Only found {len(valid)} valid entries on disk.")

# ——— Pick 3 at random ———
samples = random.sample(valid, 3)

# ——— Compare each one ———
for idx, s in enumerate(samples, start=1):
    wav_path = s["wav"]
    npy_path = s["spec"]

    # load training‐time Mel spec
    mel_train = np.load(npy_path)

    # compute inference‐time Mel spec
    signal, sr = librosa.load(wav_path, sr=SR)
    mel_live       = extract_mel_spectrogram(signal, sr)

    # numeric comparison
    print(f"\n=== Sample #{idx} ===")
    print("WAV:", wav_path)
    print("NPY:", npy_path)
    print("Train shape:", mel_train.shape)
    print("Live  shape:", mel_live.shape)

    if mel_train.shape == mel_live.shape:
        diff = np.abs(mel_train - mel_live)
        print("Max abs diff :", np.max(diff))
        print("Mean abs diff:", np.mean(diff))
    else:
        print("❌ Shape mismatch: check your FFT/hop settings.")

# (Optional) if you’d like to visualize one of them, uncomment:
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.title("Train"); plt.imshow(mel_train, aspect='auto'); plt.colorbar()
plt.subplot(1,2,2); plt.title("Live");  plt.imshow(mel_live,  aspect='auto'); plt.colorbar()
plt.tight_layout(); plt.show()
