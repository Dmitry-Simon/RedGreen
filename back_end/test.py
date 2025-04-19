from mel_utils import extract_mel_spectrogram, SR
import numpy as np, librosa, json, os

json_path = r"..\watermelon_dataset\ripeness_with_specs.json"

with open(json_path, "r", encoding="utf-8") as f:   # ← explicit encoding
    entries = json.load(f)

entry = entries[0]        # first example
print("Loaded first entry OK")

# --- load spectrogram used in training ------------------------
train_spec = np.load(os.path.join(
        r"..\watermelon_dataset", entry["spectrogram_path"]
))

# --- recompute spectrogram the “API way” ----------------------
signal, _  = librosa.load(entry["audio_path"], sr=SR)
api_spec   = extract_mel_spectrogram(signal, SR)

print("train_spec shape:", train_spec.shape)
print("api_spec   shape:", api_spec.shape)
print("max abs diff:", np.abs(train_spec - api_spec).max())
