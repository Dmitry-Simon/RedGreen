import json
import numpy as np
import librosa
from pathlib import Path

from back_end.mel_utils import extract_mel_spectrogram, SR

# 1) locate this script’s directory
script_dir   = Path(__file__).resolve().parent

# 2) go up one level (to the project root) and into watermelon_dataset/
dataset_root = script_dir.parent / "watermelon_dataset"

labels_path  = dataset_root / "ripeness_labels.json"
out_root     = dataset_root / "processed_spectrograms"

# make sure output root exists
out_root.mkdir(parents=True, exist_ok=True)

# load your labels
with labels_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    wav_path = dataset_root / entry["audio_path"]
    signal, sr = librosa.load(str(wav_path), sr=SR)

    mel_spec   = extract_mel_spectrogram(signal, sr)
    spec_name  = wav_path.stem + ".npy"
    class_dir  = out_root / entry["ripeness_label"]
    class_dir.mkdir(exist_ok=True)

    full_spec  = class_dir / spec_name
    np.save(str(full_spec), mel_spec)

    # store relative spectrogram path
    entry["spectrogram_path"] = full_spec.relative_to(dataset_root).as_posix()

# finally, write out the augmented JSON alongside your spectrograms
out_labels = dataset_root / "ripeness_with_specs.json"

with out_labels.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Spectrograms saved under {out_root}")
print(f"✅ Updated JSON written to {out_labels}")
