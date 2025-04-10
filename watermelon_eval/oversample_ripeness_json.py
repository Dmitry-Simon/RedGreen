import json
import random
import os
from collections import defaultdict

# === CONFIG ===
INPUT_JSON = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\ripeness_with_specs.json"
OUTPUT_JSON = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\processed_spectrograms\balanced_ripeness.json"

# === Load original entries ===
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)

# === Group entries by label ===
grouped = defaultdict(list)
for entry in entries:
    grouped[entry['ripeness_label']].append(entry)

# === Determine max class size ===
max_size = max(len(g) for g in grouped.values())

# === Oversample minority classes ===
balanced_entries = []
for label, samples in grouped.items():
    if len(samples) < max_size:
        extra = random.choices(samples, k=max_size - len(samples))  # random duplication
        samples += extra
    balanced_entries.extend(samples)

# === Shuffle for good measure ===
random.shuffle(balanced_entries)

# === Save new JSON ===
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(balanced_entries, f, ensure_ascii=False, indent=2)

print(f"✅ Balanced dataset saved with {len(balanced_entries)} entries at:\n{OUTPUT_JSON}")
