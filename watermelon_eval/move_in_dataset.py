import os
import re
import json

import librosa
import numpy as np

def get_ripeness_label(sugar_level): # todo: look into it after the model trains
    if sugar_level < 10.2:
        return "unripe"
    elif 10.2 <= sugar_level < 10.8:
        return "mild"
    elif 10.8 <= sugar_level < 11.4:
        return "sweet"
    else:
        return "very_sweet"


def extract_sugar_level(folder_name):
    match = re.search(r'_(\d+(\.\d+)?)', folder_name)
    if match:
        return float(match.group(1))
    return None

def scan_dataset(base_path):
    data_entries = []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        sugar_level = extract_sugar_level(folder)
        if sugar_level is None:
            print(f"Skipping folder without sugar level: {folder}")
            continue

        ripeness_label = get_ripeness_label(sugar_level)
        audio_path = os.path.join(folder_path, 'audios')

        if not os.path.exists(audio_path):
            continue

        for file in os.listdir(audio_path):
            if file.endswith('.wav'):
                full_path = os.path.join(audio_path, file)
                data_entries.append({
                    'audio_path': full_path,
                    'sugar_level': sugar_level,
                    'ripeness_label': ripeness_label
                })

    return data_entries


# dataset_dir = r"C:\Users\thele\Documents\RedGreen\watermelon_dataset\datasets" #todo: remove
dataset_dir = '../watermelon_dataset/datasets'
entries = scan_dataset(dataset_dir)

# Save to JSON
output_path = os.path.join(dataset_dir, "ripeness_labels.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"Saved {len(entries)} entries to {output_path}")
