import os
import re
import json
from collections import Counter
from pathlib import Path


def get_ripeness_label(sugar_level):
    if sugar_level < 9.7:
        return "un_sweet"
    elif 9.7 <= sugar_level < 10.4:
        return "low_sweet"
    elif 10.4 <= sugar_level < 11.1:
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
                dataset_root = Path(dataset_dir).parent
                audio_dir = Path(folder_path) / "audios"
                wav_path = audio_dir / file
                # make it relative to your watermelon_dataset root, and POSIX-style:
                rel_path = wav_path.relative_to(dataset_root).as_posix()
                data_entries.append({
                    'audio_path': rel_path,
                    'sugar_level': sugar_level,
                    'ripeness_label': ripeness_label
                })

    return data_entries


dataset_dir = '../watermelon_dataset/datasets'
save_dir = '../watermelon_dataset'
entries = scan_dataset(dataset_dir)

# Save to JSON
output_path = os.path.join(save_dir, "ripeness_labels.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"Saved {len(entries)} entries to {output_path}")


# Load the JSON file
with open("../watermelon_dataset/ripeness_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Count the occurrences of each ripeness label
labels = [entry["ripeness_label"] for entry in data]
label_counts = Counter(labels)

# Display the results
print("Ripeness label distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
