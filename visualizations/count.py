import os
import hashlib
import matplotlib.pyplot as plt
import numpy as np

# Path to the watermelon dataset folder
src_folder = r"../watermelon_dataset/datasets"

# Lists to store results
watermelons = []
audio_counts = []
duplicate_counts = []

# Helper function to hash file content
def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Iterate through each watermelon folder
for watermelon in os.listdir(src_folder):
    watermelon_path = os.path.join(src_folder, watermelon, "audios")

    if os.path.isdir(watermelon_path):
        audio_count = 0
        hashes_seen = set()
        duplicates = 0

        # Walk through the "audios" folder
        for root, _, files in os.walk(watermelon_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = get_file_hash(file_path)
                if file_hash in hashes_seen:
                    duplicates += 1
                else:
                    hashes_seen.add(file_hash)

                if file.endswith(('.wav', '.mp3', '.m4a')):
                    audio_count += 1

        watermelons.append(watermelon)
        audio_counts.append(audio_count)
        duplicate_counts.append(duplicates)

# Bar chart: Audio files per watermelon
fig1, ax1 = plt.subplots(figsize=(12, 6))
x = range(len(watermelons))
ax1.bar(x, audio_counts, width=0.6, label='Audio Files')
ax1.set_xlabel('Watermelon Labels')
ax1.set_ylabel('Number of Audio Files')
ax1.set_title('Audio Files per Watermelon')
ax1.set_xticks(x)
ax1.set_xticklabels(watermelons, rotation=45)
ax1.legend()
plt.tight_layout()
plt.show()
