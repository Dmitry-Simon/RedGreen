import os
import hashlib
import matplotlib.pyplot as plt
import numpy as np

# Path to the watermelon dataset folder
src_folder = "C:\\Users\\thele\\Desktop\\watermelon_dataset(1)\\datasets"

# Lists to store results
watermelons = []
audio_counts = []
image_counts = []
total_counts = []
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
    watermelon_path = os.path.join(src_folder, watermelon, "chu")

    # Check if the "chu" folder exists inside the watermelon folder
    if os.path.isdir(watermelon_path):
        audio_count = 0
        image_count = 0
        hashes_seen = set()
        duplicates = 0

        # Walk through the subdirectories of "chu"
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
                elif file.endswith(('.jpg', '.png')):
                    image_count += 1

        # Append the results
        watermelons.append(watermelon)
        audio_counts.append(audio_count)
        image_counts.append(image_count)
        total_counts.append(audio_count + image_count)
        duplicate_counts.append(duplicates)

# Plotting stacked bar chart (audio + image)
fig1, ax1 = plt.subplots(figsize=(12, 6))
x = range(len(watermelons))

ax1.bar(x, audio_counts, width=0.4, label='Audio Files', align='center')
ax1.bar(x, image_counts, width=0.4, bottom=audio_counts, label='Image Files', align='center')

ax1.set_xlabel('Watermelon Labels')
ax1.set_ylabel('Number of Files')
ax1.set_title('Number of Audio and Image Files per Watermelon (chu only)')
ax1.set_xticks(x)
ax1.set_xticklabels(watermelons, rotation=45)
ax1.legend()

plt.tight_layout()
plt.show()

# Plotting pie chart of total file type distribution
total_audio = sum(audio_counts)
total_images = sum(image_counts)

fig2, ax2 = plt.subplots()
ax2.pie([total_audio, total_images], labels=['Audio Files', 'Image Files'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Overall File Type Distribution (chu only)')
plt.axis('equal')
plt.show()

# Plotting line graph of total files per watermelon
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(watermelons, total_counts, marker='o', linestyle='-', color='blue')
ax3.set_xlabel('Watermelon Labels')
ax3.set_ylabel('Total Number of Files')
ax3.set_title('Total Files per Watermelon (chu only)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# New graph: Audio to Image Ratio
ratios = [a / i if i > 0 else 0 for a, i in zip(audio_counts, image_counts)]
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.bar(watermelons, ratios, color='green')
ax4.set_xlabel('Watermelon Labels')
ax4.set_ylabel('Audio/Image Ratio')
ax4.set_title('Audio to Image File Ratio per Watermelon')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# New graph: Standard Deviation of Counts
counts_array = np.array([audio_counts, image_counts])
std_devs = np.std(counts_array, axis=0)
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.bar(watermelons, std_devs, color='orange')
ax5.set_xlabel('Watermelon Labels')
ax5.set_ylabel('Standard Deviation')
ax5.set_title('Standard Deviation between Audio and Image Counts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# New graph: Duplicate files per watermelon
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.bar(watermelons, duplicate_counts, color='red')
ax6.set_xlabel('Watermelon Labels')
ax6.set_ylabel('Duplicate File Count')
ax6.set_title('Number of Duplicate Files per Watermelon (chu only)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
