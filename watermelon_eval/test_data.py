import json
from collections import Counter

# Load the JSON file
with open("/watermelon_dataset/ripeness_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Count the occurrences of each ripeness label
labels = [entry["ripeness_label"] for entry in data]
label_counts = Counter(labels)

# Display the results
print("Ripeness label distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
