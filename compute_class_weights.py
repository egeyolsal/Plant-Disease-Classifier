import json
import pandas as pd

df = pd.read_csv("manifest_split.csv")

with open("configs/class_mapping.json") as f:
    class_mapping = json.load(f)

df["target_class"] = df["class_name"].map(class_mapping)

train_df = df[df["final_split"] == "train"]
class_counts_train = train_df["target_class"].value_counts()

n_classes = len(class_counts_train)
n_samples = len(train_df)

class_weights = {
    cls: round(n_samples / (n_classes * count), 4)
    for cls, count in class_counts_train.items()
}

print("Class weight - top 5 (least represented target classes)")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1])[:5]:
    print(f"  {cls}: {w}  (train: {class_counts_train[cls]} images)")

print("\nClass weight - bottom 5 (most represented target classes)")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1])[-5:]:
    print(f"  {cls}: {w}  (train: {class_counts_train[cls]} images)")

with open("configs/class_weights.json", "w", encoding="utf-8") as f:
    json.dump(class_weights, f, indent=2, ensure_ascii=False)
print("\nSaved: configs/class_weights.json")