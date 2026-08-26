import json
import pandas as pd

with open("configs/classes.json") as f:
    classes_config = json.load(f)

disease_supported = set(classes_config["disease_supported_species"])

df = pd.read_csv("manifest_split.csv")
unique_pairs = df[["class_name", "species"]].drop_duplicates()

mapping = {}
for _, row in unique_pairs.iterrows():
    if row["species"] in disease_supported:
        mapping[row["class_name"]] = row["class_name"]
    else:
        mapping[row["class_name"]] = row["species"]

with open("configs/class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

target_classes = sorted(set(mapping.values()))
print(f"Original classes: {len(mapping)}")
print(f"Final target classes: {len(target_classes)}")
for c in target_classes:
    print(" -", c)