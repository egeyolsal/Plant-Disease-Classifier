import pandas as pd
import json

df = pd.read_csv("plantdoc_manifest.csv")

EXCLUDE_INSUFFICIENT_DATA = ["Tomato___Spider_mites Two-spotted_spider_mite"]

before = len(df)
df = df[~df["target_class"].isin(EXCLUDE_INSUFFICIENT_DATA)].copy()
print(f"Excluded {before - len(df)} crops from classes with insufficient PlantDoc data: {EXCLUDE_INSUFFICIENT_DATA}")

df.to_csv("plantdoc_manifest.csv", index=False)

with open("configs/classes.json") as f:
    classes_config = json.load(f)

no_coverage = [
    "Apple___Black_rot", "Apple___healthy", "Corn_(maize)___healthy",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange", "Potato___healthy",
    "Tomato___Target_Spot", "Tomato___healthy",
]
insufficient_coverage = EXCLUDE_INSUFFICIENT_DATA

classes_config["plantdoc_no_coverage"] = no_coverage
classes_config["plantdoc_insufficient_coverage"] = insufficient_coverage
classes_config["plantdoc_coverage_note"] = (
    f"{len(no_coverage)} target classes have zero PlantDoc representation; "
    f"{len(insufficient_coverage)} more have only 1-2 crops after quality filtering "
    f"and are excluded from domain adaptation and real-world evaluation despite "
    f"nominally having a category match."
)

with open("configs/classes.json", "w", encoding="utf-8") as f:
    json.dump(classes_config, f, indent=2, ensure_ascii=False)

print(f"\nFinal PlantDoc dataset: {len(df)} crops across {df['target_class'].nunique()} classes")
print(f"Total target classes with real-world coverage: {df['target_class'].nunique()} / 34")
print(f"Total target classes WITHOUT usable real-world coverage: {len(no_coverage) + len(insufficient_coverage)} / 34")
print("\nBy official split:")
print(df["official_split"].value_counts())