import pandas as pd
import json
import os
import random
from collections import defaultdict

random.seed(42)  # Reproducibility

df = pd.read_csv("manifest.csv")
os.makedirs("configs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

leaf_class_check = df.groupby('leaf_id')['class_name'].nunique()
inconsistent = leaf_class_check[leaf_class_check > 1]
if len(inconsistent) > 0:
    print(f"WARNING: Found {len(inconsistent)} inconsistent leaf_ids assigned to multiple classes:")
    print(inconsistent.head())

leaf_groups = df.groupby('leaf_id').agg(
    class_name=('class_name', 'first'),
    species=('species', 'first'),
    n_images=('rel_path', 'count'),
    rel_paths=('rel_path', list),
).reset_index()

print(f"\nTotal leaf_id groups: {len(leaf_groups)}")
print(f"Images per group - min: {leaf_groups['n_images'].min()}, max: {leaf_groups['n_images'].max()}, avg: {leaf_groups['n_images'].mean():.2f}")

split_assignment = {}
warnings_small_classes = []

for class_name, group_df in leaf_groups.groupby('class_name'):
    groups = group_df.sample(frac=1, random_state=42).to_dict('records')
    total_images = sum(g['n_images'] for g in groups)

    train_target = total_images * 0.70
    val_target = total_images * 0.85

    cum = 0
    for g in groups:
        cum += g['n_images']
        if cum <= train_target:
            split_assignment[g['leaf_id']] = 'train'
        elif cum <= val_target:
            split_assignment[g['leaf_id']] = 'val'
        else:
            split_assignment[g['leaf_id']] = 'test'

    test_count = sum(g['n_images'] for g in groups if split_assignment[g['leaf_id']] == 'test')
    if test_count < 20:
        warnings_small_classes.append((class_name, total_images, test_count))

df['final_split'] = df['leaf_id'].map(split_assignment)

train_ids = set(df[df['final_split'] == 'train']['leaf_id'])
val_ids = set(df[df['final_split'] == 'val']['leaf_id'])
test_ids = set(df[df['final_split'] == 'test']['leaf_id'])

leak_train_val = train_ids & val_ids
leak_train_test = train_ids & test_ids
leak_val_test = val_ids & test_ids

print(f"\n--- LEAKAGE TEST ---")
print(f"train∩val: {len(leak_train_val)}")
print(f"train∩test: {len(leak_train_test)}")
print(f"val∩test: {len(leak_val_test)}")
assert len(leak_train_val) == 0 and len(leak_train_test) == 0 and len(leak_val_test) == 0, "LEAKAGE DETECTED, HALTING"
print("Leakage test PASSED.")

print(f"\n--- Total images by split ---")
print(df['final_split'].value_counts())

if warnings_small_classes:
    print(f"\n--- Small class warnings (<20 images in test set) ---")
    for cls, total, test_c in warnings_small_classes:
        print(f"  {cls}: total {total}, only {test_c} in test")

df.to_csv("manifest_split.csv", index=False)

splits_metadata = {
    "source_manifest": "manifest_split.csv",
    "random_seed": 42,
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "split_unit": "leaf_id",
    "leakage_test_passed": True,
    "note": "Full per-record split assignments live in manifest_split.csv, not duplicated here.",
}
with open("configs/splits.json", "w", encoding="utf-8") as f:
    json.dump(splits_metadata, f, indent=2, ensure_ascii=False)

classes_config = {
    "all_classes": sorted(df['class_name'].unique().tolist()),
    "disease_supported_species": ["Tomato", "Potato", "Apple", "Grape", "Corn_(maize)"],
    "species_only": sorted(set(df['species'].unique()) - {"Tomato", "Potato", "Apple", "Grape", "Corn_(maize)"}),
}
with open("configs/classes.json", "w", encoding="utf-8") as f:
    json.dump(classes_config, f, indent=2, ensure_ascii=False)

preprocessing_config = {
    "input_width": 224,
    "input_height": 224,
    "channels": 3,
    "color_order": "RGB",
    "resize_method": "bilinear",
    "normalization": "rescale_1_255",  # pixel / 255.0
    "augmentation_train_only": ["horizontal_flip", "rotation_15deg", "zoom_0.1", "brightness_0.1"],
}
with open("configs/preprocessing.json", "w", encoding="utf-8") as f:
    json.dump(preprocessing_config, f, indent=2, ensure_ascii=False)

print("\nSaved: manifest_split.csv, configs/splits.json, configs/classes.json, configs/preprocessing.json")