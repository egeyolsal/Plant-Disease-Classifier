import json
import os

PLANTDOC_DIR = "plantdoc_raw"

for split in ["train", "test", "valid"]:
    split_dir = os.path.join(PLANTDOC_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"{split}: directory not found, skipping")
        continue

    ann_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
    if not ann_files:
        print(f"{split}: no annotation file found")
        continue

    ann_path = os.path.join(split_dir, ann_files[0])
    with open(ann_path) as f:
        coco = json.load(f)

    print(f"\n=== {split} ({ann_files[0]}) ===")
    print("Images:", len(coco["images"]))
    print("Annotations:", len(coco["annotations"]))
    print("Categories:", len(coco["categories"]))
    for cat in coco["categories"]:
        print(" -", cat["id"], cat["name"])

    print("\nSample image record:")
    print(json.dumps(coco["images"][0], indent=2))

    print("\nSample annotation record:")
    print(json.dumps(coco["annotations"][0], indent=2))