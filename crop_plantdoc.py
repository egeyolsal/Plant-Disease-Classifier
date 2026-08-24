import json
import os
import pandas as pd
from PIL import Image

PLANTDOC_DIR = "plantdoc_raw"
OUTPUT_DIR = "plantdoc_cropped"

with open("configs/class_mapping.json") as f:
    pv_class_mapping = json.load(f)
target_classes = sorted(set(pv_class_mapping.values()))

# PlantDoc category name -> our target class, or None to exclude.
# None = ambiguous health status ("leaf" alone) or no species/disease info.
PLANTDOC_TO_TARGET = {
    "leaves": None,
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": None,
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell",
    "Bell_pepper leaf spot": "Pepper,_bell",
    "Blueberry leaf": "Blueberry",
    "Cherry leaf": "Cherry_(including_sour)",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach",
    "Potato leaf": None,
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry",
    "Soyabean leaf": "Soybean",
    "Soybean leaf": "Soybean",
    "Squash Powdery mildew leaf": "Squash",
    "Strawberry leaf": "Strawberry",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": None,
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": None,
    "grape leaf black rot": "Grape___Black_rot",
}

covered = set(v for v in PLANTDOC_TO_TARGET.values() if v)
missing = sorted(set(target_classes) - covered)
print(f"Target classes with NO PlantDoc coverage ({len(missing)}):")
for t in missing:
    print(" -", t)


def crop_split(split):
    ann_path = os.path.join(PLANTDOC_DIR, split, "_annotations.coco.json")
    img_dir = os.path.join(PLANTDOC_DIR, split)
    out_dir = os.path.join(OUTPUT_DIR, split)

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_category = {cat["id"]: cat["name"] for cat in coco["categories"]}

    records = []
    skipped_unsupported = 0
    skipped_invalid = 0

    for ann in coco["annotations"]:
        cat_name = id_to_category[ann["category_id"]]
        target_class = PLANTDOC_TO_TARGET.get(cat_name)
        if target_class is None:
            skipped_unsupported += 1
            continue

        img_filename = id_to_filename[ann["image_id"]]
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            skipped_invalid += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            skipped_invalid += 1
            continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            crop = im.crop((x, y, x + w, y + h))
            if crop.width < 10 or crop.height < 10:
                skipped_invalid += 1
                continue

            safe_class = target_class.replace("/", "_").replace(",", "")
            crop_dir = os.path.join(out_dir, safe_class)
            os.makedirs(crop_dir, exist_ok=True)
            crop_filename = f"{ann['id']}.jpg"
            crop_path = os.path.join(crop_dir, crop_filename)
            crop.save(crop_path, "JPEG", quality=95)

        records.append({
            "source_image": img_filename,
            "source_image_id": ann["image_id"],
            "bbox_id": ann["id"],
            "source_class": cat_name,
            "target_class": target_class,
            "crop_path": crop_path,
            "official_split": split,
        })

    print(f"{split}: {len(records)} crops saved, {skipped_unsupported} unsupported category, {skipped_invalid} invalid")
    return records


all_records = []
for split in ["train", "valid", "test"]:
    all_records.extend(crop_split(split))

df = pd.DataFrame(all_records)
df.to_csv("plantdoc_manifest.csv", index=False)

print(f"\nTotal crops: {len(df)}")
print(df["official_split"].value_counts())
print("\nCrops per target class:")
print(df["target_class"].value_counts())

with open("configs/plantdoc_class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(PLANTDOC_TO_TARGET, f, indent=2, ensure_ascii=False)
print("\nSaved: configs/plantdoc_class_mapping.json")