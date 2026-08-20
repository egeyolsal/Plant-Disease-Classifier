from huggingface_hub import hf_hub_download
import os, json
import pandas as pd

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"
EXTRACT_DIR = "data_extracted"

# Bu dosyalar zaten önbellekte (cache), yeniden indirmeyecek, sadece yolunu alacak
train_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_train.txt")
test_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_test.txt")
leaf_map_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="leaf_grouping/leaf-map.json")

with open(leaf_map_path, "r", encoding="utf-8") as f:
    leaf_map = json.load(f)

def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def compute_leaf_id(file_rel_path, class_name, leaf_map):
    # Orijinal plant_village.py'deki mantığın birebir aynısı
    file_name = os.path.basename(file_rel_path)
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    for ext in [".jpg", ".JPG", ".png", ".PNG"]:
        image_identifier = image_identifier.replace(ext, "")
    image_identifier = image_identifier.strip()
    lookup_key = image_identifier.lower().strip()
    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for s in suggestions:
            if class_name in s:
                return s
        return f"fallback_{image_identifier}"
    return f"fallback_{image_identifier}"

def build_manifest(split_file, split_name):
    paths = load_split(split_file)
    records = []
    missing = 0
    for rel_path in paths:
        full_path = os.path.join(EXTRACT_DIR, rel_path)
        parts = rel_path.split("/")
        if len(parts) < 4:
            print("Beklenmedik path formatı, atlanıyor:", rel_path)
            continue
        class_name = parts[2]
        species = class_name.split("___")[0]
        exists = os.path.exists(full_path)
        if not exists:
            missing += 1
        leaf_id = compute_leaf_id(rel_path, class_name, leaf_map)
        records.append({
            "split": split_name, "rel_path": rel_path, "full_path": full_path,
            "exists": exists, "class_name": class_name, "species": species,
            "leaf_id": leaf_id,
        })
    print(f"{split_name}: {len(records)} kayıt işlendi, {missing} dosya eksik")
    return records

print("Manifest oluşturuluyor...")
train_records = build_manifest(train_split_path, "train")
test_records = build_manifest(test_split_path, "test")

df = pd.DataFrame(train_records + test_records)
print(f"\nToplam kayıt: {len(df)}")
print(f"Diskte var olan dosya: {df['exists'].sum()} / {len(df)}")

print("\n=== Split bazında sayı ===")
print(df['split'].value_counts())

print(f"\n=== Benzersiz sınıf sayısı: {df['class_name'].nunique()} ===")
print(f"=== Benzersiz tür sayısı: {df['species'].nunique()} ===")
print("Türler:", sorted(df['species'].unique()))

print(f"\n=== Benzersiz leaf_id sayısı: {df['leaf_id'].nunique()} ===")
fallback_count = df['leaf_id'].str.startswith("fallback_").sum()
print(f"leaf-map'te bulunamayan (fallback) kayıt sayısı: {fallback_count} ({fallback_count/len(df)*100:.1f}%)")

df.to_csv("manifest.csv", index=False)
print("\nKaydedildi: manifest.csv")