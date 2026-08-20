from huggingface_hub import hf_hub_download
import zipfile
import os

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"

print("data.zip indiriliyor... (görüntülerin tamamı, biraz sürebilir, sabırlı ol)")
zip_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="data.zip")
print("İndirildi:", zip_path)

train_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_train.txt")
test_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_test.txt")
leaf_map_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="leaf_grouping/leaf-map.json")

print("Split ve leaf-map dosyaları indirildi:")
print(" -", train_split_path)
print(" -", test_split_path)
print(" -", leaf_map_path)

extract_dir = "data_extracted"
print("\nZip çıkartılıyor (bu da biraz sürebilir)...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
print("Çıkartıldı:", extract_dir)

# Çıkan klasör yapısını gösterelim - split dosyalarındaki path'lerle eşleşiyor mu göreceğiz
print("\nÇıkartılan klasör yapısı (ilk 2 seviye):")
for root, dirs, files in os.walk(extract_dir):
    depth = root.replace(extract_dir, "").count(os.sep)
    if depth <= 1:
        print(f"{root} -> {len(files)} dosya, alt klasörler: {dirs[:5]}")