from huggingface_hub import hf_hub_download
import zipfile
import os

REPO_ID = "mohanty/PlantVillage"
REPO_TYPE = "dataset"

print("Downloading data.zip... (full image archive, this may take a while)")
zip_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="data.zip")
print("Downloaded:", zip_path)

train_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_train.txt")
test_split_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="splits/color_test.txt")
leaf_map_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="leaf_grouping/leaf-map.json")

print("Split and leaf-map files downloaded:")
print(" -", train_split_path)
print(" -", test_split_path)
print(" -", leaf_map_path)

extract_dir = "data_extracted"
print("\nExtracting zip (this may also take a while)...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
print("Extracted to:", extract_dir)

print("\nExtracted directory structure (first 2 levels):")
for root, dirs, files in os.walk(extract_dir):
    depth = root.replace(extract_dir, "").count(os.sep)
    if depth <= 1:
        print(f"{root} -> {len(files)} files, subdirs: {dirs[:5]}")