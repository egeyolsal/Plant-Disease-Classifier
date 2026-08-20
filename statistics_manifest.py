import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

df = pd.read_csv("manifest.csv")
os.makedirs("reports", exist_ok=True)

# === Sınıf dağılımı grafiği ===
class_counts = df['class_name'].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 12))
class_counts.plot(kind='barh', ax=ax)
ax.set_xlabel("Görüntü sayısı")
ax.set_title("PlantVillage - Sınıf Başına Görüntü Sayısı (38 sınıf)")
plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=150)
plt.close()
print("Kaydedildi: reports/class_distribution.png")

species_counts = df['species'].value_counts().sort_values(ascending=False)
print("\n=== Tür başına toplam görüntü ===")
print(species_counts)

# === Örnek görüntüler (her türden bir tane) ===
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
sample_df = df.groupby('species').first().reset_index().head(10)
for i, ax in enumerate(axes.flat):
    if i < len(sample_df):
        row = sample_df.iloc[i]
        img = mpimg.imread(row['full_path'])
        ax.imshow(img)
        ax.set_title(f"{row['species']}\n{row['class_name']}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig("reports/sample_images.png", dpi=150)
plt.close()
print("Kaydedildi: reports/sample_images.png")

# === Özet JSON ===
fallback_count = int(df['leaf_id'].str.startswith("fallback_").sum())
summary = {
    "total_images": int(len(df)),
    "train_images": int((df['split']=='train').sum()),
    "test_images": int((df['split']=='test').sum()),
    "num_classes": int(df['class_name'].nunique()),
    "num_species": int(df['species'].nunique()),
    "species_list": sorted(df['species'].unique().tolist()),
    "class_counts": class_counts.to_dict(),
    "species_counts": species_counts.to_dict(),
    "unique_leaf_ids": int(df['leaf_id'].nunique()),
    "leaf_id_fallback_count": fallback_count,
    "leaf_id_fallback_percentage": round(fallback_count/len(df)*100, 1),
    "all_source_files_found_on_disk": bool(df['exists'].all()),
}
with open("reports/dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Kaydedildi: reports/dataset_summary.json")

print("\n=== ÖZET ===")
skip = {'class_counts', 'species_counts', 'species_list'}
print(json.dumps({k: v for k, v in summary.items() if k not in skip}, indent=2, ensure_ascii=False))