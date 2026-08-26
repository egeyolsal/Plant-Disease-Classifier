import pandas as pd
from PIL import Image

df = pd.read_csv("plantdoc_manifest.csv")

df["width"] = df["crop_path"].apply(lambda p: Image.open(p).size[0])
df["height"] = df["crop_path"].apply(lambda p: Image.open(p).size[1])
df["min_dim"] = df[["width", "height"]].min(axis=1)

print("=== Min dimension distribution per class ===")
print(df.groupby("target_class")["min_dim"].describe()[["min", "25%", "50%", "mean"]])

MIN_SIZE = 50
too_small = (df["min_dim"] < MIN_SIZE).sum()
print(f"\nCrops smaller than {MIN_SIZE}px on their shortest side: {too_small} / {len(df)} ({too_small/len(df)*100:.1f}%)")
print("\nBreakdown by class:")
print(df[df["min_dim"] < MIN_SIZE]["target_class"].value_counts())

df.to_csv("plantdoc_manifest.csv", index=False)