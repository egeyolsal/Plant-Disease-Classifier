import pandas as pd

df = pd.read_csv("plantdoc_manifest.csv")

MIN_SIZE = 50
before = len(df)
df_filtered = df[df["min_dim"] >= MIN_SIZE].copy()
after = len(df_filtered)

print(f"Removed {before - after} crops below {MIN_SIZE}px ({(before-after)/before*100:.1f}%)")
print(f"Remaining: {after}")

print("\nRemaining crops per class:")
print(df_filtered["target_class"].value_counts())

# Flag any class that dropped to a concerning size after filtering
low_count = df_filtered["target_class"].value_counts()
concerning = low_count[low_count < 20]
if len(concerning) > 0:
    print(f"\nWARNING: classes with fewer than 20 crops remaining after filtering:")
    print(concerning)

df_filtered.to_csv("plantdoc_manifest.csv", index=False)
print("\nSaved filtered manifest: plantdoc_manifest.csv")