import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

df = pd.read_csv("plantdoc_manifest.csv")
os.makedirs("reports", exist_ok=True)

sample_classes = ["Blueberry", "Tomato___Early_blight", "Potato___Early_blight",
                   "Corn_(maize)___Northern_Leaf_Blight", "Apple___Apple_scab",
                   "Grape___Black_rot"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, cls in zip(axes.flat, sample_classes):
    row = df[df["target_class"] == cls].iloc[0]
    img = mpimg.imread(row["crop_path"])
    ax.imshow(img)
    ax.set_title(f"{cls}\n{img.shape[1]}x{img.shape[0]}px", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("reports/plantdoc_crop_examples.png", dpi=150)
plt.close()
print("Saved: reports/plantdoc_crop_examples.png")