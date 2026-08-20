import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

df = pd.read_csv("manifest_split.csv")

# Sadece training split'ten örnek al - augmentation SADECE train'e uygulanır
train_df = df[df['final_split'] == 'train']
samples = train_df.sample(4, random_state=42)

# Augmentation pipeline - preprocessing.json'daki listeyle birebir tutarlı
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.042),   # ~15 derece
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for row_idx, (_, sample_row) in enumerate(samples.iterrows()):
    img = mpimg.imread(sample_row['full_path'])
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)  # batch boyutu ekle

    axes[row_idx, 0].imshow(img)
    axes[row_idx, 0].set_title(f"Orijinal\n{sample_row['class_name']}", fontsize=8)
    axes[row_idx, 0].axis('off')

    for col_idx in range(1, 4):
        augmented = augmenter(img_tensor, training=True)
        augmented_img = tf.clip_by_value(augmented[0], 0, 255).numpy().astype("uint8")
        axes[row_idx, col_idx].imshow(augmented_img)
        axes[row_idx, col_idx].set_title(f"Augment {col_idx}", fontsize=8)
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.savefig("reports/augmentation_examples.png", dpi=150)
plt.close()
print("Kaydedildi: reports/augmentation_examples.png")
print("\nTensorFlow versiyonu:", tf.__version__)
print("GPU görünüyor mu (lokalde 'False' beklenen, normal):", len(tf.config.list_physical_devices('GPU')) > 0)