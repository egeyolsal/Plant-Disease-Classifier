import pandas as pd
import json

df = pd.read_csv("manifest_split.csv")

# === 1. DOĞRULAMA: en küçük sınıfların gerçek split dağılımı ===
class_totals = df['class_name'].value_counts().sort_values()
smallest_classes = class_totals.head(8).index.tolist()

print("=== En küçük 8 sınıfın gerçek split dağılımı ===\n")
for cls in smallest_classes:
    sub = df[df['class_name'] == cls]
    counts = sub['final_split'].value_counts()
    total = len(sub)
    print(f"{cls} (toplam {total}):")
    for split in ['train', 'val', 'test']:
        c = int(counts.get(split, 0))
        pct = c/total*100 if total > 0 else 0
        print(f"    {split}: {c} ({pct:.1f}%)")
    print()

# === 2. class_weight hesaplama (sadece train split üzerinden) ===
train_df = df[df['final_split'] == 'train']
class_counts_train = train_df['class_name'].value_counts()

n_classes = len(class_counts_train)
n_samples = len(train_df)

class_weights = {}
for cls, count in class_counts_train.items():
    class_weights[cls] = round(n_samples / (n_classes * count), 4)

print("=== Class weight - en yüksek 5 (en az örnekli sınıflar) ===")
sorted_weights = sorted(class_weights.items(), key=lambda x: -x[1])
for cls, w in sorted_weights[:5]:
    print(f"  {cls}: {w}  (train'de {class_counts_train[cls]} örnek)")

print("\n=== Class weight - en düşük 5 (en çok örnekli sınıflar) ===")
for cls, w in sorted_weights[-5:]:
    print(f"  {cls}: {w}  (train'de {class_counts_train[cls]} örnek)")

with open("configs/class_weights.json", "w", encoding="utf-8") as f:
    json.dump(class_weights, f, indent=2, ensure_ascii=False)
print("\nKaydedildi: configs/class_weights.json")