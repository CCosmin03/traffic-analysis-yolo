import os
import random
import shutil
from pathlib import Path

# Configurare
random.seed(42)
val_ratio = 0.2

base_dir = Path("dataset_pedestrians")
train_src_img = base_dir / "train" / "images"
train_src_lbl = base_dir / "train" / "labels"

# Destinații noi
valid_dir = base_dir / "valid"
train_dst_img = valid_dir.parent / "train" / "images"   # rămâne acolo unde e
train_dst_lbl = valid_dir.parent / "train" / "labels"
valid_dst_img = valid_dir / "images"
valid_dst_lbl = valid_dir / "labels"

# Creare directoare (în cazul în care nu există deja)
for d in [train_dst_img, train_dst_lbl, valid_dst_img, valid_dst_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Listează imaginile din train
imgs = [f for f in os.listdir(train_src_img) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
random.shuffle(imgs)
n_val = int(len(imgs) * val_ratio)

# Mutare (sau copiere, vezi comentariu)
for i, img in enumerate(imgs):
    base = os.path.splitext(img)[0]
    lbl_filename = base + ".txt"

    # Decide destinația
    if i < n_val:
        dst_img_dir = valid_dst_img
        dst_lbl_dir = valid_dst_lbl
    else:
        dst_img_dir = train_dst_img
        dst_lbl_dir = train_dst_lbl

    src_img_path = train_src_img / img
    src_lbl_path = train_src_lbl / lbl_filename

    dst_img_path = dst_img_dir / img
    dst_lbl_path = dst_lbl_dir / lbl_filename

    # Mutați (sau copiați dacă vrei să păstrezi originalul)
    shutil.move(src_img_path, dst_img_path)
    if src_lbl_path.is_file():
        shutil.move(src_lbl_path, dst_lbl_path)
    else:
        print(f" Lipsă label pentru {img}")

print("Split complet: 20% valid, 80% train. Structura este acum:")
print(f"  Train images: {train_dst_img}")
print(f"  Train labels: {train_dst_lbl}")
print(f"  Val images:   {valid_dst_img}")
print(f"  Val labels:   {valid_dst_lbl}")
