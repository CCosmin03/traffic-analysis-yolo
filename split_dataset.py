import os, random, shutil

# Config
src_img = 'dataset/all/images'
src_lbl = 'dataset/all/labels'
dst_img_train = 'dataset/images/train'
dst_lbl_train = 'dataset/labels/train'
dst_img_val   = 'dataset/images/val'
dst_lbl_val   = 'dataset/labels/val'
val_ratio = 0.2

# Creare directoare
for d in [dst_img_train, dst_lbl_train, dst_img_val, dst_lbl_val]:
    os.makedirs(d, exist_ok=True)

# Listă de imagini
imgs = [f for f in os.listdir(src_img) if f.lower().endswith(('.jpg','.png','.webp'))]
random.seed(42)
random.shuffle(imgs)
n_val = int(len(imgs) * val_ratio)

# Split și mutare
for i, img in enumerate(imgs):
    base = os.path.splitext(img)[0]
    lbl = base + '.txt'
    if i < n_val:
        dst_img, dst_lbl = dst_img_val, dst_lbl_val
    else:
        dst_img, dst_lbl = dst_img_train, dst_lbl_train
    shutil.move(os.path.join(src_img, img), os.path.join(dst_img, img))
    # verificăm că label-ul există înainte de mutare
    src_lbl_path = os.path.join(src_lbl, lbl)
    if os.path.isfile(src_lbl_path):
        shutil.move(src_lbl_path, os.path.join(dst_lbl, lbl))
    else:
        print(f"⚠️ Lipsă label pentru {img}")

print("Gata! Imagini și label-uri împărțite în train/val.")

