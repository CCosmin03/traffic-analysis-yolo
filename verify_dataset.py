# verify_dataset.py
import os

for split in ('train','val'):
    imgs = {f.rsplit('.',1)[0] for f in os.listdir(f'dataset/images/{split}') if f.lower().endswith(('.jpg','.png'))}
    lbs  = {f.rsplit('.',1)[0] for f in os.listdir(f'dataset/labels/{split}') if f.endswith('.txt')}
    missing = imgs - lbs
    print(f"{split}: {len(imgs)} img, {len(lbs)} lbl — missing: {missing or 'none'}")
