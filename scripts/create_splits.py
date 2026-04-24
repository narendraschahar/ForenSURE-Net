from pathlib import Path
import random

SEED = 42
random.seed(SEED)

cover_dir = Path("data/BOSSBase/cover")
stego_dir = Path("data/BOSSBase/stego")
split_dir = Path("data/splits")
split_dir.mkdir(parents=True, exist_ok=True)

valid_exts = [".pgm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

cover_images = sorted([p for p in cover_dir.iterdir() if p.suffix.lower() in valid_exts])
stego_images = sorted([p for p in stego_dir.iterdir() if p.suffix.lower() in valid_exts])

print("Cover images:", len(cover_images))
print("Stego images:", len(stego_images))

if len(cover_images) == 0 or len(stego_images) == 0:
    raise ValueError("Both cover and stego folders must contain images.")

min_count = min(len(cover_images), len(stego_images))

cover_images = cover_images[:min_count]
stego_images = stego_images[:min_count]

cover_samples = [(str(img), 0) for img in cover_images]
stego_samples = [(str(img), 1) for img in stego_images]

random.shuffle(cover_samples)
random.shuffle(stego_samples)

def split_class(samples):
    n = len(samples)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]

cover_train, cover_val, cover_test = split_class(cover_samples)
stego_train, stego_val, stego_test = split_class(stego_samples)

splits = {
    "train.txt": cover_train + stego_train,
    "val.txt": cover_val + stego_val,
    "test.txt": cover_test + stego_test
}

for filename, split_samples in splits.items():
    random.shuffle(split_samples)

    with open(split_dir / filename, "w") as f:
        for path, label in split_samples:
            f.write(f"{path},{label}\n")

    labels = [label for _, label in split_samples]
    print(filename)
    print("  total:", len(labels))
    print("  cover:", labels.count(0))
    print("  stego:", labels.count(1))