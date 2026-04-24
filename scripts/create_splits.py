from pathlib import Path
import random


SEED = 42
random.seed(SEED)

cover_dir = Path("data/BOSSBase/cover")
stego_dir = Path("data/BOSSBase/stego")
split_dir = Path("data/splits")
split_dir.mkdir(parents=True, exist_ok=True)

cover_images = sorted(list(cover_dir.glob("*")))
stego_images = sorted(list(stego_dir.glob("*")))

samples = []

for img in cover_images:
    samples.append((str(img), 0))

for img in stego_images:
    samples.append((str(img), 1))

random.shuffle(samples)

n = len(samples)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

splits = {
    "train.txt": samples[:train_end],
    "val.txt": samples[train_end:val_end],
    "test.txt": samples[val_end:]
}

for filename, split_samples in splits.items():
    with open(split_dir / filename, "w") as f:
        for path, label in split_samples:
            f.write(f"{path},{label}\n")

print("Total samples:", n)
print("Train:", len(splits["train.txt"]))
print("Val:", len(splits["val.txt"]))
print("Test:", len(splits["test.txt"]))