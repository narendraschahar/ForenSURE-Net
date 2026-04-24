from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

cover_dir = Path("data/BOSSBase/cover")
stego_dir = Path("data/BOSSBase/stego")
stego_dir.mkdir(parents=True, exist_ok=True)

cover_images = sorted(list(cover_dir.glob("*.pgm")))

if len(cover_images) == 0:
    raise ValueError("No .pgm images found in data/BOSSBase/cover")

print("Cover images found:", len(cover_images))

for img_path in tqdm(cover_images, desc="Generating LSB stego images"):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Random payload bits
    payload = np.random.randint(0, 2, size=arr.shape, dtype=np.uint8)

    # Replace least significant bit
    stego_arr = (arr & 254) | payload

    stego_img = Image.fromarray(stego_arr, mode="L")
    stego_img.save(stego_dir / img_path.name)

print("Stego images saved to:", stego_dir)