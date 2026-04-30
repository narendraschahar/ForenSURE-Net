"""
generate_lsb_stego.py — Generate LSB steganography for pipeline verification.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def embed_lsb(img_arr: np.ndarray, payload_bpp: float = 0.5, seed: int = 0) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    flat = img_arr.flatten().astype(np.uint8)
    n_embed  = int(flat.size * payload_bpp)
    positions = rng.choice(flat.size, n_embed, replace=False)
    bits      = rng.integers(0, 2, n_embed, dtype=np.uint8)
    flat[positions] = (flat[positions] & 0xFE) | bits
    return flat.reshape(img_arr.shape)

def main():
    cover_dir = Path("data/ForenSURE_Dataset/cover")
    stego_dir = Path("data/ForenSURE_Dataset/stego_lsb_05")
    stego_dir.mkdir(parents=True, exist_ok=True)
    covers = sorted(cover_dir.glob("*.pgm"))
    print(f"Generating LSB stego for {len(covers)} images...")
    for i, path in enumerate(tqdm(covers)):
        arr   = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        stego = embed_lsb(arr, payload_bpp=0.5, seed=i)
        Image.fromarray(stego, mode="L").save(stego_dir / path.name)
    print("\n=== Sanity Check ===")
    for path in sorted(cover_dir.glob("*.pgm"))[:3]:
        cover = np.array(Image.open(path)).astype(np.float32)
        stego = np.array(Image.open(stego_dir / path.name)).astype(np.float32)
        changed = int(np.sum(cover != stego))
        print(f"{path.name}: {changed} changed pixels ({changed/cover.size*100:.1f}%) — expected ~25%")

if __name__ == "__main__":
    main()