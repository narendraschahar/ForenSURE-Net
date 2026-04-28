import os
import shutil
from pathlib import Path
from tqdm import tqdm

from generate_hill_stego import calculate_hill_costs, embed_payload
from PIL import Image
import numpy as np

def prepare_unified_dataset():
    base_dir = Path("data")
    dataset_dir = base_dir / "ForenSURE_Dataset"
    cover_dir = dataset_dir / "cover"
    stego_dir = dataset_dir / "stego_hill_04"
    
    cover_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)
    
    # Use standard dataset folders
    bossbase_dir = base_dir / "BOSSBase" / "cover"
    bows2_dir = base_dir / "BOWS2" / "cover"
    
    if not bossbase_dir.exists() or not bows2_dir.exists():
        print("Error: Could not find the cover folders inside data/BOSSBase or data/BOWS2.")
        return

    # 1. Merge Covers
    print("Merging BOSSBase and BOWS2 covers...")
    
    boss_files = list(bossbase_dir.glob("*.pgm"))
    bows2_files = list(bows2_dir.glob("*.pgm"))
    
    print(f"Found {len(boss_files)} BOSSBase images and {len(bows2_files)} BOWS2 images.")
    
    for f in tqdm(boss_files, desc="Copying BOSSBase Covers"):
        target = cover_dir / f"boss_{f.name}"
        if not target.exists():
            shutil.copy(f, target)
            
    for f in tqdm(bows2_files, desc="Copying BOWS2 Covers"):
        target = cover_dir / f"bows_{f.name}"
        if not target.exists():
            shutil.copy(f, target)

    # 2. Generate HILL Stego for all 20,000
    all_covers = list(cover_dir.glob("*.pgm"))
    print(f"Generating Advanced HILL Steganography for {len(all_covers)} images...")
    
    bpp = 0.4
    for img_path in tqdm(all_covers, desc="Embedding HILL"):
        target_path = stego_dir / img_path.name
        if target_path.exists():
            continue # Skip if already done
            
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        
        costs = calculate_hill_costs(arr)
        stego_arr = embed_payload(arr, costs, bpp)
        
        stego_img = Image.fromarray(stego_arr, mode="L")
        stego_img.save(target_path)
        
    print("Unified Dataset Generation Complete! You are ready to train on Kaggle.")

if __name__ == "__main__":
    prepare_unified_dataset()
