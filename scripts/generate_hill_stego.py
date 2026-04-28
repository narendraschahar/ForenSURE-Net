import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import math

def calculate_hill_costs(img_arr):
    """Calculates HILL distortion costs for an image."""
    img_arr = img_arr.astype(np.float32)
    
    # High-pass filter
    H1 = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1]
    ]) / 4.0
    
    # Low-pass filter 1
    H2 = np.ones((3, 3)) / 9.0
    
    # Low-pass filter 2
    H3 = np.ones((15, 15)) / 225.0

    # 1. High pass residual
    R = convolve2d(img_arr, H1, mode='same', boundary='symm')
    
    # 2. Base cost (inverse of residual)
    C = 1.0 / (np.abs(R) + 1e-10)
    
    # 3. First Low pass smoothing
    C1 = convolve2d(C, H2, mode='same', boundary='symm')
    
    # 4. Second Low pass smoothing
    Cost = convolve2d(C1, H3, mode='same', boundary='symm')
    
    return Cost

def embed_payload(img_arr, costs, payload_bpp):
    """Simulates optimal STC embedding based on distortion costs."""
    # Binary search for optimal lambda
    lambda_val = 1.0
    lambda_min = 1e-3
    lambda_max = 1e3
    
    target_payload = payload_bpp * img_arr.size
    
    # Costs for +1 and -1 modification (ternary embedding approximation)
    rho = costs
    
    # Iterate to find lambda that matches payload
    for _ in range(20):
        # Probabilities of modifying pixel by +1 or -1
        p_mod = np.exp(-rho / lambda_val)
        p_mod = p_mod / (1 + 2 * p_mod)
        
        # Avoid log(0)
        p_mod = np.clip(p_mod, 1e-10, 0.5 - 1e-10)
        p_unmod = 1 - 2 * p_mod
        
        # Entropy (payload size)
        entropy = -2 * p_mod * np.log2(p_mod) - p_unmod * np.log2(p_unmod)
        current_payload = np.sum(entropy)
        
        if current_payload > target_payload:
            lambda_max = lambda_val
        else:
            lambda_min = lambda_val
            
        lambda_val = (lambda_min + lambda_max) / 2.0
        
    # Final probabilities with optimized lambda
    p_mod = np.exp(-rho / lambda_val)
    p_mod = p_mod / (1 + 2 * p_mod)
    
    # Simulate embedding
    rand_vals = np.random.rand(*img_arr.shape)
    
    modifications = np.zeros_like(img_arr, dtype=np.int8)
    modifications[rand_vals < p_mod] = 1
    modifications[(rand_vals >= p_mod) & (rand_vals < 2 * p_mod)] = -1
    
    # Apply modifications ensuring pixels stay within [0, 255]
    stego_arr = img_arr.astype(np.int16) + modifications
    stego_arr = np.clip(stego_arr, 0, 255).astype(np.uint8)
    
    return stego_arr

def main():
    bpp = 0.4
    cover_dir = Path("data/BOSSBase/cover")
    stego_dir = Path("data/BOSSBase/stego_hill")
    stego_dir.mkdir(parents=True, exist_ok=True)
    
    cover_images = sorted(list(cover_dir.glob("*.pgm")))
    print(f"Found {len(cover_images)} cover images. Generating HILL at {bpp} bpp...")
    
    for img_path in tqdm(cover_images):
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        
        costs = calculate_hill_costs(arr)
        stego_arr = embed_payload(arr, costs, bpp)
        
        stego_img = Image.fromarray(stego_arr, mode="L")
        stego_img.save(stego_dir / img_path.name)

if __name__ == "__main__":
    main()
