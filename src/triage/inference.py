import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from src.models.residual_stegnet import ResidualStegNet
from src.triage.triage_scorer import score_triage

class ForensicScanner:
    def __init__(self, models_dict, device=None, tile_size=256):
        """
        models_dict: dict of {"Algorithm Name": {"weights": "path.pth", "temperature": "path.pth"}}
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Initializing ForensicScanner on {self.device}...")
        self.tile_size = tile_size
        self.models = {}
        
        for algo, paths in models_dict.items():
            model = ResidualStegNet().to(self.device)
            try:
                model.load_state_dict(torch.load(paths["weights"], map_location=self.device))
                model.eval()
                
                temp = 1.0
                if paths.get("temperature") and os.path.exists(paths["temperature"]):
                    ckpt = torch.load(paths["temperature"], map_location="cpu")
                    temp = float(ckpt["temperature"].item())
                
                self.models[algo] = {"model": model, "temperature": temp}
                print(f"Loaded ensemble model for: {algo}")
            except Exception as e:
                print(f"Failed to load {algo}: {e}")
            
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def extract_tiles(self, img_tensor):
        """Extracts non-overlapping 256x256 tiles from a tensor of shape (1, H, W)"""
        _, h, w = img_tensor.shape
        tiles = []
        
        if h < self.tile_size or w < self.tile_size:
            # Pad if too small
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
            _, h, w = img_tensor.shape

        for i in range(0, h - self.tile_size + 1, self.tile_size):
            for j in range(0, w - self.tile_size + 1, self.tile_size):
                tile = img_tensor[:, i:i+self.tile_size, j:j+self.tile_size]
                tiles.append(tile)
                
        if not tiles:
            # Fallback for weird edge cases
            tiles.append(img_tensor[:, :self.tile_size, :self.tile_size])
            
        return torch.stack(tiles)

    def scan_image(self, image_path):
        """Scans a single image with all loaded models using tile scanning."""
        try:
            img = Image.open(image_path)
            img_tensor = self.transform(img)
            
            # Extract 256x256 tiles
            tiles = self.extract_tiles(img_tensor).to(self.device)
            
            best_triage_score = -1.0
            best_algo = "Unknown"
            best_prob = 0.0
            best_reliability = 0.0
            best_uncertainty = 1.0
            
            for algo, m_dict in self.models.items():
                model = m_dict["model"]
                temp = m_dict["temperature"]
                
                # Check all tiles
                mean_prob, reliability, uncertainty, triage = score_triage(
                    model, tiles, self.device, temp, mc_passes=5 # Lower passes for speed on tiles
                )
                
                # Max score among all tiles for this algorithm
                max_tile_idx = triage.argmax()
                algo_triage = float(triage[max_tile_idx])
                
                if algo_triage > best_triage_score:
                    best_triage_score = algo_triage
                    best_algo = algo
                    best_prob = float(mean_prob[max_tile_idx])
                    best_reliability = float(reliability[max_tile_idx])
                    best_uncertainty = float(uncertainty[max_tile_idx])
                    
            return {
                "filename": Path(image_path).name,
                "filepath": str(image_path),
                "stego_probability": best_prob,
                "reliability_score": best_reliability,
                "uncertainty_score": best_uncertainty,
                "triage_score": best_triage_score,
                "suspected_algorithm": best_algo
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def scan_directory(self, input_dir):
        """Scans a directory of images and returns ensemble triage scores."""
        input_dir = Path(input_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".pgm", ".tif", ".tiff", ".bmp"}
        
        image_paths = [p for p in input_dir.glob("**/*") if p.suffix.lower() in valid_exts]
        
        if not image_paths:
            return []
            
        print(f"Scanning {len(image_paths)} images with {len(self.models)} algorithms...")
        
        results = []
        for idx, path in enumerate(image_paths):
            res = self.scan_image(path)
            if res:
                results.append(res)
            if (idx + 1) % 10 == 0:
                print(f"Scanned {idx + 1}/{len(image_paths)}...")
                
        results = sorted(results, key=lambda x: x["triage_score"], reverse=True)
        return results
