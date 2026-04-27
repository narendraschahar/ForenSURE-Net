import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from src.models.residual_stegnet import ResidualStegNet
from src.triage.triage_scorer import score_triage

class ForensicScanner:
    def __init__(self, weights_path, temperature_path=None, device=None, image_size=256):
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
        
        self.model = ResidualStegNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        if temperature_path and os.path.exists(temperature_path):
            ckpt = torch.load(temperature_path, map_location="cpu")
            self.temperature = float(ckpt["temperature"].item())
        else:
            # Fallback to uncalibrated
            self.temperature = 1.0 
            
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def scan_directory(self, input_dir):
        """Scans a directory of images and returns triage scores."""
        input_dir = Path(input_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".pgm", ".tif", ".tiff", ".bmp"}
        
        image_paths = [p for p in input_dir.glob("**/*") if p.suffix.lower() in valid_exts]
        
        if not image_paths:
            print(f"No valid images found in {input_dir}")
            return []
            
        print(f"Found {len(image_paths)} images. Scanning...")
        
        results = []
        
        # Process one by one (or could be batched for speed, but simple is robust for forensics)
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                img_tensor = self.transform(img).unsqueeze(0)
                
                mean_prob, reliability, uncertainty, triage = score_triage(
                    self.model, img_tensor, self.device, self.temperature, mc_passes=10
                )
                
                results.append({
                    "filename": str(path.name),
                    "filepath": str(path.absolute()),
                    "stego_probability": float(mean_prob[0]),
                    "reliability_score": float(reliability[0]),
                    "uncertainty_score": float(uncertainty[0]),
                    "triage_score": float(triage[0])
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"Scanned {idx + 1}/{len(image_paths)}...")
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        # Sort by triage score descending (most suspicious first)
        results = sorted(results, key=lambda x: x["triage_score"], reverse=True)
        return results
