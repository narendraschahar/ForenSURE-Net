from pathlib import Path
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class BOSSBasePairedDataset(Dataset):
    """
    Loads Cover and Stego images in perfectly synchronized pairs.
    This guarantees that the exact same random crop and flip are applied to both,
    allowing the SRNet to calculate residual differences without image variance interference.
    """
    def __init__(self, cover_dir, stego_dir, image_names, image_size=256, is_train=False):
        self.cover_dir = Path(cover_dir)
        self.stego_dir = Path(stego_dir)
        self.image_names = image_names
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        cover_path = self.cover_dir / img_name
        stego_path = self.stego_dir / img_name
        
        cover_img = Image.open(cover_path).convert('L')
        stego_img = Image.open(stego_path).convert('L')

        # 1. Synchronized Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(cover_img, output_size=(self.image_size, self.image_size))
        cover_img = TF.crop(cover_img, i, j, h, w)
        stego_img = TF.crop(stego_img, i, j, h, w)

        # 2. Synchronized Random Flips
        if self.is_train:
            if random.random() > 0.5:
                cover_img = TF.hflip(cover_img)
                stego_img = TF.hflip(stego_img)
            if random.random() > 0.5:
                cover_img = TF.vflip(cover_img)
                stego_img = TF.vflip(stego_img)

        # 3. Convert to Tensor & Normalize
        cover_tensor = TF.to_tensor(cover_img)
        cover_tensor = TF.normalize(cover_tensor, mean=[0.5], std=[0.5])
        
        stego_tensor = TF.to_tensor(stego_img)
        stego_tensor = TF.normalize(stego_tensor, mean=[0.5], std=[0.5])

        # 4. Stack them: shape [2, 1, H, W]
        images = torch.stack([cover_tensor, stego_tensor], dim=0)
        
        # Labels: Cover=0, Stego=1
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

        return images, labels