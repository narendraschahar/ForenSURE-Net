from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BOSSBaseDataset(Dataset):
    def __init__(self, split_file, image_size=256):
        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                path, label = line.strip().split(",")
                self.samples.append((path, int(label)))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path)
        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label