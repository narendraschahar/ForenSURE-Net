from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BOSSBaseDataset(Dataset):
    def __init__(self, split_data, image_size=256, is_train=False):
        self.samples = []
        self.is_train = is_train

        if isinstance(split_data, str):
            with open(split_data, "r") as f:
                for line in f:
                    path, label = line.strip().split(",")
                    self.samples.append((path, int(label)))
        elif isinstance(split_data, list):
            self.samples = split_data

        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop((image_size, image_size), pad_if_needed=True)
        ]

        if self.is_train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path)
        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label