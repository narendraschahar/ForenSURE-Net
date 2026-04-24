import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader
from src.datasets.bossbase_dataset import BOSSBaseDataset


dataset = BOSSBaseDataset(
    split_file="data/splits/train.txt",
    image_size=256
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)