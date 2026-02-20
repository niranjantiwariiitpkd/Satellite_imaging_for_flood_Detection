import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
data_dir = "../data/EuroSAT"

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

water_classes = ["River", "SeaLake"]

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.idx_to_class[label]

        if class_name in water_classes:
            binary_label = 1
        else:
            binary_label = 0

        return img, torch.tensor(binary_label).float()

binary_dataset = BinaryDataset(dataset)

# Split 80 / 20
train_size = int(0.8 * len(binary_dataset))
val_size = len(binary_dataset) - train_size

train_dataset, val_dataset = random_split(binary_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)