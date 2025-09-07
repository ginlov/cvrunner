from torch.utils.data import Dataset, DataLoader

import torch

# -----------------------------
# Dummy Dataset
# -----------------------------
class DummyDataset(Dataset):
    def __init__(self, size=100, num_features=10, num_classes=2):
        super().__init__()
        self.x = torch.randn(size, num_features)
        self.y = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"inputs": self.x[idx], "labels": self.y[idx]}