import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class TripletRuneDataset(Dataset):
    def __init__(self, root: str):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.base = datasets.ImageFolder(root=root, transform=self.transform)
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(self.base):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels = list(self.label_to_indices.keys())
        assert len(self.labels) >= 2, "Require at least two labels to do triplet loss"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_img, anchor_label = self.base[index]

        pos_indices = self.label_to_indices[anchor_label]
        pos_index = index
        while pos_index == index and len(pos_indices) > 1:
            pos_index = random.choice(pos_indices)
        positive_img, _ = self.base[pos_index]

        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_index = random.choice(self.label_to_indices[neg_label])
        negative_img, _ = self.base[neg_index]

        return anchor_img, positive_img, negative_img
