import torch
import torch.nn as nn
import torch.nn.functional as f

class RuneEmbedding(nn.Module):
    def __init__(self, emb_dim: int=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = f.normalize(x, dim=1)
        return x
