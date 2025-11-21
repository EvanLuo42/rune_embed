import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class RuneResNetEmbedding(nn.Module):
    def __init__(self, emb_dim=128, freeze_backbone=False):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        original_conv1 = backbone.conv1

        new_conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            new_conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True)

        backbone.conv1 = new_conv1

        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Linear(512, emb_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, dim=1)
