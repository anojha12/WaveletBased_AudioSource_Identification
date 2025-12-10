import torch.nn as nn
from .resblock import ResBlock1D


class RawWaveCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, 8, kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.layer1 = ResBlock1D(8, 16)
        self.layer2 = ResBlock1D(16, 32)
        self.layer3 = ResBlock1D(32, 64)
        self.layer4 = ResBlock1D(64, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.relu(self.bn0(self.stem(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out).squeeze(-1)
        out = self.fc(out)
        return out
