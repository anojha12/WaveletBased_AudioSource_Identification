import torch.nn as nn
from .resblock import ResBlock1D

class WaveletCNN(nn.Module):
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
        x = self.relu(self.bn0(self.stem(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
