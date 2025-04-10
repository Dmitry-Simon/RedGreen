import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ECAPA_TDNN_Lite(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        self.res2block = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SEBlock(128),
        )

        self.conv2 = nn.Conv1d(128, 192, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input: (B, 1, Mel, Time) → squeeze to (B, Mel, Time)
        x = x.squeeze(1)

        x = self.relu(self.conv1(x))         # (B, 128, T)
        x = self.res2block(x) + x            # residual
        x = self.conv2(x)                    # (B, 192, T)
        x = self.pooling(x).squeeze(-1)      # (B, 192)
        x = self.classifier(x)               # (B, num_classes)
        return x
