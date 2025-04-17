import torch
import torch.nn as nn
import torch.nn.functional as F

# SE block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# Res2 dilated block with SE
class SE_Res2Block(nn.Module):
    def __init__(self, channels, scale=8):
        super().__init__()
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=3, dilation=i+1, padding=i+1)
            for i in range(scale - 1)
        ])
        self.bn = nn.BatchNorm1d(channels) # todo: consider LayerNorm
        self.se = SEBlock(channels)

    def forward(self, x):
        splits = torch.split(x, self.width, dim=1)
        out = [splits[0]]
        for i in range(1, self.scale):
            s = splits[i] + out[i - 1] if i > 1 else splits[i]
            out.append(self.convs[i - 1](s))
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        return self.se(out) + x  # Residual

# Attentive Statistics Pooling
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        alpha = self.attention(x)
        mean = torch.sum(x * alpha, dim=2)
        std = torch.sqrt(torch.sum(((x - mean.unsqueeze(2))**2) * alpha, dim=2).clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

# Full ECAPA-TDNN model
class ECAPA_TDNN_Full(nn.Module):
    def __init__(self, input_dim=64, channels=512, num_classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        self.res2_1 = SE_Res2Block(channels)
        self.res2_2 = SE_Res2Block(channels)
        self.res2_3 = SE_Res2Block(channels)

        self.concat_conv = nn.Conv1d(channels * 3, channels, kernel_size=1)
        self.asp = AttentiveStatsPool(channels)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(channels * 2),
            nn.Linear(channels * 2, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)  # [B, 1, T, F] → [B, F, T]
        x = self.layer1(x)
        out1 = self.res2_1(x)
        out2 = self.res2_2(out1)
        out3 = self.res2_3(out2)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.concat_conv(out)
        out = self.asp(out)
        out = self.fc(out)
        return out
