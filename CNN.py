import pandas as pd
import torch.nn as nn
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FasterCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            DepthwiseSeparableConv(3, 48),  # Reduced channels (64 → 48)
            nn.ReLU(),
            nn.BatchNorm2d(48),
            DepthwiseSeparableConv(48, 96),  # Reduced channels (128 → 96)
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),
            DepthwiseSeparableConv(96, 192),  # Reduced channels (256 → 192)
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2, 2),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.4),  # Added dropout to reduce overfitting
            nn.Linear(192, num_classes)  # Adjusted input size (256 → 192)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x