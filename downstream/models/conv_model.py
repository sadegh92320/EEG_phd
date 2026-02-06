import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEEG(nn.Module):
    def __init__(self, C = 14, T= 784, num_classes=3):
        super().__init__()
        self.conv = nn.Conv1d(C, 32, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.fc = nn.Linear(32 * 32, num_classes)

    def forward(self, x):     # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = F.relu(self.conv(x))
        x = self.pool(x)        # (B, 32, 32)
        x = x.flatten(1)        # (B, 32*32)
        return self.fc(x)


class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet (Lawhern et al., 2018)
    
    Expected input:
        - x: (B, 1, C, T)  or  (B, C, T)  where
            C = num_channels, T = num_samples
    
    Args:
        num_classes   : number of classes.
        num_channels  : number of EEG channels (C).
        num_samples   : number of time samples (T).
        F1            : number of temporal filters.
        D             : depth multiplier (for depthwise conv).
        kernel_length : temporal kernel length (original paper often uses 64).
        dropout       : dropout rate.
    """
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_samples: int,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D

        # ------------------------------------------------------------------
        # Block 1: Temporal convolution
        # ------------------------------------------------------------------
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            stride=1,
            padding=(0, kernel_length // 2),  # "same" padding in time
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ------------------------------------------------------------------
        # Block 1 (cont’d): Depthwise spatial convolution
        # ------------------------------------------------------------------
        self.conv_depthwise = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(num_channels, 1),
            stride=1,
            groups=F1,          # depthwise over F1 filters
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Block 2: Separable convolution (depthwise + pointwise)
        # ------------------------------------------------------------------
        # depthwise (in time)
        self.conv_separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            stride=1,
            padding=(0, 16 // 2),
            groups=F1 * D,   # depthwise
            bias=False,
        )
        # pointwise
        self.conv_separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Classification layer
        # ------------------------------------------------------------------
        # After pool1 (1,4) and pool2 (1,8) we downsample time by factor 32
        # T_out = floor(num_samples / 32)
        self.t_out = num_samples // 32
        self.classifier = nn.Linear(368, num_classes)

    def forward(self, x):
       
        

        # Accept (B, C, T) or (B, 1, C, T)
        if x.dim() == 3:
            # (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # assume it's (B, 1, C, T)
            pass
        else:
            raise ValueError(f"Expected input shape (B,C,T) or (B,1,C,T), got {x.shape}")

        # Block 1: temporal conv
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = F.elu(x)

        # Block 1: depthwise spatial conv
        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2: separable conv
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten and classify
        # x: (B, F2, 1, T_out) -> (B, F2 * T_out)
        x = x.squeeze(2)           # remove channel dimension (C -> 1)
        x = x.reshape(x.size(0), -1)
        logits = self.classifier(x)
        return logits


        

        




            


        
    
