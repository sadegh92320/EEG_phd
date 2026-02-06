import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,dilation=dilation, groups=in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation, groups=in_channels)
        self.downsample = nn.Conv1d(in_channels, in_channels, 1,groups=in_channels) if in_channels != out_channels else None

    def forward(self, x):
        # Compute left padding so conv remains causal (no lookahead)
        pad = (self.kernel_size - 1) * self.dilation

        # First causal convolution (pad only on left)
        out = F.pad(x, (pad, 0))
        out = F.relu(self.conv1(out))

        # Second causal convolution
        out = F.pad(out, (pad, 0))
        out = self.conv2(out)

        # Add residual (identity) connection
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class ConvSequence(nn.Module):
    def __init__(self, in_channels, kernel_size, norm = "batchnorm"):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, 3*in_channels, kernel_size=kernel_size, groups=in_channels, padding="same")
        self.conv2 = nn.Conv1d(3*in_channels, in_channels, kernel_size=1, groups=in_channels,padding="same")

        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(3*in_channels)
            self.norm2 = nn.BatchNorm1d(in_channels)
        else:
            self.norm = nn.LayerNorm(3*in_channels)
            self.norm2 = nn.LayerNorm(in_channels)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.relu(x)


class SelfAttention(nn.Module):
    def __init__(self, d_in = 1,d_out=3):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
       

        self.W_K = nn.Linear(d_in, d_out)
        self.W_Q = nn.Linear(d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)

    def forward(self, x):
        K = self.W_K(x)
        Q = self.W_Q(x)
        V = self.W_V(x)

        sim = (torch.bmm(Q, K.transpose(2,1)))/math.sqrt(self.d_out)
        score = torch.softmax(sim, dim=-1)
        out = torch.bmm(score, V)
        return out, score

class AttentionPerChannel(nn.Module):
    def __init__(self, d_module, d_out):
        super().__init__()
        
        self.emb = nn.Linear(1, d_module)
        self.att = SelfAttention(d_module, d_out)
    
    def forward(self, x):
        B, C, T = x.shape
        x = x.view(B*C, T,1)
        x = self.emb(x)
        x, att = self.att(x)
        x = x.view(B,C,T,-1)
        att = att.view(B,C,T,T)

        return x, att


        

        


class CNNmodule(nn.Module):
    def __init__(self, in_channels, num_conv, reduction = 4, pool_size = 16, final_size_CNN = 30):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.ModuleList([ConvSequence(in_channels=in_channels, kernel_size=(2*i)+1) for i in range(num_conv)])
        self.tcn = nn.ModuleList([ResidualBlock(in_channels, in_channels, kernel_size=3, dilation=i+1) for i in range(num_conv)])
        
        self.reduction = nn.Sequential(
            nn.Linear(3840, 1000, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500, bias=False),
            
        )

        sequence = [ResidualBlock(14, 14, 3, 2), nn.BatchNorm1d(in_channels)]
        self.tcns = nn.ModuleList()
        for _ in range(3):
            self.tcns.extend(sequence)

        self.output_pool = nn.AdaptiveAvgPool1d(final_size_CNN)

        self.gcn_conv = nn.ModuleList([GCNConv(final_size_CNN, 200), GCNConv(200, 100)])
        self.fc = nn.Linear(final_size_CNN * 14, 3)




    def forward(self, x):
       
        x = x.permute(0,2,1).float()
        result = []

        for conv in self.conv:
            x1 = conv(x)
            result.append(x1)
    
        cat = torch.stack(result, dim=1)
       
        #To reduce dimension can think about pooling here like the paper
        red = self.reduction(cat)

        #instead of mean we can try to flatten and do linear
        att = torch.softmax(red.mean(dim = -1), dim = 1)
       
        weight = att.unsqueeze(dim = -1)

        r = cat * weight
        result = cat.sum(dim = 1)
        #Next need to do pooling and TCN

            
        out = self.output_pool(result)

        for tcn in self.tcns:
            out = tcn(out)

       
        
        
        
       

        #graph = build_data(out)
        out = out.view(-1, out.shape[1] * out.shape[2])
        return self.fc(out)
        #for conv in self.gcn_conv:
        #    graph = conv(graph.x, graph.edge_index, graph.batch)



    

import torch
import torch.nn as nn
import torch.nn.functional as F


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


        


if __name__ == "__main__":
    inp = torch.rand(1, 4, 14)
    inp = inp.permute(0,2,1)
    for _ in range(inp.shape[1]):
        model = SelfAttention()
        i = (inp[:,_,:])
        i = i.unsqueeze(dim = -1)
        model(i)
    #module = nn.Linear(1,3)
    #model = CNNmodule(in_channels=14, num_conv=3)
    #out = model(inp)
  

        




            


        
    
