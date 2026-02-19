import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
import torch.nn.functional as F
from MAE_pretraining.graph_embedding import GraphDataset


class GATModel(nn.Module):
    def __init__(self, in_features = 3, hidden = 128, enc_dim = 768, num_head = 1, dropout = 0.0):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=in_features, out_channels=hidden, heads=num_head, dropout=dropout)
        self.bn1 = nn.LayerNorm(hidden*num_head)     #2
        self.conv2 = GATv2Conv(in_channels=hidden * num_head, out_channels=enc_dim, concat=False, dropout=dropout)
        self.dropout = dropout
        self.norm = nn.LayerNorm(enc_dim)
        self.fc = nn.Linear(in_features, enc_dim)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        res = self.fc(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = x + res
        return self.norm(x)
    
if __name__ == "__main__":
    graph = GraphDataset()
    


