import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
import torch.nn.functional as F
from MAE_pretraining.graph_embedding import GraphDataset
import yaml


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
        print(res.shape)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = x + res
        return self.norm(x)
    
def get_chan_idx(channel_list, total):
        channel_id = []
        for ch in channel_list:
         
            idx = total.get(ch)
            if idx is None:
                raise ValueError(f"Channel '{ch}' not found in general channel_info.yaml mapping.")
            channel_id.append(idx)
        return channel_id
    
if __name__ == "__main__":
    with open("MAE_pretraining/info_dataset/channel_info_red.yaml") as f:
        config = yaml.safe_load(f)
    ch = config["channels_mapping"]

    with open("MAE_pretraining/info_dataset/auditory.yaml") as f:
        config_2 = yaml.safe_load(f)
    ch_get = config_2["channel_list"]
    graph = GraphDataset()

    ch_id = get_chan_idx(channel_list=ch_get, total=ch)
    print(ch_id)
    
    g = graph.create_graph(ch_names=ch, show_graph=False, radius=0.4)
    print(g.x.shape)
    
    model = GATModel()
    out = model(g)
    out = out[ch_id, :]



    print(out.shape)


    


