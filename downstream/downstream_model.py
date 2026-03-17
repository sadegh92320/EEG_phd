import torch.nn as nn
import torch
import numpy as np
from MAE_pretraining.pretraining import PatchEEG
from MAE_pretraining.pretraining import TemporalEncoding
from einops import rearrange
from MAE_pretraining.graph_embedding import GraphDataset
from MAE_pretraining.gnn import GATModel
import yaml
from torch_geometric.data import Data


class Downstream(nn.Module):
    """Basic downstream model"""
    def __init__(self, encoder, temporal_embedding, channel_embedding, class_token, path_eeg,\
                enc_dim, num_classes, norm_enc = nn.LayerNorm,gnn = None,aggregation = "class", use_rope = False, use_graph = False):
        super().__init__()

        #Define all pretrained layer 
        self.encoder = encoder 
        self.temporal_embedding = temporal_embedding
        self.channel_embedding = channel_embedding
        self.patch = path_eeg
        self.norm_enc = norm_enc(enc_dim)
        self.class_token = class_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Define graph modules
        #Def graph embeddings variable
        with open("MAE_pretraining/info_dataset/channel_info_red.yaml") as f:
            config = yaml.safe_load(f)
        ch_total = config["channels_mapping"]

        ordered_channels = [k for k, v in sorted(ch_total.items(), key=lambda item: item[1])]
        gnn_data = GraphDataset()
        g = gnn_data.create_graph(ch_names=ordered_channels, radius=0.4)
        self.register_buffer("g_x", g.x)
        self.register_buffer("g_edge_index", g.edge_index)
        self.use_graph = use_graph
        self.gnn_enc = gnn

        #Freeze all pretrained layers
        for layer in [self.encoder, self.temporal_embedding, self.channel_embedding, self.patch]:
            for p in layer.parameters():
                p.requires_grad = False
        self.class_token.requires_grad = False
        
        #Define linear probing layers
        self.fc = nn.Linear(enc_dim, num_classes)
        self.aggregration = aggregation

        self.use_rope = use_rope

    def forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device
        
        x = self.patch(x)
        N = x.shape[1]
        x = rearrange(x, "b n c d -> b (n c) d")
        L = x.shape[1]

        # Correct Channel Embedding
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)

        if not self.use_graph:
            # (B, C) -> (B, 1, C) -> (B, N, C) -> (B, N*C)
            chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
            #(B,N*C,enc_dim)
            chan_embedding = self.channel_embedding(chan_id)
        else:
            #Output of size (B,chan_total, enc)
            g_device = Data(x=self.g_x, edge_index=self.g_edge_index)
            chan_total = self.gnn_enc(g_device)
            chan_total = chan_total[channel_list]

            #(B,1,C,enc)
            chan_embedding = chan_total.unsqueeze(1).repeat(1,N,1,1).view(B,L,-1)
        x = x + chan_embedding 

        # Temporal Embedding
        if not self.use_rope:
            seq_idx  = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
            x = x + self.temporal_embedding(seq_idx)

        # Correct Class Token Expansion
        class_token = self.class_token + self.temporal_embedding.get_class_token().view(1,1,-1).to(device)
        class_token = class_token.expand(B, 1, -1) # Fixed Crash Here
        
        x = torch.concat([class_token, x], dim = 1)
        
        # Removed no_grad block for fine-tuning flexibility
        for transformer in self.encoder:
            x = transformer(x)

        x = self.norm_enc(x)
        
        class_token, x = x[:,:1,:], x[:,1:,:]
        
        if self.aggregration == "class":
            out = class_token.squeeze(1) # Slightly cleaner than view(B, -1)
        elif self.aggregration == "mean":
            out = x.mean(dim = 1)
            
        return self.fc(out)