import torch.nn as nn
import torch
import numpy as np
from MAE_pretraining.pretraining import PatchEEG
from MAE_pretraining.pretraining import TemporalEncoding
from einops import rearrange


class Downstream(nn.Module):
    """Basic downstream model"""
    def __init__(self, encoder, temporal_embedding, channel_embedding, class_token, path_eeg,\
                enc_dim, num_classes, aggregation = "class", use_rope = True):
        super().__init__()

        #Define all pretrained layer 
        self.encoder = encoder 
        self.temporal_embedding = temporal_embedding
        self.channel_embedding = channel_embedding
        self.patch = path_eeg
        self.class_token = class_token

        #Freeze all pretrained layers
        for layer in [self.encoder, self.temporal_embedding, self.channel_embedding, self.patch]:
            for p in layer.parameters():
                p.requires_grad = False
        self.class_token.requires_grad = False
        
        #Define linear probing layers
        self.fc = nn.Linear(enc_dim, num_classes)
        self.aggregration = aggregation

        self.use_rope = use_rope

    def forward(self, x):
        B, C, T = x.shape
        device = x.device
        
        x = self.patch(x)
        N = x.shape[1]

        #Add channel en temporal embedding 
        chan_embedding = self.channel_embedding(torch.arange(0,C, device=device))
        chan_embedding = rearrange(chan_embedding, "c d -> 1 1 c d")
        x = x + chan_embedding 


        class_token = self.class_token
        #Add temporal embedding if rotary embedding are not used
        if not self.use_rope:
            temp_embedding = self.temporal_embedding(seq_length = N, num_channel = C)
            temp_embedding = rearrange(temp_embedding, "b (n c) d -> b n c d", c = C)
            class_token += self.temporal_embedding.get_class_token()
            x += temp_embedding

        x = rearrange(x, "b n c d -> b (n c) d")

        #Concat the class token and pass the input through the transformer layers
        class_token = class_token.expand(B, 1, -1)
        x = torch.concat([class_token, x], dim = 1)
        with torch.no_grad():
            for transformer in self.encoder:
                x = transformer(x)
        
        #Extract class token pass through fully connected
        class_token, x = x[:,:1,:], x[:,1:,:]
        if self.aggregration == "class":
            out = class_token.view(B,-1)
        if self.aggregration == "mean":
            out = x.mean(dim = 1)
        return self.fc(out)