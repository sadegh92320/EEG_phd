
import torch
import torch.nn as nn

REGION_MAP = {
    "frontal": [
        "fp1", "fp2", "fpz", "af3", "af4", "af7", "af8", "afz",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "fz",
        "fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "fcz"
    ],
    "central": [
        "c1", "c2", "c3", "c4", "c5", "c6", "cz",
        "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cpz"
    ],
    "temporal": [
        "ft7", "ft8", "ft9", "ft10", 
        "t7", "t8", "t3", "t4", # T3/T4 are older names for T7/T8
        "tp7", "tp8", "tp9", "tp10", "t5", "t6" # T5/T6 are older names for P7/P8 (often grouped here or parietal)
    ],
    "parietal": [
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pz",
        "po3", "po4", "po7", "po8", "poz"
    ],
    "occipital": [
        "o1", "o2", "oz", "iz"
    ]
}

class BrainMapper:
    """Class that creates the adjaceny matrix for each montage"""
    def __init__(self):
        self.region_map = REGION_MAP
        self.region_name = list(self.region_map.keys())
        self.num_regions = len(self.region_name)
        self.region_id = {name:idx for idx, name in enumerate(self.region_name)}
        self.channel_to_region = {}
        

        for region, channel in self.region_map.items():
            for ch in channel:
                self.channel_to_region[ch] = self.region_id[region]

    def get_region_name(self, channel_name):
        """
            Get region name to which the channel belongs to. If 
            the channel is unknown returns unknown.
        """
        idx = self.channel_to_region.get(channel_name.lower())
        if idx is not None:
            return self.region_name[idx]
        return "unknown"

    def build_adjacency_matrix(self, channel_list, device):
        """Build the adjaceny matrix of the hypergraph for one or several montage."""

        #Check if the channel list is only one montage or a batch of montages
        if isinstance(channel_list[0], str):
            #Initialize the adjacency matrix
            adj = torch.zeros((len(channel_list), self.num_regions), device=device)
            #Attribute each channel to its corresponding brain region
            for i, ch in enumerate(channel_list):
                ch = ch.lower()
                if ch in self.channel_to_region:
                    reg = self.channel_to_region[ch]
                    adj[i,reg] = 1.0
        else:
            #Here it's for a graph so the same operation is done but for each montage in batch
            num_c = [len(ch) for ch in channel_list]
            adj = torch.zeros((len(channel_list), max(num_c), self.num_regions), device=device)
            for b in range(len(channel_list)):
                for i, ch in enumerate(channel_list[b]):
                    ch = ch.lower()
                    if ch in self.channel_to_region:
                        reg = self.channel_to_region[ch]
                        adj[b,i,reg] = 1.0
        #Normalize for each region
        if adj.dim() == 2:
            adj = adj/(torch.sum(adj, dim=0)).clamp(min=1)
        else:
            adj = adj/(torch.sum(adj,dim=1).unsqueeze(1)).clamp(min=1)
        return adj

class RegionToken(nn.Module):
    def __init__(self, in_features, out_features, use_bias = True):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
    
    def forward(self,x, adj):

        #Input the x of size B,N,C,T and return the new size B,C,N,H
        x = self.linear(x)

        #Multiply the adjacency matrix with our signal, sum the resulting embedding
        if adj.dim() == 2:
            reg_token = torch.einsum("bnct,cr -> bnrt", x, adj)
        else:
            reg_token = torch.einsum("bnct,bcr -> bnrt", x, adj)

        return self.norm(self.activation(reg_token))





if __name__ == "__main__":
    mapper = BrainMapper()
    print(mapper.region_id)
    print(mapper.channel_to_region)
    print(mapper.region_name)