import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_undirected

class GraphDataset():
    def __init__(self):
        super().__init__()
        self.montage = mne.channels.make_standard_montage("standard_1005")
        self.pos = self.montage.get_positions()["ch_pos"]
        self.pos_dict = {k.lower(): torch.tensor(v) for k,v in self.pos.items()}

    def create_graph(self, ch_names, k = 3, show_graph = False):

        node_features = []
        for ch in ch_names:
            if ch.lower() in self.pos_dict:
                node_features.append(self.pos_dict[ch])
            else:
                node_features.append(torch.tensor([0.001, 0.001, 0.001]))
        node_features = torch.stack(node_features, dim = 0)

        
        edge_index = self.geodesic_sphere_torch(k=k, x=node_features)
        g = self.return_graph(x = node_features, edge_index=edge_index)
        if show_graph:
            self.show_graph(g=g)
        return g


    def geodesic_sphere_torch(self, k, x):
        x = x / x.norm(dim=1, keepdim=True)
        cos = x @ x.T
        cos = torch.clamp(cos, -1.0, 1.0)
        D = torch.acos(cos)
        D.fill_diagonal_(float("inf"))
       
        idx = D.topk(k, largest=False).indices
        
        idx = idx.view(-1).unsqueeze(0)
        source = torch.arange(x.shape[0]).repeat_interleave(k).unsqueeze(0)
        edge_index = torch.cat([source, idx], dim = 0)
        return to_undirected(edge_index)
    
    def return_graph(self, x, edge_index):
       
        graph = Data(x = x, edge_index=edge_index, pos=x)
        return graph

    def show_graph(self, g):
        G = to_networkx(g, to_undirected=True)
        coords = g.pos.detach().cpu().numpy()
        pos = {i: coords[i, :2] for i in range(coords.shape[0])}

        plt.figure(figsize=(6, 6))
        nx.draw(G, pos=pos, with_labels=True, node_size=250, font_size=7)
        plt.axis("equal")
        plt.show()
           



if __name__ == "__main__":
    
    data = GraphDataset()

    