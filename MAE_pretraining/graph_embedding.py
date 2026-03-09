import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_networkx
import mne
import networkx as nx
import matplotlib.pyplot as plt
import yaml

class GraphDataset():
    def __init__(self):
        super().__init__()
        self.montage = mne.channels.make_standard_montage("standard_1005")
        self.pos = self.montage.get_positions()["ch_pos"]
        self.pos_dict = {k.lower(): torch.tensor(v, dtype=torch.float32) for k, v in self.pos.items()}
        self.created_graph = {}

    def create_graph(self, ch_names, radius, show_graph=False):
        node_features = []
        # Fix: Convert to tuple for dictionary hashing
        lower_ch = tuple([ch.lower() for ch in ch_names])
        
        # Fix: Reference the correct dictionary attribute
        if lower_ch not in self.created_graph:
            for ch in ch_names:
                if ch.lower() in self.pos_dict:
                    node_features.append(self.pos_dict[ch.lower()])
                else:
                    # Explicit warning for geometry violation
                    print(f"Warning: Channel {ch} missing. Assigning arbitrary topology.")
                    # Distribute missing nodes or handle downstream to avoid k-NN clustering
                    node_features.append(torch.tensor([0.0, 0.0, 0.0])) 
                    
            node_features = torch.stack(node_features, dim=0)
            
            edge_index = self.geodesic_radius_graph(radius=radius, x=node_features)
            g = self.return_graph(x=node_features, edge_index=edge_index)
            
            # Cache the result
            self.created_graph[lower_ch] = g
        else:
            g = self.created_graph[lower_ch]
            
        if show_graph:
            self.show_graph(g=g)
        return g

    def geodesic_radius_graph(self, radius, x):
        """
        Create edge features based on a global geodesic distance threshold (epsilon-graph).
        This ensures anatomical neighbors connect regardless of local grid density.
        """
        # Normalize vectors to unit sphere
        norms = x.norm(dim=1, keepdim=True)
        norms[norms == 0] = 1.0 
        x_norm = x / norms
        
        # Compute geodesic distance matrix
        cos = x_norm @ x_norm.T
        cos = torch.clamp(cos, -1.0, 1.0)
        D = torch.acos(cos)
        D.fill_diagonal_(float("inf"))
       
        # Create edges for any pair within the global anatomical radius
        valid_edges = D <= radius
        
        # Convert boolean mask to edge_index format
        source, target = valid_edges.nonzero(as_tuple=True)
        edge_index = torch.stack([source, target], dim=0)
        
        return to_undirected(edge_index)
    
    def return_graph(self, x, edge_index):
        graph = Data(x=x, edge_index=edge_index, pos=x)
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
    with open("MAE_pretraining/info_dataset/channel_info.yaml") as f:
        config = yaml.safe_load(f)
    ch = config["channels_mapping"]
    data = GraphDataset()
    data.create_graph(ch_names=ch, show_graph=False, radius=0.4)

    