import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RouteSafetyGNN(nn.Module):
    def __init__(self, node_in_channels=1, edge_in_channels=4, hidden_channels=32, heads=2):
        super(RouteSafetyGNN, self).__init__()
        # We use a GAT to learn rich node embeddings using local neighborhood
        self.conv1 = GATConv(node_in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        
        # MLP for Edge Scoring: inputs are (source_node_emb, dest_node_emb, edge_features)
        self.edge_mlp = nn.Sequential(
            nn.Linear((hidden_channels * 2) + edge_in_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Output risk score [0, 1]
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Message Passing to generate rich node embeddings
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        node_embs = self.conv2(out, edge_index)
        
        # 2. Extract source and destination node embeddings for every edge
        src, dst = edge_index
        emb_src = node_embs[src]
        emb_dst = node_embs[dst]
        
        # 3. Concatenate (Source, Destination, Edge Attributes) to predict edge safety/risk
        concat_features = torch.cat([emb_src, emb_dst, edge_attr], dim=1)
        
        # 4. Predict
        edge_scores = self.edge_mlp(concat_features)
        return edge_scores.squeeze()

if __name__ == "__main__":
    # Dummy run
    model = RouteSafetyGNN()
    x = torch.ones(10, 1) # 10 nodes
    edge_idx = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_feat = torch.rand(4, 4)
    out = model(x, edge_idx, edge_feat)
    print("Edge scores prediction shape:", out.shape)
