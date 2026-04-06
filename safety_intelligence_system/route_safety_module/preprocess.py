import osmnx as ox
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import math

def generate_crime_layer(graph, num_centers=5, spread=0.01):
    """
    Simulates a Gaussian spatial distribution of crime over the road network nodes.
    """
    # Extract node coordinates
    nodes = list(graph.nodes(data=True))
    lats = np.array([data.get('y', 0.0) for node, data in nodes])
    lons = np.array([data.get('x', 0.0) for node, data in nodes])
    
    # Pick random crime epicenters
    idx_centers = np.random.choice(len(nodes), num_centers, replace=False)
    center_lats = lats[idx_centers]
    center_lons = lons[idx_centers]
    
    crime_scores = np.zeros(len(nodes))
    
    for i in range(len(nodes)):
        # Calculate gaussian decay distance to the nearest epicenter
        dist_sq = (lats[i] - center_lats)**2 + (lons[i] - center_lons)**2
        max_crime = np.max(np.exp(-dist_sq / (2 * spread**2)))
        crime_scores[i] = max_crime
        
    return crime_scores

def process_graph(center_point=(28.6139, 77.2090), dist=3000):
    print(f"Downloading street network for Central Delhi ({dist}m radius)...")
    try:
        # We use a central point with a specific radius to ensure a solid and precise graph 
        # is pulled without relying on vague map boundary polygons.
        G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
    except Exception as e:
        print(f"Failed to extract place via OSMnx: {e}")
        return None, None, None
    print("Graph downloaded. Constructing edge features...")
    
    # 1. Simulate crime on nodes, then broadcast to edges
    crime_scores = generate_crime_layer(G)
    for i, (node, data) in enumerate(G.nodes(data=True)):
        data['node_crime'] = crime_scores[i]
        
    # 2. Iterate edges: Build features & compute labels
    # Label formula: safety = 0.5*crime + 0.3*congestion + 0.2*road_weight
    edge_index = []
    edge_attr = []
    edge_labels = []
    
    # Map raw road types to numeric weights
    road_types_map = {
        'motorway': 1.0, 'trunk': 0.9, 'primary': 0.8,
        'secondary': 0.6, 'tertiary': 0.4, 'residential': 0.2, 'unclassified': 0.1
    }
    
    nodes_list = list(G.nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    
    for u, v, data in G.edges(data=True):
        # Edge source / dest crime average
        u_crime = G.nodes[u].get('node_crime', 0)
        v_crime = G.nodes[v].get('node_crime', 0)
        edge_crime = (u_crime + v_crime) / 2.0
        
        # Road length (normalized roughly)
        length = float(data.get('length', 10.0))
        length_feat = min(length / 200.0, 1.0) 
        
        # Road Type weighting
        hw_type = data.get('highway', 'unclassified')
        if isinstance(hw_type, list): hw_type = hw_type[0]
        road_weight = road_types_map.get(hw_type, 0.1)
        
        # Simulated Congestion Context (0 to 1)
        congestion = np.random.uniform(0.1, 0.9)
        
        # Calculate Risk/Safety Score 
        # (Since 1=crime, 1=congestion, high score = UNSAFE)
        risk_score = 0.5 * edge_crime + 0.3 * congestion + 0.2 * (1.0 - road_weight)
        
        feature_vector = [edge_crime, length_feat, road_weight, congestion]
        
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_attr.append(feature_vector)
        edge_labels.append(risk_score)
        
    # Build PyTorch Geometric Data format
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
    
    # Dummy node features (just using ones since we focus on edge regression)
    x = torch.ones((len(nodes_list), 1), dtype=torch.float)
    
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)
    return pyg_data, G, node_to_idx

if __name__ == "__main__":
    data, G, _ = process_graph()
    if data:
        print(f"Generated PyG Graph: Nodes={data.num_nodes}, Edges={data.num_edges}")
        # Save processed dataset
        torch.save(data, "delhi_graph_data.pt")
        print("Dataset saved to delhi_graph_data.pt")
