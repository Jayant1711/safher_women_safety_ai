import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

from model import RouteSafetyGNN
from preprocess import process_graph

def train_model(epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load or Build Dataset
    data_path = "delhi_graph_data.pt"
    if os.path.exists(data_path):
        print("Loading cached graph database...")
        data = torch.load(data_path, weights_only=False).to(device)
    else:
        print("Building graph database from OSMNx...")
        data, G, _ = process_graph()
        if data is None: return
        torch.save(data, data_path)
        data = data.to(device)
        
    print(f"Graph loaded with {data.num_nodes} nodes and {data.edge_index.size(1)} edges.")
    
    # Generate mock Train/Val edge mask (80/20)
    num_edges = data.edge_index.size(1)
    indices = torch.randperm(num_edges)
    train_idx = indices[:int(0.8 * num_edges)]
    val_idx = indices[int(0.8 * num_edges):]
    
    # 2. Init Model
    model = RouteSafetyGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    
    print("Starting GNN Training Loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (computes over all edges)
        predictions = model(data.x, data.edge_index, data.edge_attr)
        
        # Compute loss ONLY on training edges
        loss = criterion(predictions[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        # Evaluation Phase
        model.eval()
        with torch.no_grad():
            val_preds = predictions[val_idx].cpu().numpy()
            val_labels = data.y[val_idx].cpu().numpy()
            
            rmse = mean_squared_error(val_labels, val_preds) ** 0.5
            mae = mean_absolute_error(val_labels, val_preds)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss(MSE): {loss.item():.4f} | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f}")

    # Save weights
    torch.save(model.state_dict(), "route_gnn_model.pth")
    print("Training Complete. Model securely mapped over edges and saved as route_gnn_model.pth")

if __name__ == '__main__':
    train_model(epochs=30)
