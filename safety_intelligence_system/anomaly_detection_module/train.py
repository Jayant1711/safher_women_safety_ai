import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import kagglehub
import numpy as np
import os

from preprocess import load_geolife_data, TrajectoryDataset
from model import LSTMAutoencoder

def train_anomaly_model(epochs=15, batch_size=32, lr=1e-3, seq_len=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Downloading Geolife Trajectories via Kagglehub...")
    try:
        dataset_path = kagglehub.dataset_download("arashnic/microsoft-geolife-gps-trajectory-dataset")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return
        
    print(f"Processing GPS .plt files off {dataset_path} ...")
    # Limiting specifically to 300 files to ensure memory/time limits aren't hit locally.
    df = load_geolife_data(dataset_path, max_files=300)
    if df.empty:
        print("No valid driving trajectories found!")
        return
        
    print(f"Extracted {len(df)} active vehicle trajectory points.")
    
    dataset = TrajectoryDataset(df, seq_len=seq_len)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTMAutoencoder(input_dim=3).to(device)
    
    # Autoencoder operates on MSE reconstruction loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting LSTM Autoencoder Training Circuit...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_seq in train_loader:
            batch_seq = batch_seq.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_seq)
            loss = criterion(reconstructed, batch_seq)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seq in val_loader:
                batch_seq = batch_seq.to(device)
                reconstructed = model(batch_seq)
                loss = criterion(reconstructed, batch_seq)
                val_loss += loss.item()
                
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Reconstruction Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")

    # Determine acceptable reconstruction threshold
    print("Calculating nominal anomaly threshold from validation set...")
    val_losses = []
    with torch.no_grad():
         for batch_seq in val_loader:
             batch_seq = batch_seq.to(device)
             recon = model(batch_seq)
             # Compute loss per sequence in batch
             loss_per_seq = torch.mean((recon - batch_seq)**2, dim=[1,2])
             val_losses.extend(loss_per_seq.cpu().numpy())
             
    # E.g., 95th percentile error becomes boundary constraint
    threshold = float(np.percentile(val_losses, 95))
    print(f"Anomaly strict MSE threshold bound calculated at: {threshold:.6f}")
    
    # Pack model info
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'anomaly_threshold': threshold
    }

    torch.save(checkpoint, "anomaly_detector.pth")
    print("Training Complete. Model securely mapped over sequences and saved as anomaly_detector.pth")

if __name__ == '__main__':
    train_anomaly_model(epochs=10)
