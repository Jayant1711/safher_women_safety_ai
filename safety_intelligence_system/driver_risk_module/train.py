import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as pd
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import kagglehub
from kagglehub import KaggleDatasetAdapter

from model import DriverRiskTransformer
from preprocess import engineer_features, DriverSequenceDataset

def train_model(epochs=10, batch_size=32, lr=1e-3, seq_len=2):
    # 1. Load Data
    print("Loading data via kagglehub...")
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "yashdevladdha/uber-ride-analytics-dashboard",
            "ncr_ride_bookings.csv"
        )
        
        # Column Mapping for this specific dataset
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['driver_id'] = df['Customer ID'].astype(str) # Proxy for driver ID
        df['driver_rating'] = df['Driver Ratings'].fillna(3.5)
        df['trip_duration'] = df['Avg CTAT'].fillna(15.0)
        df['cancellations'] = df['Cancelled Rides by Driver'].fillna(0).astype(int)
        df['trip_distance'] = df['Ride Distance'].fillna(5.0)
        df['pickup_time'] = df['timestamp']
        
        print("Successfully loaded and mapped dataset:")
        print(df.head())
    except Exception as e:
         print(f"Error loading dataset from Kaggle: {e}")
         return
        
    df = engineer_features(df)
    
    # 2. Dataset & Dataloader
    dataset = DriverSequenceDataset(df, seq_len=seq_len)
    
    # Split Train/Val (85% / 15%)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model Definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DriverRiskTransformer(num_features=7).to(device)
    
    # BCE Loss for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_seq, batch_labels in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_seq)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_seq, batch_labels in val_loader:
                 batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
                 predictions = model(batch_seq)
                 loss = criterion(predictions, batch_labels)
                 val_loss += loss.item()
                 
                 all_preds.extend(predictions.cpu().numpy())
                 all_labels.extend(batch_labels.cpu().numpy())
                 
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            val_roc_auc = roc_auc_score(all_labels, all_preds)
            val_pr_auc = average_precision_score(all_labels, all_preds)
            val_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
            val_f1 = f1_score(all_labels, val_preds_binary)
        else:
            val_roc_auc, val_pr_auc, val_f1 = 0, 0, 0
                 
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f} | F1: {val_f1:.4f}")

    # Save weights
    torch.save(model.state_dict(), "driver_risk_model.pth")
    print("Training Complete. Model saved as driver_risk_model.pth")

if __name__ == '__main__':
    # Usage: Automatically downloads dataset using Kaggle API
    train_model(epochs=5)
