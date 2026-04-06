import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from preprocess import build_context_features, ContextSequenceDataset
from model import TemporalAttentionTransformer

def train_context_model(epochs=15, batch_size=32, lr=1e-3, seq_len=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading Context Data via KaggleHub...")
    try:
        # Load Ride Safety Dataset of Mumbai and Delhi
        path = kagglehub.dataset_download("advaitasen/ride-safety-dataset-of-mumbai-and-delhi")
        import os
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            print("No CSV found in the downloaded Kaggle dataset.")
            return
        df = pd.read_csv(os.path.join(path, csv_files[0]))
    except Exception as e:
        print(f"Error loading Context kaggle dataset: {e}")
        return

    print("Data Downloaded. Processing Temporal Engineering...")
    df = build_context_features(df)
    
    dataset = ContextSequenceDataset(df, seq_len=seq_len)
    
    # 80 / 20 Train Validation Split using sequential order (no shuffling across time)
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TemporalAttentionTransformer(num_features=6).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Task 3 Model Training...")
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
            
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_seq, batch_labels in val_loader:
                batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
                predictions = model(batch_seq)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        if len(set(all_labels)) > 1:
            val_roc_auc = roc_auc_score(all_labels, all_preds)
            val_pr_auc = average_precision_score(all_labels, all_preds)
        else:
            val_roc_auc, val_pr_auc = 0.0, 0.0
            
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val ROC-AUC: {val_roc_auc:.4f}")

    torch.save(model.state_dict(), "context_risk_model.pth")
    print("Optimization Complete. Model saved successfully as context_risk_model.pth.")

if __name__ == '__main__':
    train_context_model()
