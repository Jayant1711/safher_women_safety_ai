import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    # Filter for Delhi if text column exists
    for col in df.columns:
        if 'city' in col.lower() or 'location' in col.lower():
            df = df[df[col].astype(str).str.contains('Delhi', case=False, na=False)]
            break
            
    # Assuming timestamp column exists, else dummy create it if missing for some reason
    if 'timestamp' in df.columns:
         df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
         # Fallback chronological timestamp if missing
         df['timestamp'] = pd.date_range(start='2025-01-01', periods=len(df), freq='h')

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['hour_of_day'] = df['timestamp'].dt.hour
    
    # Derivations
    df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if x >= 21 or x <= 5 else 0)
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 8 and x <= 11) or (x >= 17 and x <= 20) else 0)
    
    # Interaction Feature
    if 'congestion_level' in df.columns:
        df['night_congestion'] = df['is_night'] * df['congestion_level']
    else:
        # Fallback dummy 
        df['congestion_level'] = np.random.uniform(0.1, 0.9, len(df))
        df['night_congestion'] = df['is_night'] * df['congestion_level']
        
    if 'accident_count' not in df.columns:
        df['accident_count'] = np.random.choice([0, 1, 2], len(df), p=[0.9, 0.08, 0.02])
        
    if 'average_speed' not in df.columns:
        df['average_speed'] = 50 - (df['congestion_level'] * 30)

    # Label Definition: risk = 1 if accident OR (high congestion AND night)
    # Threshold for high congestion: e.g. 0.7
    cond_accident = df['accident_count'] > 0
    cond_night_cong = (df['congestion_level'] > 0.7) & (df['is_night'] == 1)
    df['risk_label'] = (cond_accident | cond_night_cong).astype(int)
    
    return df

class ContextSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=10):
        self.seq_len = seq_len
        self.features = ['hour_of_day', 'is_night', 'is_peak_hour', 'congestion_level', 
                         'night_congestion', 'average_speed']
        
        # Scale continuous/categorical features appropriately
        self.scaler = StandardScaler()
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        self.sequences = []
        self.labels = []
        
        values = df[self.features].values
        labels = df['risk_label'].values
        
        # Global timeline sliding window (since context is global per city)
        if len(values) >= seq_len:
            for i in range(len(values) - seq_len + 1):
                self.sequences.append(values[i:i+seq_len])
                # We predict the risk of the next immediate timeframe or the last step
                self.labels.append(labels[i+seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

if __name__ == '__main__':
    print("Preprocessing structures ready for Context Risk.")
