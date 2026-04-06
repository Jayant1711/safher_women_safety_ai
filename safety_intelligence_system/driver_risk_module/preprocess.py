import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Assuming df has: timestamp, driver_rating, trip_duration, cancellations, pickup_time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    
    # Sort for sequence generation
    df = df.sort_values(by=['driver_id', 'timestamp'])
    
    # Derived Feature: Night trip
    df['is_night_trip'] = df['pickup_time'].dt.hour.apply(lambda x: 1 if x >= 21 or x < 5 else 0)
    
    # Feature Engineering Aggregations (Expanding Window)
    df['rating_variance'] = df.groupby('driver_id')['driver_rating'].expanding().var().reset_index(0,drop=True).fillna(0)
    df['night_trip_ratio'] = df.groupby('driver_id')['is_night_trip'].expanding().mean().reset_index(0,drop=True)
    df['cancellation_rate'] = df.groupby('driver_id')['cancellations'].expanding().mean().reset_index(0,drop=True)
    df['avg_trip_duration'] = df.groupby('driver_id')['trip_duration'].expanding().mean().reset_index(0,drop=True)
    
    # Label Definition
    # risk = 1 if (rating <= 4.0 AND night_trip) OR high cancellations else 0
    df['high_cancellations'] = (df['cancellations'] > 1).astype(int)
    cond1 = (df['driver_rating'] <= 4.0) & (df['is_night_trip'] == 1)
    cond2 = df['high_cancellations'] == 1
    df['risk_label'] = (cond1 | cond2).astype(int)
    
    return df

class DriverSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=10):
        self.seq_len = seq_len
        self.features = ['trip_duration', 'trip_distance', 'driver_rating', 'rating_variance', 
                         'night_trip_ratio', 'cancellation_rate', 'avg_trip_duration']
        
        self.scaler = StandardScaler()
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        self.sequences = []
        self.labels = []
        
        # Group by driver and create sliding windows
        for driver_id, group in df.groupby('driver_id'):
            values = group[self.features].values
            labels = group['risk_label'].values
            
            if len(values) >= seq_len:
                for i in range(len(values) - seq_len + 1):
                    self.sequences.append(values[i:i+seq_len])
                    # Label is the risk of the last trip in the sequence
                    self.labels.append(labels[i+seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Dummy test entrypoint
if __name__ == "__main__":
    print("Preprocessing module ready.")
