import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on earth in meters."""
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r * 1000 # returns meters

def load_geolife_data(dataset_path: str, max_files=200):
    all_data = []
    files_processed = 0
    
    # Traverse Geolife nested folders
    for root, _, files in os.walk(dataset_path):
        plt_files = [f for f in files if f.endswith('.plt')]
        for file in plt_files:
            if files_processed >= max_files:
                break
                
            file_path = os.path.join(root, file)
            try:
                # GeoLife PLT format: lat, lon, 0, alt, days, date, time
                # First 6 lines are header info
                df = pd.read_csv(file_path, skiprows=6, header=None, 
                                 names=['lat', 'lon', 'zero', 'alt', 'days', 'date', 'time'])
                
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # Compute Speed using Haversine
                lat_shift = df['lat'].shift(1)
                lon_shift = df['lon'].shift(1)
                time_shift = df['datetime'].shift(1)
                
                # distance in meters, time in seconds
                df['dist_m'] = df.apply(lambda row: haversine(row['lat'], row['lon'], 
                                        lat_shift[row.name], lon_shift[row.name]) if pd.notnull(lat_shift[row.name]) else 0.0, axis=1)
                
                df['dt_sec'] = (df['datetime'] - time_shift).dt.total_seconds().fillna(0)
                
                # compute speed (meters/second)
                df['speed'] = np.where(df['dt_sec'] > 0, df['dist_m'] / df['dt_sec'], 0.0)
                
                all_data.append(df[['datetime', 'lat', 'lon', 'speed']])
                files_processed += 1
                
            except Exception as e:
                continue
                
        if files_processed >= max_files:
            break
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Filter: remove low-speed (walking). Speed > 5 m/s
    final_df = final_df[final_df['speed'] > 5.0].reset_index(drop=True)
    return final_df

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=20):
        self.seq_len = seq_len
        self.features = ['lat', 'lon', 'speed']
        
        self.scaler = StandardScaler()
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        self.sequences = []
        values = df[self.features].values
        
        # Build rolling windows 
        # (Technically should split by discrete rides, but for anomaly 
        # modeling on prefiltered contiguous subsets, this suffices)
        if len(values) >= seq_len:
            for i in range(len(values) - seq_len + 1):
                self.sequences.append(values[i:i+seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

if __name__ == '__main__':
    print("Pre-processing utilities ready for Geolife Trajectory parsing.")
