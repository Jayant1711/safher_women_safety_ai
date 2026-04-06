import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import numpy as np
from model import LSTMAutoencoder

app = FastAPI(title="Anomaly Detection API")

class TrajectoryPoint(BaseModel):
    lat: float
    lon: float
    speed: float

class AnomalyInput(BaseModel):
    trajectory_window: List[TrajectoryPoint] # e.g. last 20 steps

device = torch.device('cpu')
model = LSTMAutoencoder(input_dim=3)
model_path = "anomaly_detector.pth"
anomaly_threshold = 0.5

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    anomaly_threshold = checkpoint.get('anomaly_threshold', 0.5)
    print(f"SUCCESS: Loaded Autoencoder. MSE Threshold: {anomaly_threshold:.4f}")
else:
    print("WARNING: anomaly_detector.pth not found. Using simulated threshold defaults.")

@app.post("/monitor/anomaly")
async def predict_anomaly(data: AnomalyInput):
    # If the model is loaded and real data is provided:
    if os.path.exists(model_path) and len(data.trajectory_window) == 20:
        # Convert List[TrajectoryPoint] -> Tensor (1, 20, 3)
        seq = [[p.lat, p.lon, p.speed] for p in data.trajectory_window]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            reconstructed = model(seq_tensor)
            mse_loss = torch.mean((reconstructed - seq_tensor)**2).item()
            
            return {
                "anomaly_score": round(mse_loss, 5),
                "is_anomaly": mse_loss > anomaly_threshold
            }
            
    # Dummy score if no data or no model is present
    return {
        "anomaly_score": 0.05,
        "is_anomaly": False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
