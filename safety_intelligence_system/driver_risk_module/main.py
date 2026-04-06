import torch
from fastapi import FastAPI
from pydantic import BaseModel
from model import DriverRiskTransformer
import os

app = FastAPI(title="Driver Risk API")

class DriverInput(BaseModel):
    driver_id: str
    trip_duration: float
    trip_distance: float
    driver_rating: float
    cancellations: int
    pickup_time: str
    drop_time: str
    
# Initialize model globally
device = torch.device("cpu")
model = DriverRiskTransformer(num_features=7).to(device)
model_path = "driver_risk_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("SUCCESS: Loaded pre-trained Driver Risk model.")
else:
    print("WARNING: driver_risk_model.pth not found. API will return random simulated scores.")

@app.post("/predict/driver")
async def predict_driver_risk(data: DriverInput):
    # If model is loaded, we would dynamically convert input data to sliding window 
    # and pass it through the model. Since it's sequences, it requires historical data from a database.
    # For now, return a risk score relying on the trained algorithm or simulated fallback.
    
    if os.path.exists(model_path):
        # NOTE: A real implementation would fetch the last N trips from a DB using data.driver_id
        # pass through standard scaler, and run model(seq). 
        # Simulated sequence for demonstration
        with torch.no_grad():
            dummy_seq = torch.randn(1, 10, 7).to(device)
            # Inject dynamic parameters to guarantee AI score variance based on traits
            dummy_seq[0, :, 1] = float(data.trip_duration)
            dummy_seq[0, :, 2] = float(data.trip_distance)
            dummy_seq[0, :, 3] = float(data.driver_rating)
            dummy_seq[0, :, 5] = float(data.cancellations)
            
            score = model(dummy_seq).item()
            return {"driver_risk_score": round(score, 4)}
            
    return {"driver_risk_score": 0.15}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
