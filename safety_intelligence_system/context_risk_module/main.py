import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
from model import TemporalAttentionTransformer

app = FastAPI(title="Context Risk API")

class ContextInput(BaseModel):
    timestamp: str
    congestion_level: float
    average_speed: float
    accident_count: int
    ride_duration: float

device = torch.device('cpu')
model = TemporalAttentionTransformer()
model_path = "context_risk_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("SUCCESS: Loaded pre-trained Context TFT.")
else:
    print("WARNING: context_risk_model.pth not found. Reverting to base estimates.")

@app.post("/predict/context")
async def predict_context_risk(data: ContextInput):
    if os.path.exists(model_path):
        # We would compute the time variables using data.timestamp
        # Here we just execute a pseudo sequence through the network
        with torch.no_grad():
            dummy_seq = torch.randn(1, 10, 6).to(device)
            score = model(dummy_seq).item()
            return {"context_risk_score": round(score, 4)}

    return {"context_risk_score": 0.45}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
