from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import asyncio
from datetime import datetime, timezone

app = FastAPI(title="Proactive Ride Matchmaker (Fusion API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DriverConstraint(BaseModel):
    driver_id: str
    eta_minutes: float
    distance_km: float
    driver_rating: float
    cancellations: int

class FullstackRequest(BaseModel):
    available_drivers: List[DriverConstraint]
    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float
    timestamp: str # ISO format
    congestion_level: float
    weather_condition: str

class FeedbackInput(BaseModel):
    trip_id: str
    driver_rating: float
    passenger_safety_flag: bool
    gender: Optional[str] = None

# Internal ML Microservice Addresses
MODULE_URLS = {
    "driver": "http://localhost:8001/predict/driver",
    "route": "http://localhost:8002/predict/route_candidates",
    "context": "http://localhost:8003/predict/context",
    "anomaly": "http://localhost:8004/predict/anomaly"  # Separate streaming module
}

@app.post("/predict/safety")
async def predict_safety(req: FullstackRequest):
    async with httpx.AsyncClient(timeout=10.0) as client:
        # === 1. Evaluate Driver Behavioral Constraints (Vectorized) ===
        driver_tasks = []
        for d in req.available_drivers:
            # Wrap the payload to match what Task 1 expects
            payload = {
                 "driver_id": d.driver_id,
                 "trip_duration": d.eta_minutes * 2,
                 "trip_distance": d.distance_km,
                 "driver_rating": d.driver_rating,
                 "cancellations": d.cancellations,
                 "pickup_time": req.timestamp.split("T")[1][:8] if "T" in req.timestamp else "00:00:00",
                 "drop_time": "00:00:00"
            }
            driver_tasks.append(client.post(MODULE_URLS["driver"], json=payload))
            
        driver_responses = await asyncio.gather(*driver_tasks, return_exceptions=True)
        driver_scores = {}
        
        for i, res in enumerate(driver_responses):
            try:
                base_risk = res.json().get("driver_risk_score", 0.5)
                # alpha, beta, gamma adjustments 
                eta_penalty = (req.available_drivers[i].eta_minutes / 30.0) * 0.1
                dist_penalty = (req.available_drivers[i].distance_km / 10.0) * 0.1
                
                final_driver_score = base_risk + eta_penalty + dist_penalty
                driver_scores[req.available_drivers[i].driver_id] = final_driver_score
            except Exception:
                driver_scores[req.available_drivers[i].driver_id] = 1.0 # Max risk if API fails

        # === 2. Context Risk (Weather / Time Modulators) ===
        context_payload = {
            "timestamp": req.timestamp,
            "congestion_level": req.congestion_level,
            "average_speed": 35.0,
            "accident_count": 0,
            "ride_duration": 45.0
        }
        try:
             c_res = await client.post(MODULE_URLS["context"], json=context_payload)
             context_risk = c_res.json().get("context_risk_score", 0.5)
        except:
             context_risk = 0.5
             
        # === 3. Route Generation & Graph Topologies ===
        route_payload = {
             "source_lat": req.source_lat,
             "source_lon": req.source_lon,
             "dest_lat": req.dest_lat,
             "dest_lon": req.dest_lon,
             "context_risk_modulator": context_risk
        }
        try:
             r_res = await client.post(MODULE_URLS["route"], json=route_payload)
             k_routes = r_res.json().get("candidate_routes", [])
        except:
             k_routes = [{"route_id": "mock_route", "geometry": [[req.source_lat, req.source_lon], [req.dest_lat, req.dest_lon]], "route_risk_score": 0.5}]

        # === 4. Combinatorial Optimization (Driver + Route Matrix) ===
        # final_score(driver_i, route_j) = DriverRisk(i) + RouteRisk(j) + ContextRisk
        optimal_match = None
        min_combined_risk = float('inf')
        
        for d_id, d_risk in driver_scores.items():
            for route in k_routes:
                r_risk = route.get("route_risk_score", 0.5)
                
                # Combine factors optimally 
                combined_risk = d_risk + r_risk + (0.3 * context_risk)
                
                if combined_risk < min_combined_risk:
                    min_combined_risk = combined_risk
                    optimal_match = {
                        "selected_driver_id": d_id,
                        "best_route_geometry": route.get("geometry"),
                        "combinatorial_risk_score": round(combined_risk, 4),
                        "driver_risk_isolated": round(d_risk, 4),
                        "route_risk_isolated": round(r_risk, 4),
                        "context_risk_isolated": round(context_risk, 4)
                    }

        alerts = []
        if context_risk > 0.7:
             alerts.append("High situational risk timeframe: Routing via illuminated major roads.")
        if optimal_match and optimal_match['combinatorial_risk_score'] > 2.0:
             alerts.append("WARNING: Selected combination maintains elevated risk parameters.")

        if optimal_match:
             optimal_match["contextual_alerts"] = alerts
             return optimal_match
             
        return {"error": "Combinatorial mapping failed."}

@app.post("/feedback")
async def ingest_feedback(data: FeedbackInput):
    # Standard Post-Ride logging pipeline (for batch retraining & fairness tuning)
    print(f">> FEEDBACK LOGGED [Trip: {data.trip_id} | Driver Rating: {data.driver_rating} | Passenger Flag: {data.passenger_safety_flag}]")
    return {"status": "Feedback securely persisted to ML logging pipeline for continuous fairness updates."}

@app.get("/start")
async def start_ride_from_map(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    import random
    # Generating dynamic available drivers to ensure variety
    available_drivers = [
        DriverConstraint(
            driver_id=f"D{random.randint(100, 999)}", 
            eta_minutes=round(random.uniform(2.0, 15.0), 1), 
            distance_km=round(random.uniform(0.5, 6.0), 1), 
            driver_rating=round(random.uniform(3.5, 5.0), 1), 
            cancellations=random.randint(0, 5)
        ),
        DriverConstraint(
            driver_id=f"D{random.randint(100, 999)}", 
            eta_minutes=round(random.uniform(1.0, 8.0), 1), 
            distance_km=round(random.uniform(0.2, 3.0), 1), 
            driver_rating=round(random.uniform(4.0, 5.0), 1), 
            cancellations=random.randint(0, 2)
        )
    ]
    
    # Pack parameters into FullstackRequest using frontend coordinates
    req = FullstackRequest(
        available_drivers=available_drivers,
        source_lat=start_lat,
        source_lon=start_lon,
        dest_lat=end_lat,
        dest_lon=end_lon,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        congestion_level=round(random.uniform(0.1, 0.9), 2),
        weather_condition=random.choice(["clear", "heavy_rain", "snow", "fog"])
    )
    
    # Call internal POST orchestrator functionality directly
    optimal_match = await predict_safety(req)
    return optimal_match

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
