import subprocess
import time
import httpx
import asyncio
import json
import os
import sys

async def run_test():
    processes = []
    print("Initializing 5 Multi-Modal AI Microservices...")
    
    cwd = os.getcwd()
    modules = [
        ("driver_risk_module", 8001),
        ("route_safety_module", 8002),
        ("context_risk_module", 8003),
        ("anomaly_detection_module", 8004),
        ("fusion_module", 8000),
    ]
    
    for mod, port in modules:
        # Spin up each FastAPI silently
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)],
                             cwd=os.path.join(cwd, mod), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        
    print("Waiting 10 seconds for standard model initializations...")
    time.sleep(10)
    
    url = "http://localhost:8000/predict/safety"
    
    # MOCK DATA payload exactly matching fusion gateway expectations
    payload = {
        "available_drivers": [
            {
                "driver_id": "D991",
                "eta_minutes": 12.0,
                "distance_km": 4.5,
                "driver_rating": 3.9,
                "cancellations": 3
            },
            {
                "driver_id": "D104",
                "eta_minutes": 5.0,
                "distance_km": 1.2,
                "driver_rating": 4.8,
                "cancellations": 0
            }
        ],
        "source_lat": 28.6562, 
        "source_lon": 77.2410,
        "dest_lat": 28.5450,   
        "dest_lon": 77.1926,
        "timestamp": "2026-03-28T21:30:00Z", 
        "congestion_level": 0.85, 
        "weather_condition": "heavy_rain"
    }

    try:
        print("Sending Target Payload (Red Fort -> IIT Delhi) to Global Gateway...")
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            print("\n================== FUSION ORCHESTRATOR RESULT ==================")
            print(f"Status: {response.status_code}")
            print(json.dumps(response.json(), indent=4))
            print("=================================================================\n")
    except Exception as e:
        print(f"API Sub-Error: Did not receive payload properly. Logs: {e}")
        
    finally:
        print("Gracefully terminating Microservices...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    asyncio.run(run_test())
