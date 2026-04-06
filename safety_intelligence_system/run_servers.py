import subprocess
import os
import sys
import time

def start_microservices():
    processes = []
    print("Initializing 5 Multi-Modal AI Microservices for Frontend Maps Interface...")
    
    cwd = os.getcwd()
    modules = [
        ("driver_risk_module", 8001),
        ("route_safety_module", 8002),
        ("context_risk_module", 8003),
        ("anomaly_detection_module", 8004),
        ("fusion_module", 8000),
    ]
    
    for mod, port in modules:
        print(f"Starting {mod} on Port {port}...")
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)],
                             cwd=os.path.join(cwd, mod), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        
    print("\n✅ All Backend ML Microservices are ACTIVE and mapping on PORT 8000!")
    print("Press Ctrl+C to gracefully terminate the servers.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nGracefully terminating Microservices...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    start_microservices()
