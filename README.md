# 🛡️ SafHer: Multi-Modal Context-Aware Safety Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-2024-61DAFB?style=for-the-badge\&logo=react\&logoColor=black)](https://react.dev/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Microservices](https://img.shields.io/badge/Architecture-FastAPI-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)

---

## 📌 Overview

**SafHer** is a multi-modal, context-aware AI framework architected to structurally enhance passenger safety, with a deliberate focus on safeguarding vulnerable demographics—particularly women—within ride-hailing ecosystems.

Unlike conventional systems that rely on **post-event intervention (SOS triggers, static heuristics)**, SafHer introduces a **predictive safety paradigm** by integrating:

* Spatial graph intelligence
* Temporal behavioral modeling
* Environmental context fusion
* Real-time kinematic anomaly detection

The system is designed to **preemptively identify and mitigate risk before trip initiation**, rather than reacting after unsafe conditions emerge.

---

## 🚀 Key Innovation: Proactive vs Reactive Safety

### ❌ Reactive Paradigm (Industry Standard)

* **Static Rating Systems:** Fail to capture temporal behavioral drift (e.g., late-night volatility)
* **Shortest Path Routing (Dijkstra/A*):** Optimizes for distance/ETA, not safety topology
* **Geofence/SOS Mechanisms:** Trigger only after significant deviation or halt

### ✅ Proactive Intelligence (SafHer)

* **Temporal Behavioral Analysis:** Detection of driver degradation patterns across time windows
* **Topological Risk Routing:** Graph Neural Network-based routing over safety-weighted edges
* **Kinematic Anomaly Detection:** Continuous monitoring of velocity/momentum microstates

---

## 🏗️ System Architecture

SafHer implements a **fully decoupled, asynchronous FastAPI microservice architecture**, orchestrating multiple inference pipelines in parallel.

---

### 1. 🧠 Driver Risk Module (FT-Transformer)

* **Model:** Feature Tokenizer Transformer
* **Input:** Sequential driver trip embeddings over last $N$ trips
* **Function:** Captures temporal inconsistency and latent behavioral drift
* **Outcome:** Flags drivers with time-dependent risk escalation (e.g., nocturnal instability)

---

### 2. 🗺️ Route Safety Module (Graph Attention Networks)

* **Model:** GATConv (PyTorch Geometric)
* **Graph Source:** OpenStreetMap via OSMnx
* **Edge Features:** Crime density, illumination proxies, connectivity centrality
* **Function:** Learns attention-weighted safety scores over graph edges
* **Outcome:** Generates safety-optimized routes prioritizing high-integrity roads

---

### 3. ☁️ Context Risk Module (Temporal Fusion Transformer)

* **Model:** Lightweight TFT surrogate
* **Inputs:** Time-of-day, weather, traffic density signals
* **Function:** Contextual modulation of baseline risk scores
* **Outcome:** Dynamic reclassification of environmental safety (e.g., temporal hostility shift)

---

### 4. 🛰️ Anomaly Detection Module (Unsupervised LSTM Autoencoder)

* **Model:** Deep LSTM Autoencoder
* **Input:** GPS trajectory micro-sequences (velocity, acceleration, heading)
* **Function:** Reconstruction-based anomaly scoring
* **Outcome:** Instant detection of forced deviations / evasive kinematics

---

### 5. ⚙️ Fusion Engine (Combinatorial Optimization Layer)

* **Function:** Resolves global matching across driver-risk, route-risk, and contextual modulation
* **Objective Function:**

[
\text{Optimal Match} = \arg\min (\text{Driver_Risk} + \text{Route_Risk} + \text{Contextual_Modulator})
]

* **Outcome:** Globally optimal assignment ensuring minimum composite risk

---

## 🛠️ Tech Stack

### Backend & AI

* Python 3.10+
* FastAPI, Uvicorn
* PyTorch, PyTorch Geometric, Scikit-Learn
* OSMnx, Geopandas, NetworkX

### Frontend

* React (Vite architecture)
* Tailwind CSS
* Leaflet / React-Leaflet

---

## 🚦 Execution Guide (STRICT ORDER – DO NOT MODIFY)

### ⚠️ System Requirements

* Latest Python installed
* Node.js installed
* Stable high-speed internet (required for OSM graph + routing)

---

### ▶️ Step 1: Project Initialization

* Pull the repository into a single folder
* Open the folder in a code editor

---

### ▶️ Step 2: Backend + AI Module Activation

Open terminal (`Ctrl + ~`) and run:

```bash
cd safety_intelligence_system
python run_servers.py
```

✔ This initializes all AI pipelines and FastAPI microservices

---

### ▶️ Step 3: Frontend Server Execution

Open a **new terminal (do NOT close the first one)**

```bash
cd map
cd client
npm start
```

---

### ▶️ Step 4: Application Access

* Terminal will return a **localhost URL**
* Open it in your browser

---

## 🌍 Operational Notes

* High network bandwidth is required for:

  * OpenStreetMap graph loading
  * Route rendering
  * AI inference synchronization

* ⚠️ Current optimization scope:

  * **Delhi NCR region (best performance)**

---

## 📊 Dataset Attribution

* Uber Ride Analytics (KaggleHub)
* Microsoft Geolife Trajectories
* OpenStreetMap (OSMNx)
* Open-Meteo API

---

## 📜 License

Strictly private project.
No redistribution, reuse, or modification permitted without explicit authorization.

---
