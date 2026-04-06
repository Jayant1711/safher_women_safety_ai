# Project Report: Multi-Modal Context-Aware Safety Intelligence System (SafHer)
**Course Code:** CSE274
**Domain:** Deep Learning, Graph Neural Networks, and Microservices

---

## 1. Executive Summary
The **Safety Intelligence System (SafHer)** is a sophisticated, multi-modal AI framework engineered to structurally enhance passenger safety, with a focused architectural bias toward protecting vulnerable demographic groups, particularly women, in ride-hailing ecosystems. Instead of relying solely on proximity matching, the system integrates spatial graph models, situational attention networks, and time-series driver analysis to score risk. The project utilizes a fully decoupled asynchronous FastAPI microservice architecture orchestrating five parallel inference pipelines.

---

## 2. System Architecture
The framework relies on five specialized AI services communicating via parallel inference via a central API Gateway (Fusion Engine).

1.  **Driver Risk Module:** Temporal assessment of driver behavioral consistency.
2.  **Route Safety Module:** Spatial and topological risk inference using Graph Neural Networks (GNNs).
3.  **Context Risk Module:** Environmental variable modulation.
4.  **Anomaly Detection Module:** Unsupervised live tracking for route deviations or unexpected behaviors.
5.  **Fusion Module:** Central combinatorial optimization matrix.

---

## 3. Deep Learning Modules: Technical Specifications

### A. Driver Risk Module 

*   **AI Model & Architecture:**
    Uses a time-series adaptation of a **Feature Tokenizer Transformer (FT-Transformer)**. It sequences temporal data and maps 7 continuous features to a dense latent space ($d_{model} = 64$). The transformer uses 4 encoder layers, 4 multi-head attention blocks, and a dense readout MLP mapped to a Sigmoid layer to output risk probabilites.
*   **Data Preprocessing:**
    *   **Dataset:** `yashdevladdha/uber-ride-analytics-dashboard` (KaggleHub).
    *   **Logic:** Aggregates rolling windows ($N=10$ sequential trips per driver).
    *   **Features:** Scaled via `StandardScaler`. Features include `trip_duration`, `trip_distance`, `driver_rating`, `rating_variance`, `night_trip_ratio`, `cancellation_rate`, and `avg_trip_duration`.
*   **Mathematical Formulations:**
    *   *Positional Encoding (Analytical):*
        $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
        $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$
    *   *Attention Vector Mapping:*
        $$Attention(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$
*   **Fine-Tuning & Hyperparameters:**
    Optimized using AdamW optimizer (`lr=1e-3`) using Binary Cross Entropy Loss (BCELoss). Regularization enforced via dropout ($0.1$) on fully connected layers. Model performance is evaluated via ROC-AUC and F1-score tracking.
*   **Novelty vs. Uber/Rapido (Industry Gap Solved):**
    Current ride-hailing apps (Uber, Rapido) evaluate drivers using a static lifetime mean rating (e.g., 4.7 stars) and raw cancellation rates. They act reactively. The SafHer module evaluates **temporal behavioral degradation**. It mathematically deduces if a driver is becoming volatile (e.g., their rating drops strictly at night, or they cancel frequently after long shifts), allowing the platform to preemptively block them from matching with vulnerable passengers at night.

---

### B. Route Safety Module 

*   **AI Model & Architecture:**
    Based on **Graph Attention Networks (GATConv)** using PyTorch Geometric. Uses a 2-layer GAT structure. The first layer attends over neighbors to build local embeddings (concat=True). The second layer consolidates them (concat=False). Finally, an internal Edge-MLP evaluates combinations of nodes to predict edge-wise topological risk.
*   **Data Preprocessing:**
    *   **Dataset:** OpenStreetMap (OSMNx) dynamic drive-graphs configured locally (e.g., Central Delhi bounding).
    *   **Logic:** Graph nodes denote intersections; Graph edges denote roads. Simulated multivariate layers cast spatial gaussian distributions over roads representing crime density. Real-time characteristics (`road_weight`, `length`, `congestion`) are encoded into a 4-dimensional edge vector.
*   **Mathematical Formulations:**
    *   *Label Definition:*
        $$\text{Target Safety (Y)} = 0.5 \times \text{crime} + 0.3 \times \text{congestion} + 0.2 \times (1.0 - \text{road\_weight})$$
    *   *Graph Attention Coefficient:*
        $$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_k]))}$$
    *   *Edge Scoring MLP:*
        $$\text{Risk}_{uv} = \text{MLP}([\mathbf{h}_{src} || \mathbf{h}_{dst} || \mathbf{edge\_features}_{uv}])$$ 
*   **Fine-Tuning & Hyperparameters:**
    Trained via MSELoss comparing predicted edge scores vs algorithmic target risk. Tuned with Adam (`lr=1e-3`), integrating steep L2 kernel decay (`weight_decay=5e-4`) to prevent spatial over-smoothing over the OSMnx network. Masked edge validation (80/20 train/val split).
*   **Novelty vs. Uber/Rapido (Industry Gap Solved):**
    Uber Maps and Google Maps route exclusively using A* or Dijkstra constraints designed to minimize Expected Time of Arrival (ETA) or distance. They do not factor in topological safety. SafHer routes passengers over roads with a higher structural integrity—bypassing unlit, narrow, or historically isolated alleys, prioritizing well-lit, populated arterial roads even if it adds minutes to the journey.

---

### C. Context Risk Module

*   **AI Model & Architecture:**
    A lightweight time-series surrogate of a **Temporal Fusion Transformer**. Input features are projected via linear scaling, merged with a parameter-based positional matrix, and squeezed through two `batch_first` transformer layers yielding context representations from final hidden states.
*   **Data Preprocessing:**
    *   **Dataset:** Open-Meteo context emulation filtered strictly for "Delhi" limits.
    *   **Logic:** Arrays scaled uniformly, building chronological windows parsing: `hour_of_day`, `is_night`, `is_peak_hour`, `congestion_level`, `night_congestion`, `average_speed`. Sequence label defined as High-Risk if an accident is active, or extreme congestion arises deep into the night.
*   **Mathematical Formulations:**
    *   *Trainable Positional Embeddings:* 
        $$E_{pos} = \mathbf{W}_{param} \in \mathbb{R}^{100 \times d_{model}}$$
    *   *Situation Modulator Readout:*
        $$\gamma \in [0, 1] = \text{Sigmoid}(\text{GELU}(\text{Linear}(\mathbf{h}_{T})))$$
*   **Fine-Tuning & Hyperparameters:**
    $d_{model}=32$, 4 attention heads. Optimized aggressively using MSELoss acting as a regression anchor for real-time temporal safety modulation.
*   **Novelty vs. Uber/Rapido (Industry Gap Solved):**
    Uber predominantly uses time and weather constraints exclusively to enact dynamic Surge Pricing logic. The Context Risk Module utilizes situational data dynamically as a Safety Modulator. A road that is perfectly safe at 3 PM can become hostile at 3 AM. This AI strictly limits permissible routes to highly policed avenues when contextual environmental risks amplify.

---

### D. Anomaly Detection Module 

*   **AI Model & Architecture:**
    Utilizes an **Unsupervised Deep LSTM Autoencoder**. It employs a stacked symmetric RNN format. The Encoder compresses a temporal coordinate sequence down to a 32-dim bottleneck vector. The Decoder iteratively unrolls this embedding and projects back to reconstruct the sequence.
*   **Data Preprocessing:**
    *   **Dataset:** Microsoft Geolife PLT Trajectories.
    *   **Logic:** Cleans standard GPS sequences translating primitive Lat/Lon shifts locally to meters using `Haversine` transforms. Vectors map speed in $m/s$. Walking trajectories ($<5.0$ m/s) are scrubbed. Sequences mapped as rigorous blocks of $T=20$ timesteps.
*   **Mathematical Formulations:**
    *   *Haversine Spatial Differential:*
        $$Dist = 2 r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
    *   *LSTM Cell Bottleneck:* $h_t = \text{LSTM}(x_t, h_{t-1})$; Seq_Embedding $Z = h_T$
    *   *Reconstruction Loss Metric:*
        $$MSE = \frac{1}{T \times F} \sum_{t=1}^{T} || x_{t} - \hat{x}_{t} ||^2_2$$
*   **Fine-Tuning & Hyperparameters:**
    Learning driven by identifying normative representations of GPS momentum. Two-layer LSTMs. Validated not on typical accuracy, but by thresholding Reconstruction Error. When $MSE$ exceeds the 95th-percentile benchmark established on normalized safety tracks, alarms trip natively.
*   **Novelty vs. Uber/Rapido (Industry Gap Solved):**
    Uber’s "RideCheck" acts reactively. It relies almost exclusively on static geometric geofences, triggering only if a vehicle halts for extensive durations (15+ min) or experiences total GPS signal failure. SafHer’s Autoencoder monitors kinematic velocity micro-states iteratively via momentum. It instantaneously flags if a vehicle begins a high-speed forced evasion sequence or micro-deviates off a structured highway into an isolated region without needing user SOS intervention.

---

## 4. The Fusion Engine (Combinatorial Orchestration)
The **Fusion Module** (`fusion_module/main.py`) operates as a synchronous pipeline integrating the dimensions modeled above.

*   **Workflow Algorithm:**
    1. Evaluates parallel REST dispatches mapping driver sets and candidate routes spanning origin bounding to destination graphs.
    2. Modulates constraints heavily based upon temporal weather/traffic context thresholds ($Risk > 0.7$ prompts specific navigation restrictions).
    3. Resolves global combinatorics evaluating potential pairings across topological risk vs behavioral profiles natively using:
*   **Combinatorial Risk Formula (Ground-Truth):**
    $$\text{Final Driver Score}_i = \text{Model}_{BCE} + (0.1 \times \frac{ETA_{min}}{30}) + (0.1 \times \frac{Dist_{km}}{10})$$
    $$\text{Optimal Match} \rightarrow \arg\min (\text{Driver}_i + \text{Route}_j + (0.3 \times \text{Context}))$$

*   **Novelty vs. Uber/Rapido:** Uber operates siloed matchers (proximity pairing maps Driver and Nav strictly outputs ETA line). SafHer operates a unified optimization matrix where Driver behavior scores and Route topology scores directly restrict each other natively, guaranteeing multi-dimensional passenger parity.
