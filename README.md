# 🛰️ SatVision AI — Autonomous Satellite Anomaly & Threat Attribution Detector
### by Calypso
> **Track:** SpaceTech & Satellite Intelligence | Orcathon 2026
> An AI-powered geospatial intelligence system designed to detect rapid terrestrial anomalies, delineate affected boundaries, quantify environmental disruption, and distinguish natural ecological shifts from anthropogenic damage.

---

## 📌 Problem Overview
Manual inspection of multi-temporal satellite imagery is slow and resource-intensive during environmental crises and industrial disasters. **SatVision AI** addresses this challenge by providing an end-to-end autonomous analysis pipeline:
1. Ingests temporal satellite image pairs (Baseline $T_0$ vs. Target $T_1$).
2. Automatically identifies surface variances, contour disruptions, and geometric signatures.
3. Classifies events into four distinct environmental and security categories.
4. Distinguishes natural environmental shifts from human-induced degradation with automated report generation.

---

## 🎯 Target Categories & Detection Benchmarks

| Scenario | Primary Attribution | Key Visual & Spectral Indicators |
| :--- | :--- | :--- |
| **Forest Destruction** | Anthropogenic / Human Activity | Linear road cuttings and geometric fishbone logging clearings across dense canopy. |
| **Coastal Changes** | Natural Phenomenon / Ecological | Inland sea/lake desiccation, shoreline retreat, and exposed saltbed contours. |
| **Infrastructure Damage** | Anthropogenic / Catastrophic Disaster | High-energy blast craters, structural fragmentation, and industrial warehouse collapse. |
| **Unusual Land Excavation**| Anthropogenic / Earthworks | Concentric terraced open pits, haulage routes, and extensive strip mining. |

---

## ⚙️ Architecture & Tech Stack

* **Multimodal Vision AI:** Google Gemini 1.5 Flash via `google-genai` SDK for high-level semantic context, root-cause attribution, and automated technical summary synthesis.
* **Computer Vision Pipeline:** OpenCV (`cv2`) & NumPy for image alignment, illumination normalization (LAB histogram equalization), morphological difference mapping, and contour extraction.
* **Backend Engine:** FastAPI & Uvicorn for asynchronous REST API handling and image stream base64 encoding.
* **Command Console UI:** Vanilla HTML5, CSS3 (responsive dark-mode glassmorphism interface), and JavaScript (Fetch API).

---

## 🚀 Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TanusnSN/satellite_anomaly_detector.git
   cd satellite_anomaly_detector

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set your Gemini API key (Optional):**
   ```bash
   Windows (PowerShell):
   PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   Linux / macOS:
   Bash
   export GEMINI_API_KEY="your_api_key_here"

4. **Run the application:**
  ```bash
  python main.py

5. **Access the Dashboard:**
   ```bash
   Open your browser and navigate to:
   http://127.0.0.1:8000/