# 📍 CASEFILE: Pan-India Missing Person Investigation System

> **AI-Powered Geographical Location & Route Prediction System** covering all **28 States** of India

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)]()

---

## 🔍 Overview

CASEFILE is a comprehensive AI-powered investigation dashboard designed for **missing person case analysis** across India. It combines **machine learning models**, **Markov chain route prediction**, **anomaly detection**, and **interactive geospatial mapping** to assist in search prioritization.

> ⚠️ **Disclaimer**: This is an **academic simulation project** using synthetic data. It is NOT intended for real-world police operations.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📋 **FIR & Case Profile** | Complete case dossier with demographic details, FIR reference, and last known location |
| 🎯 **Probable Locations (ML)** | 4-model ensemble (RandomForest, XGBoost, GradientBoosting, KNN) for location prediction |
| 🛤️ **Route Prediction** | Markov Chain with beam search decoding for probable movement routes |
| ⚠️ **Anomaly Detection** | Ensemble of Isolation Forest + LOF + One-Class SVM for unusual movement patterns |
| 🔍 **Search Priority Matrix** | 6-factor composite scoring system for search area prioritization |
| 🗺️ **Interactive Tactical Maps** | Multi-layer Folium maps with evidence overlays, heatmaps, and cluster visualization |
| 💡 **Explainability (XAI)** | Feature importance analysis and natural language prediction rationale |
| 📈 **Model Evaluation** | Complete pipeline summary with Top-K accuracy curves and model benchmarks |

---

## 🏗️ Architecture

```
CASEFILE_MISSING_PERSON/
├── app/
│   ├── app.py              # Main Streamlit dashboard (8 tabs)
│   └── components.py       # Reusable UI components
├── data/
│   ├── raw/                # Original GPS trajectory data
│   ├── processed/          # Cleaned & feature-engineered data
│   └── synthetic/          # 112 simulated cases across 28 states
├── models/                 # Trained ML models (.pkl)
├── notebooks/              # Jupyter notebooks (01-07 pipeline)
├── reports/                # Generated maps, charts & reports
├── src/                    # Source modules
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── prediction.py
│   ├── route_prediction.py
│   ├── search_priority.py
│   ├── explainability.py
│   ├── mapping.py
│   ├── case_generator.py
│   └── convert_to_india.py
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit, Plotly, Folium |
| **ML/AI** | scikit-learn, XGBoost |
| **Data** | Pandas, NumPy |
| **Mapping** | Folium, ESRI Tile Server |
| **Serialization** | Joblib |

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/patapanchalaganesh/CASEFILE_MISSING_PERSON.git
cd CASEFILE_MISSING_PERSON

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo → Branch: `main` → Main file: `app/app.py`
4. Click **Deploy!**

---

## 📊 Data Pipeline

| Stage | Technique | Output |
|-------|-----------|--------|
| Data Collection | Synthetic Generation | 902,051 GPS points |
| Preprocessing | Speed Filter | 896,818 cleaned points |
| Feature Engineering | Haversine + Stay Points | 2,101 stay points |
| Clustering | DBSCAN + K-Means | 70 → 5 areas (Silhouette: 0.446) |
| Anomaly Detection | IF + LOF + SVM | 44,840 anomalies (5%) |
| Case Generation | Profile Sampling | 112 cases (28 states) |
| Location Prediction | RF / XGB / GBM / KNN | GradientBoosting (best) |
| Route Prediction | Markov + Beam Search | 3 predicted routes |
| Search Priority | 6-Factor Composite | 3 regions ranked |
| Explainability | SHAP / Tree Importance | Feature importance |
| Mapping | Folium Layers | 3 interactive maps |

---

## 👥 Team

Academic project built for educational purposes.

---

## 📄 License

This project is for **academic and educational purposes only**. All data is synthetic and simulated.
