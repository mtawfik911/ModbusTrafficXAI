# ModbusTrafficXAI

**Explainable Deep Learning for Modbus Packet Traffic Classification** using CNN and Attention-based models, enhanced with **LIME** and **SHAP** for explainability, and deployed via **FastAPI** and **Streamlit GUI**.

This project demonstrates an end-to-end, production-ready AI system for industrial network traffic analysis with a strong focus on **model transparency and interpretability**.

---

##  Project Overview

Industrial Control Systems (ICS) such as **Modbus** networks are critical infrastructure and require reliable intrusion detection and traffic analysis.

This project provides:
- Deep Learning–based classification of Modbus packet traffic
- Attention mechanisms to improve feature learning
- Explainable AI (XAI) to understand model decisions
- REST API for integration
- Interactive GUI for demonstration

---

##  Key Features

###  Machine Learning
- **CNN model** for baseline packet classification
- **Attention-CNN model** (self-attention + CNN)
- Binary classification: **Normal vs Abnormal traffic**
- Input length: **17 Modbus packet features (bytes)**

###  Explainable AI (XAI)
- **LIME**
  - Local explanations per sample
  - Interactive HTML output
- **SHAP**
  - Model-agnostic explanations
  - Feature attribution plots
  - Saved as PNG for reproducibility

###  Deployment
- **FastAPI backend**
  - `/predict`
  - `/explain`
  - `/health`
- **Streamlit GUI**
  - Model selection (CNN / Attention)
  - Prediction + probability
  - LIME & SHAP top features
- Designed to run **without Docker** (works on Termux / proot)

---

## ⚙️ Requirements & Installation

### Install dependencies
From the project root directory:

```bash
pip install -r requirements.txt
```


## ▶️ How to Run the Project

You need to run two components in **two separate terminals**:
1. FastAPI backend (model + XAI)
2. Streamlit GUI (user interface)

---

### 1️- Start the FastAPI backend

From the project root, open Terminal 1 and run:

```bash
./scripts/run_api.sh
```

After the backend starts, open the API documentation in your browser:
http://127.0.0.1:8000/docs

This page allows you to test:
- `/predict` – model prediction
- `/explain` – LIME & SHAP explanations
- `/health` – service status

---

### 2️- Start the Streamlit GUI

Open **another terminal** (Terminal 2) and run:

```bash
./scripts/run_gui.sh
```
Then open the GUI in your browser:
http://127.0.0.1:8501

##  Repository Structure

```text
.
├── api/                      # FastAPI backend
│   └── app.py
├── gui/                      # Streamlit GUI
│   └── app.py
├── src/modbus_traffic_xai/   # Core Python package
│   ├── models.py             # CNN & Attention models
│   ├── preprocessing.py     # Data loading & preprocessing
│   ├── visualization.py     # Training plots
│   ├── xai/
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   └── config.py
├── scripts/
│   ├── run_api.sh
│   ├── run_gui.sh
│   ├── test_lime.py
│   └── test_shap.py
├── dataset/                  # Dataset files
├── models/                   # Trained models (.keras)
│   ├── cnn.keras
│   └── attention.keras
├── results/
│   ├── shap/
│   │   └── shap_summary_sample0.png
│   └── lime/
│       └── lime_sample0.html
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE

```

