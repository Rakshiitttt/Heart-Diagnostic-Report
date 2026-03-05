# Clinical Heart AI: XAI-Powered Prediction Dashboard

A premium, clinical-grade medical application that utilizes **Explainable AI (XAI)** and **LightGBM** to predict the likelihood of heart disease in patients based on bio-markers.

![[Demo](https://heart-diagnostic-report.streamlit.app/)]

## 🌟 Overview
This project transforms a data-science research notebook into a production-ready medical dashboard. It features a high-performance **FastAPI** backend that handles clinical feature engineering and a modern **React** frontend designed with a dark glassmorphism aesthetic for professional environments.

### 📊 Model Performance
- **Algorithm**: LightGBM (tuned via Optuna)
- **AUC Score**: 0.9559 (95.6% confidence)
- **Interpretability**: SHAP (Shapley Additive Explanations) integrated for feature transparency.

## 🚀 Features
- **Real-time Diagnostic Analysis**: Instant risk evaluation as clinical data is inputted.
- **Automated Bio-marker Engineering**: Automatically calculates:
  - `HR_Reserve` (Heart Rate Reserve)
  - `Expected_Max_HR`
  - `BP_Category` (Based on AHA Hypertension Guidelines)
- **Premium UI/UX**: 
  - Glassmorphic design with subtle animations using `framer-motion`.
  - Dark mode optimized for clinical tablets and low-light environments.
  - Interactive risk probability bars and diagnostic badges.

## 🛠 Tech Stack
- **Backend**: Python, FastAPI, LightGBM, Pandas, Scikit-learn
- **Frontend**: React (Vite), Framer Motion, Axios, Lucide React
- **ML Optimization**: Optuna (Hyperparameter Tuning), SHAP

## 🏃 Getting Started

### Prerequisites
- Python 3.9+
- Node.js & npm

### Backend Setup
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn lightgbm joblib pandas scikit-learn
   ```
2. Train the model (creates `.pkl` artifacts):
   ```bash
   python train_model.py
   ```
3. Start the API server:
   ```bash
   python main.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the dev server:
   ```bash
   npm run dev
   ```

## 📂 Project Structure
```text
├── main.py                # FastAPI Prediction API
├── train_model.py         # ML Training & Serialization Script
├── .gitignore             # Git exclusion rules
├── frontend/              # Vite + React Application
│   ├── src/App.jsx        # Prediction Dashboard Logic
│   └── src/index.css      # Premium Styling System
└── heart_disease_model.pkl # Trained Model Artifact
```

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
