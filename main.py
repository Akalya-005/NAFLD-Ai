from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import xgboost as xgb
from keras.src.models import load_model
import joblib
from pydantic import BaseModel
import os

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# --- 2. CORS Middleware ---
# This allows your frontend (running on a different address) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Load All Models, Scalers, and Databases ---
try:
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the model files
    toxicity_model_path = os.path.join(base_dir, "toxicity_predictor.json")
    lstm_model_path = os.path.join(base_dir, "lstm_progression_model.keras")
    scaler_path = os.path.join(base_dir, "alt_scaler.pkl")
    
    # Load the XGBoost toxicity model
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(toxicity_model_path)

    # Load the LSTM progression model
    model_lstm = load_model(lstm_model_path, compile=False)

    # Load the ALT scaler
    scaler = joblib.load(scaler_path)

    models_loaded = True
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

# --- 4. Define Data Models for API Requests ---
# These models ensure the data sent from the frontend is in the correct format.
class DrugDesignParams(BaseModel):
    anti_fat_effect: int
    anti_inflammatory_effect: int
    anti_fibrotic_effect: int
    mol_wt: float
    log_p: float
    num_h_donors: int
    num_h_acceptors: int

class SimulationDrug(BaseModel):
    alt_effect: float
    designParams: dict # We'll pass the original design params too

class SimulationPatient(BaseModel):
    age: int
    bmi: float
    ast: int
    alt: int

class SimulationRequest(BaseModel):
    patientData: SimulationPatient
    drug: SimulationDrug

# --- 5. Create API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Nightingale AI Backend is running."}

@app.post("/predict")
def predict_drug_profile(params: DrugDesignParams):
    if not models_loaded:
        return {"error": "Models are not loaded."}

    # Prepare data for XGBoost model
    designed_params = np.array([params.mol_wt, params.log_p, params.num_h_donors, params.num_h_acceptors])
    
    # Predict toxicity probability
    toxicity_proba = model_xgb.predict_proba(designed_params.reshape(1, -1))[0][1]
    
    efficacy_score = params.anti_fat_effect + params.anti_inflammatory_effect + params.anti_fibrotic_effect
    
    final_effect = 0
    if toxicity_proba > 0.7:
        toxicity_risk = "High Risk"
        final_effect = 20.0
    elif toxicity_proba > 0.3:
        toxicity_risk = "Moderate Risk"
        final_effect = -1 * (efficacy_score * 0.5)
    else:
        toxicity_risk = "Low Risk (Safe)"
        final_effect = -1 * (efficacy_score * 1.5)
        
    return {
        "safety_risk": toxicity_risk,
        "efficacy_score": f"{efficacy_score}/30",
        "alt_effect": final_effect,
        "formula": "C₁₄H₁₉NO₅" # Placeholder as before
    }

@app.post("/simulate")
def simulate_trial(request: SimulationRequest):
    if not models_loaded:
        return {"error": "Models are not loaded."}

    patient_data = request.patientData
    drug = request.drug

    # This part uses a simplified progression for demonstration.
    # For a full LSTM implementation, you'd need to format the input sequence correctly.
    # This example demonstrates the API structure.
    initial_alt = patient_data.alt
    input_seq = np.array([initial_alt * 0.95, initial_alt * 0.98, initial_alt])
    input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
    
    predictions_scaled = []
    current_batch = input_seq_scaled.reshape((1, 3, 1))
    
    for _ in range(12):
        pred = model_lstm.predict(current_batch, verbose=0)[0]
        predictions_scaled.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
    predictions_unscaled = scaler.inverse_transform(predictions_scaled)
    
    patient_adjustment = ((patient_data.bmi - 25) * 0.5 if patient_data.bmi > 25 else 0) + ((patient_data.age - 40) * 0.1 if patient_data.age > 40 else 0)
    patient_trend = np.linspace(0, patient_adjustment, 12)
    
    prediction_without_drug = predictions_unscaled.flatten() + patient_trend
    
    drug_effect_value = drug.alt_effect
    drug_trend = np.linspace(0, drug_effect_value, 12)
    
    prediction_with_drug = prediction_without_drug + drug_trend
    prediction_with_drug = np.maximum(prediction_with_drug, 25) # Clamp ALT to a minimum of 25

    return {
        "placebo": prediction_without_drug.tolist(),
        "treatment": prediction_with_drug.tolist()
    }
