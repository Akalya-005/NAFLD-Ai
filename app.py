
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib

# --- 1. Load All Models, Scalers, and Databases ---
try:
    # Load the XGBoost toxicity model
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model("toxicity_predictor.json")

    # Load the LSTM progression model
    model_lstm = load_model("lstm_progression_model.keras", compile=False)

    # Load the ALT scaler
    scaler = joblib.load('alt_scaler.pkl')

    # Load the molecule database for structure matching
    molecules_db = pd.read_csv("molecules_db.csv")
    models_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading a required file: {e}. Please make sure all files ('toxicity_predictor.json', 'lstm_progression_model.keras', 'alt_scaler.pkl', 'molecules_db.csv') are in the same folder.")
    models_loaded = False

# --- Session State Initialization ---
if 'drug_alt_effect' not in st.session_state:
    st.session_state.drug_alt_effect = 0
if 'drug_profile_name' not in st.session_state:
    st.session_state.drug_profile_name = "No Drug Designed Yet"

# --- Helper Functions ---
def get_liver_status(alt, ast, bmi):
    """Calculates a comprehensive liver health status."""
    score = 0
    if alt > 50: score += 1
    if alt > 100: score += 1
    if ast > 40: score += 1
    if bmi > 25: score += 1
    if bmi > 30: score += 1
    if alt > 0 and (ast / alt) < 1: score += 1
    if score <= 1: return "Healthy", "#28a745"
    elif score <= 3: return "Moderate Risk", "#ffc107"
    else: return "High Risk", "#dc3545"

def find_closest_molecule(design_params, db):
    """Finds the molecule in the DB that best matches the designed parameters."""
    db_params = db[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']]
    distances = np.linalg.norm(db_params.values - design_params, axis=1)
    best_match_idx = np.argmin(distances)
    return db.iloc[best_match_idx]['smiles']

# --- 2. Build the User Interface (UI) ---
st.title("AI Drug Design & Clinical Simulation for NAFLD")

if models_loaded:
    # --- PART 1: DRUG DESIGN LABORATORY ---
    st.header("Part 1: Drug Design Laboratory")
    st.write("Design a new drug, and our system will find a real chemical structure that matches your design.")

    st.sidebar.header("Design Your Drug Here:")
    st.sidebar.subheader("Therapeutic Profile (Efficacy)")
    anti_fat_effect = st.sidebar.slider("Anti-Fat Effect", 0, 10, 8)
    anti_inflammatory_effect = st.sidebar.slider("Anti-Inflammatory Effect", 0, 10, 8)
    anti_fibrotic_effect = st.sidebar.slider("Anti-Fibrotic Effect", 0, 10, 6)

    st.sidebar.subheader("Molecular Profile (Safety & Structure)")
    mol_wt = st.sidebar.slider("Molecular Weight (MolWt)", 0.0, 1000.0, 350.0)
    log_p = st.sidebar.slider("LogP", -5.0, 10.0, 3.0)
    num_h_donors = st.sidebar.slider("H-Bond Donors", 0, 20, 2)
    num_h_acceptors = st.sidebar.slider("H-Bond Acceptors", 0, 20, 4)

    if st.button("Design and Predict Drug Profile"):
        designed_params = np.array([mol_wt, log_p, num_h_donors, num_h_acceptors])
        
        best_match_smiles = find_closest_molecule(designed_params, molecules_db)
        toxicity_proba = model_xgb.predict_proba(designed_params.reshape(1, -1))[0][1]
        efficacy_score = anti_fat_effect + anti_inflammatory_effect + anti_fibrotic_effect
        
        final_effect = 0
        if toxicity_proba > 0.7: toxicity_risk = "High Risk"; final_effect = 20.0
        elif toxicity_proba > 0.3: toxicity_risk = "Moderate Risk"; final_effect = -1 * (efficacy_score * 0.5)
        else: toxicity_risk = "Low Risk (Safe)"; final_effect = -1 * (efficacy_score * 1.5)
            
        st.session_state.drug_alt_effect = final_effect
        st.session_state.drug_profile_name = "Newly Designed Drug"

        st.subheader("Newly Designed Drug Profile:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Safety", toxicity_risk)
            st.metric("Designed Efficacy Score", f"{efficacy_score}/30")
            st.info(f"Final Simulated ALT Effect: {final_effect:.2f}")
        with col2:
            st.write("**Matching Molecular Structure:**")
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{best_match_smiles}/PNG"
            st.image(url, use_container_width=True)
        st.success("The drug is ready for clinical trial simulation in Part 2.")

    # --- PART 2: CLINICAL TRIAL SIMULATION ON DIGITAL TWIN ---
    st.header("Part 2: Clinical Trial Simulation")
    # ... (Rest of the code for Part 2 is the same as before) ...
    st.write("Test your newly designed drug on a virtual patient.")
    st.subheader("Enter Patient Data")
    main_col1, main_col2 = st.columns(2)
    with main_col1:
        age = st.number_input("Age", 1, 120, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", 10.0, 60.0, 32.0, format="%.1f")
    with main_col2:
        ast_level = st.number_input("AST Level", 0, 500, 80)
        alt_month1 = st.number_input("ALT Level (Month 1)", 0, 500, 95)
        alt_month2 = st.number_input("ALT Level (Month 2)", 0, 500, 100)
        alt_month3 = st.number_input("ALT Level (Month 3)", 0, 500, 110)

    initial_status_text, initial_status_color = get_liver_status(alt_month3, ast_level, bmi)
    st.subheader("Virtual Liver - Initial Status")
    st.markdown(f"<div style='background-color:{initial_status_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;'>Current Status: <strong>{initial_status_text}</strong></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Run Simulation")
    apply_drug = st.checkbox(f"Apply the drug? (Current: **{st.session_state.drug_profile_name}**)")
    if st.button("Simulate 12-Month Trial"):
        input_seq = np.array([alt_month1, alt_month2, alt_month3])
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
        predictions_scaled = []
        current_batch = input_seq_scaled.reshape((1, 3, 1))
        with st.spinner("Simulating..."):
            for i in range(12):
                pred = model_lstm.predict(current_batch, verbose=0)[0]
                predictions_scaled.append(pred)
                current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)
        predictions_unscaled = scaler.inverse_transform(predictions_scaled)
        patient_adjustment = ((bmi - 25) * 0.5 if bmi > 25 else 0) + ((age - 40) * 0.1 if age > 40 else 0)
        patient_trend = np.linspace(0, patient_adjustment, 12)
        prediction_without_drug = predictions_unscaled.flatten() + patient_trend
        drug_effect_value = 0
        if apply_drug:
            drug_effect_value = st.session_state.drug_alt_effect
        drug_trend = np.linspace(0, drug_effect_value, 12)
        prediction_with_drug = prediction_without_drug + drug_trend
        prediction_with_drug = np.maximum(prediction_with_drug, 25)
        simulation_results = pd.DataFrame({
            "Without Drug (Placebo)": prediction_without_drug,
            "With Designed Drug": prediction_with_drug
        })
        st.subheader("Simulation Results Chart:")
        st.line_chart(simulation_results)
        st.markdown("---")
        st.subheader("Trial Summary & Efficacy Report")
        final_alt_no_drug = prediction_without_drug[-1]
        final_alt_with_drug = prediction_with_drug[-1]
        change = final_alt_with_drug - final_alt_no_drug
        final_ast_level = ast_level + (change * 0.2)
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            st.metric("Final ALT (Placebo)", f"{final_alt_no_drug:.2f}")
            st.metric("Final ALT (With Designed Drug)", f"{final_alt_with_drug:.2f}", delta=f"{change:.2f}")
        with report_col2:
            final_status_text, final_status_color = get_liver_status(final_alt_with_drug, final_ast_level, bmi)
            st.markdown(f"**Virtual Liver - Final Status**")
            st.markdown(f"<div style='background-color:{final_status_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;'>{final_status_text}</div>", unsafe_allow_html=True)

