import streamlit as st
import pandas as pd
import os
import subprocess
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Junior-MLE | Control Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ensure Directories Exist ---
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("generated_scripts", exist_ok=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    st.title("AutoML Agent")
    st.caption("Powered by Qwen 2.5 Coder & NVIDIA RTX")
    
    st.divider()
    st.markdown("### ⚙️ Hardware Status")
    st.success("✅ NVIDIA RTX 3050 Detected")
    st.success("✅ Ollama Local Server Active")
    st.success("✅ Docker Engine Active")
    
    st.divider()
    st.markdown("### 🧠 Active Agents")
    st.markdown("- **🕵️ Planner:** Strategy & Routing")
    st.markdown("- **💻 Coder:** Pandas & XGBoost")
    st.markdown("- **⚖️ Tuner:** GridSearchCV")
    st.markdown("- **🛡️ Critic:** Auto-Debugging")

# --- Main UI Tabs ---
tab1, tab2 = st.tabs(["🚀 Agentic Training Pipeline", "🔮 Live Inference Form"])

# ==========================================
# TAB 1: THE TRAINING PIPELINE
# ==========================================
with tab1:
    st.title("🤖 Junior-MLE: Autonomous Training Interface")
    st.markdown("Upload a raw dataset and watch the multi-agent system analyze, clean, train, and tune a GPU-accelerated Machine Learning model in real-time.")

    uploaded_file = st.file_uploader("Drop your dataset here (CSV format)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv("data/raw/train.csv", index=False)
        
        with st.expander("🔍 Preview Raw Data (First 5 Rows)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_training = st.button("🚀 Deploy Agents & Start Training Pipeline", use_container_width=True, type="primary")

        if start_training:
            st.markdown("### 📡 Live Agent Console")
            console_container = st.empty()
            log_text = ""
            progress_bar = st.progress(0)
            
            with st.spinner("Agents are initializing..."):
                process = subprocess.Popen(
                    ["python", "-X", "utf8", "-m", "src.main"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    bufsize=1
                )
                
                for line in process.stdout:
                    log_text += line
                    console_container.code(log_text, language="bash")
                    
                    if "Target Detected" in line: progress_bar.progress(20)
                    if "Strategy acquired" in line: progress_bar.progress(40)
                    if "Data cleaned successfully" in line: progress_bar.progress(60)
                    if "Model Training Complete" in line: progress_bar.progress(80)
                    if "OPTIMIZATION COMPLETE" in line: progress_bar.progress(100)
                    
                process.wait() 
            
            if process.returncode == 0:
                st.success("🎉 Pipeline Executed Successfully!")
                st.balloons()
                st.markdown("### 🏆 Final Tuned Metrics")
                col_a, col_b = st.columns(2)
                with col_a: st.info("📦 **Tuned Model Saved:** `models/tuned_model.joblib`")
                with col_b: st.info("🧹 **Cleaned Data Saved:** `data/processed/cleaned_data.csv`")
            else:
                st.error("🚨 Critical Failure: The agents encountered an error they could not self-heal.")

# ==========================================
# TAB 2: THE INFERENCE UI FORM
# ==========================================
with tab2:
    st.title("⚡ Sales Predictor: Customer Intent Engine")
    st.markdown("Enter a prospective customer's details below to instantly predict if they will purchase the travel package.")

    with st.form("prediction_form"):
        st.subheader("Customer Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=18, max_value=80, value=29)
            gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            income = st.number_input("Monthly Income ($)", min_value=10000, max_value=300000, value=38000, step=1000)

        with col2:
            st.markdown("**Pitch & Contact History**")
            contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
            designation = st.selectbox("Customer Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            duration = st.number_input("Pitch Duration (Minutes)", min_value=1, max_value=60, value=32)
            pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 5)
            follow_ups = st.slider("Number of Follow-ups", 1, 6, 5)

        with col3:
            st.markdown("**Travel Preferences**")
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            trips = st.number_input("Previous Trips", min_value=0, max_value=20, value=6)
            passport = st.selectbox("Has Passport?", ["Yes", "No"])
            prop_star = st.slider("Preferred Property Rating", 3, 5, 5)
            persons = st.number_input("Persons Visiting", min_value=1, max_value=5, value=2)
            children = st.number_input("Children Visiting", min_value=0, max_value=5, value=0)
            car = st.selectbox("Owns a Car?", ["Yes", "No"])

        st.divider()
        submitted = st.form_submit_button("🔮 Predict Purchase Intent", type="primary", use_container_width=True)

    if submitted:
        # Convert UI choices to One-Hot Encoded logic for the backend
        payload = {
            "features": {
                "Age": age,
                "CityTier": city_tier,
                "DurationOfPitch": duration,
                "NumberOfPersonVisiting": persons,
                "NumberOfFollowups": follow_ups,
                "PreferredPropertyStar": prop_star,
                "NumberOfTrips": trips,
                "Passport": 1 if passport == "Yes" else 0,
                "PitchSatisfactionScore": pitch_score,
                "OwnCar": 1 if car == "Yes" else 0,
                "NumberOfChildrenVisiting": children,
                "MonthlyIncome": income,
                
                # Dynamic One-Hot Encoding
                "TypeofContact_Company Invited": 1 if contact == "Company Invited" else 0,
                "TypeofContact_Self Enquiry": 1 if contact == "Self Enquiry" else 0,
                
                "Occupation_Free Lancer": 1 if occupation == "Free Lancer" else 0,
                "Occupation_Large Business": 1 if occupation == "Large Business" else 0,
                "Occupation_Salaried": 1 if occupation == "Salaried" else 0,
                "Occupation_Small Business": 1 if occupation == "Small Business" else 0,
                
                "Gender_Fe Male": 1 if gender == "Fe Male" else 0,
                "Gender_Female": 1 if gender == "Female" else 0,
                "Gender_Male": 1 if gender == "Male" else 0,
                
                "MaritalStatus_Divorced": 1 if marital == "Divorced" else 0,
                "MaritalStatus_Married": 1 if marital == "Married" else 0,
                "MaritalStatus_Single": 1 if marital == "Single" else 0,
                "MaritalStatus_Unmarried": 1 if marital == "Unmarried" else 0,
                
                "Designation_AVP": 1 if designation == "AVP" else 0,
                "Designation_Executive": 1 if designation == "Executive" else 0,
                "Designation_Manager": 1 if designation == "Manager" else 0,
                "Designation_Senior Manager": 1 if designation == "Senior Manager" else 0,
                "Designation_VP": 1 if designation == "VP" else 0
            }
        }

        # Send to Docker FastAPI
        try:
            with st.spinner("Analyzing Customer Profile..."):
                response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                # Beautiful Result Display
                if prediction == 1.0:
                    st.success("### 🎯 High Intent Customer: Purchase Likely!")
                    st.markdown("This profile matches historical buyers. Assign to a senior sales rep immediately.")
                    st.balloons()
                else:
                    st.warning("### 🧊 Low Intent Customer: Purchase Unlikely")
                    st.markdown("This profile historically does not convert. Consider sending automated marketing emails instead of a direct sales call.")
                    
                with st.expander("View Backend JSON Payload (Dev Mode)"):
                    st.json(payload)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Refused. Is your Docker container running? (Run: `docker run -p 8000:8000 junior-mle-api`)")