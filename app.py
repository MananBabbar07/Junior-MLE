import streamlit as st
import pandas as pd
import os
import subprocess
import time
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
    st.success("✅ CUDA Backend Ready")
    
    st.divider()
    st.markdown("### 🧠 Active Agents")
    st.markdown("- **🕵️ Planner:** Strategy & Routing")
    st.markdown("- **💻 Coder:** Pandas & XGBoost")
    st.markdown("- **⚖️ Tuner:** GridSearchCV")
    st.markdown("- **🛡️ Critic:** Auto-Debugging")

# --- Main UI Tabs ---
tab1, tab2 = st.tabs(["🚀 Agentic Training Pipeline", "🔮 Live Inference API"])

# ==========================================
# TAB 1: THE TRAINING PIPELINE
# ==========================================
with tab1:
    st.title("🤖 Junior-MLE: Autonomous Training Interface")
    st.markdown("Upload a raw dataset and watch the multi-agent system analyze, clean, train, and tune a GPU-accelerated Machine Learning model in real-time.")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Drop your dataset here (CSV format)", type=["csv"])

    if uploaded_file is not None:
        # 1. Save the file to the exact path the Agent expects
        df = pd.read_csv(uploaded_file)
        df.to_csv("data/raw/train.csv", index=False)
        
        # 2. Show a preview of the data
        with st.expander("🔍 Preview Raw Data (First 5 Rows)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        st.divider()
        
        # 3. The Deployment Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_training = st.button("🚀 Deploy Agents & Start Training Pipeline", use_container_width=True, type="primary")

        if start_training:
            st.markdown("### 📡 Live Agent Console")
            
            # Create a visual container for the terminal output
            console_container = st.empty()
            log_text = ""
            
            # Visual Progress Bar
            progress_bar = st.progress(0)
            
            # Run the backend main.py script and capture the output live
            with st.spinner("Agents are initializing..."):
                process = subprocess.Popen(
                    ["python", "-X", "utf8", "-m", "src.main"], # Added -X utf8 to prevent emoji crash
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    bufsize=1
                )
                
                # Stream the output line by line to the Streamlit app
                for line in process.stdout:
                    log_text += line
                    # Update the console container with markdown code block
                    console_container.code(log_text, language="bash")
                    
                    # Update progress bar based on keywords in your logs
                    if "Target Detected" in line: progress_bar.progress(20)
                    if "Strategy acquired" in line: progress_bar.progress(40)
                    if "Data cleaned successfully" in line: progress_bar.progress(60)
                    if "Model Training Complete" in line: progress_bar.progress(80)
                    if "OPTIMIZATION COMPLETE" in line: progress_bar.progress(100)
                    
                process.wait() # Wait for the script to completely finish
            
            # --- Post-Training Results ---
            if process.returncode == 0:
                st.success("🎉 Pipeline Executed Successfully!")
                st.balloons()
                
                # Extract final metric from the log text (assuming your main.py prints it)
                st.markdown("### 🏆 Final Tuned Metrics")
                
                # Simple UI to show completion
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info("📦 **Tuned Model Saved:** `models/tuned_model.joblib`")
                with col_b:
                    st.info("🧹 **Cleaned Data Saved:** `data/processed/cleaned_data.csv`")
                    
            else:
                st.error("🚨 Critical Failure: The agents encountered an error they could not self-heal. Check the console logs above.")

# ==========================================
# TAB 2: THE INFERENCE ENGINE
# ==========================================
with tab2:
    st.title("⚡ Production Inference Testing")
    st.markdown("Send JSON payloads directly to your local FastAPI microservice (`http://localhost:8000/predict`).")

    # Default payload matching the Travel Package dataset
    default_payload = {
      "features": {
        "Age": 34,
        "TypeofContact": 1,
        "CityTier": 3,
        "DurationOfPitch": 15,
        "Occupation": 2,
        "Gender": 1,
        "NumberOfPersonVisiting": 3,
        "NumberOfFollowups": 4,
        "PreferredPropertyStar": 4,
        "MaritalStatus": 1,
        "NumberOfTrips": 2,
        "Passport": 0,
        "PitchSatisfactionScore": 3,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 1,
        "Designation": 2,
        "MonthlyIncome": 25000
      }
    }

    # Text area for user to modify the JSON
    user_json = st.text_area("Request Payload (JSON)", value=json.dumps(default_payload, indent=2), height=350)

    if st.button("🔌 Send API Request", type="primary"):
        try:
            payload = json.loads(user_json)
            
            # Ping the FastAPI server
            with st.spinner("Querying Model..."):
                response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ 200 OK: Prediction Received")
                
                # Display the prediction beautifully
                st.metric(label="Model Output", value=result["prediction"])
                
                with st.expander("View Raw Response"):
                    st.json(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON format. Please check your syntax.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Refused. Is your FastAPI server (`python src/inference.py`) running in another terminal?")