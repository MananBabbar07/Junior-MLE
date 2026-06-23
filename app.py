import streamlit as st
import pandas as pd
import os
import subprocess
import requests
import json
import time

st.set_page_config(
    page_title="Junior-MLE | Control Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("generated_scripts", exist_ok=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    st.title("AutoML Platform")
    st.caption("Containerized MLOps Architecture")
    
    st.divider()
    st.markdown("### ⚙️ Microservice Status")
    st.success("✅ Frontend: Streamlit UI")
    st.success("✅ Backend: FastAPI (Port 8000)")
    st.success("✅ Inference: CPU Optimized")
    
    st.divider()
    st.markdown("### 🧠 Training Swarm")
    st.markdown("- **🕵️ Planner:** Strategy & Routing")
    st.markdown("- **💻 Coder:** Pandas & XGBoost")
    st.markdown("- **⚖️ Tuner:** GridSearchCV")
    st.markdown("- **🛡️ Critic:** Auto-Debugging")

tab1, tab2 = st.tabs(["🚀 Agentic Training Pipeline", "🔮 Universal Inference Engine"])


with tab1:
    st.title("🤖 Junior-MLE: Autonomous Training Interface")
    st.markdown("Upload a raw dataset and watch the multi-agent system analyze, clean, train, and tune a Machine Learning model in real-time.")

    uploaded_file = st.file_uploader("Drop your dataset here (CSV format)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv("data/raw/train.csv", index=False)
        
        target_col = df.columns[-1] 
        schema = {"numerical": {}, "categorical": {}}
        
        for col in df.columns:
            if col == target_col:
                continue
            
            
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
          
                if min_val == max_val:
                    max_val += 1.0 
                    
                schema["numerical"][col] = {
                    "min": min_val,
                    "max": max_val,
                    "mean": float(df[col].mean())
                }
            else:
                unique_vals = df[col].dropna().astype(str).unique().tolist()
                schema["categorical"][col] = unique_vals
                
        with open("models/schema.json", "w") as f:
            json.dump(schema, f, indent=4)
       
        
        with st.expander("🔍 Preview Raw Data (First 5 Rows)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns. Target Assumed: '{target_col}'")

        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_training = st.button("🚀 Deploy Agents & Start Training Pipeline", use_container_width=True, type="primary")

        if start_training:
            st.markdown("### 📡 Live Agent Console")
            console_container = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("Agents are initializing (GPU/Local LLM engaging)..."):
                
             
                run_id = int(time.time())
                log_file_path = f"pipeline_{run_id}.log"
                
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write("System Boot Sequence Initiated...\n")
        
                log_file = open(log_file_path, "a", encoding="utf-8")
                
                process = subprocess.Popen(
                    ["python", "-u", "-X", "utf8", "-m", "src.main"],
                    stdout=log_file,
                    stderr=log_file,  
                    bufsize=1,
                    universal_newlines=True
                )
                
               
                last_pos = 0
                log_text = ""
                
                while True:
                    try:
                        
                        with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new_lines = f.readlines()
                            last_pos = f.tell()
                            
                        if new_lines:
                            for line in new_lines:
                                log_text += line
                                
                              
                                lower_line = line.lower()
                                if "target" in lower_line or "planner" in lower_line: progress_bar.progress(20)
                                if "strategy" in lower_line or "coder" in lower_line: progress_bar.progress(40)
                                if "clean" in lower_line or "imput" in lower_line: progress_bar.progress(60)
                                if "train" in lower_line or "fit" in lower_line: progress_bar.progress(80)
                                if "optimiz" in lower_line or "complete" in lower_line: progress_bar.progress(100)
                            
                            # Keep only the last 30 lines so the UI doesn't lag
                            display_lines = log_text.split('\n')
                            if len(display_lines) > 30:
                                display_text = '\n'.join(display_lines[-30:])
                            else:
                                display_text = log_text
                                
                            console_container.code(display_text, language="bash")
                            
                    except PermissionError:
                       
                        pass
                  
                    if process.poll() is not None:
                        break
                        
                    
                    time.sleep(0.5)
                
                
                log_file.close()
                    
            if process.returncode == 0:
                progress_bar.progress(100)
                st.success("🎉 Pipeline Executed Successfully!")
                st.balloons()
                st.markdown("### 🏆 Final Tuned Metrics")
                col_a, col_b = st.columns(2)
                with col_a: st.info("📦 **Tuned Model Saved:** `models/tuned_model.joblib`")
                with col_b: st.info("🧹 **Cleaned Data Saved:** `data/processed/cleaned_data.csv`")
            else:
                st.error("🚨 Critical Failure: The agents encountered an error they could not self-heal.")


with tab2:
    st.title("⚡ Universal AI Predictor")
    st.markdown("This form is dynamically generated based on the dataset you uploaded in Tab 1. Predictions are served via the Docker CPU Microservice.")

    if not os.path.exists("models/schema.json"):
        st.warning("⚠️ No dataset schema found. Please upload a dataset in the 'Agentic Training Pipeline' tab first.")
    else:
        with open("models/schema.json", "r") as f:
            schema = json.load(f)

        with st.form("dynamic_prediction_form"):
            st.subheader("Input Features")
            
            user_inputs = {}
            ui_cols = st.columns(3)
            col_idx = 0
            
            
            for col_name, stats in schema.get("numerical", {}).items():
                with ui_cols[col_idx % 3]:
                    user_inputs[col_name] = st.number_input(
                        col_name,
                        min_value=stats["min"],
                        max_value=stats["max"],
                        value=stats["mean"]
                    )
                col_idx += 1
                
          
            for col_name, options in schema.get("categorical", {}).items():
                with ui_cols[col_idx % 3]:
                    user_inputs[col_name] = st.selectbox(col_name, options)
                col_idx += 1

            st.divider()
            submitted = st.form_submit_button("🔮 Predict Outcome", type="primary", use_container_width=True)

        if submitted:
            
            payload_features = {}
            
            for col_name, val in user_inputs.items():
                if col_name in schema.get("numerical", {}):
                    payload_features[col_name] = val
                else:
                  
                    for option in schema["categorical"][col_name]:
                        ohe_key = f"{col_name}_{option}"
                        payload_features[ohe_key] = 1.0 if val == option else 0.0

            payload = {"features": payload_features}

            
            try:
                with st.spinner("Pinging Docker API..."):
                    response = requests.post("http://localhost:8000/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    
                    st.success(f"### 🎯 Model Prediction: `{prediction}`")
                        
                    with st.expander("View Dynamic Backend JSON Payload (Dev Mode)"):
                        st.json(payload)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Refused. Is your Docker container running? (Run: `docker run -p 8000:8000 junior-mle-api`)")