# 🤖 Junior-MLE: Autonomous End-to-End MLOps Pipeline

[![Ollama](https://img.shields.io/badge/LLM-Ollama-red.svg)](https://ollama.ai/)
[![Hardware](https://img.shields.io/badge/GPU-RTX--3050-green.svg)](https://nvidia.com)
[![ML Engine](https://img.shields.io/badge/Engine-XGBoost-f37021.svg)](https://xgboost.readthedocs.io/)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Deploy](https://img.shields.io/badge/Deploy-Docker-2496ED.svg)](https://www.docker.com/)

---

## 🚀 Overview

**Junior-MLE** is a local, autonomous, self-healing MLOps pipeline that acts like an AI Machine Learning Engineer.

It transforms raw tabular data → generates ML code → debugs itself → trains optimized models → deploys them instantly via a containerized API.



---

## 🌟 Key Features

- 🔒 100% Local & Private (Ollama + Qwen 2.5 Coder)
- 🤖 Autonomous Code Generation
- ♻️ Self-Healing Execution (auto-debugging)
- ⚡ GPU-Optimized Training (XGBoost + CUDA)
- 🐳 Instant Docker Deployment (FastAPI)

---

## 🏗️ System Architecture

### 🧠 Multi-Agent Training Swarm
- 🕵️ Planner Agent → dataset understanding  
- 💻 Coder Agent → pipeline generation  
- ⚖️ Critic Agent → error fixing  
- 📈 Tuner Agent → hyperparameter optimization  

### ⚡ Backend (FastAPI + Docker)
- Model exported as `.joblib`
- Served via REST API

### 🎛️ Frontend (Streamlit)
- Upload dataset
- Monitor agents
- Run predictions

---

## 📊 Performance Benchmarks

| Dataset | Rows | Task | Metric |
|--------|------|------|--------|
| Bank Marketing | 45,211 | Classification | **94.0% Accuracy** 🏆 |
| Travel Package | ~4,900 | Classification | **91.0% Accuracy** |
| Diabetes | 253,680 | Classification | **85.03% Accuracy** |
| Titanic | 891 | Classification | **83.1% Accuracy** |
| Housing | 20,640 | Regression | **0.817 R²** |

---

## ⚡ Setup & Run (Full Pipeline)

```bash
git clone https://github.com/MananBabbar07/junior-mle.git
cd junior-mle
pip install -r requirements.txt
docker build -t junior-mle-api .
docker run -p 8000:8000 junior-mle-api
```

Open another terminal and run:

```bash
streamlit run app.py
```

---

## 🎯 Test the System

- Open → http://localhost:8501  
- Go to → **🔮 Live Inference Form**  
- Click → **Predict Purchase Intent**

---

## 🔄 End-to-End Flow

Dataset → AI Agents → Code Generation → Self-Debug → Model Training → Docker API → Streamlit UI

---

## 🔌 API Reference

POST /predict

```json
{
"features":{
"Age":29
"CityTier":1
"DurationOfPitch":32
"NumberOfPersonVisiting":2
"NumberOfFollowups":5
"PreferredPropertyStar":5
"NumberOfTrips":6
"Passport":1
"PitchSatisfactionScore":5
"OwnCar":1
"NumberOfChildrenVisiting":0
"MonthlyIncome":38000
"TypeofContact_Company Invited":0
"TypeofContact_Self Enquiry":1
"Occupation_Free Lancer":0
"Occupation_Large Business":0
"Occupation_Salaried":1
"Occupation_Small Business":0
"Gender_Fe Male":0
"Gender_Female":0
"Gender_Male":1
"MaritalStatus_Divorced":0
"MaritalStatus_Married":0
"MaritalStatus_Single":1
"MaritalStatus_Unmarried":0
"Designation_AVP":0
"Designation_Executive":1
"Designation_Manager":0
"Designation_Senior Manager":0
"Designation_VP":0
}
}
```

Response:

```json
{
  "status": "success",
  "prediction": 1.0
}
```

---

## 📂 Project Structure

junior-mle/
├── app.py  
├── Dockerfile  
├── requirements.txt  
├── data/  
├── src/  
├── models/  
├── generated_scripts/  

---

## 👨‍💻 Author

**Manan Babbar**  
https://github.com/MananBabbar07

---

## ⭐ Support

If you like this project:
- Star ⭐ the repo  
- Fork 🍴 it  
- Build on top 🚀
