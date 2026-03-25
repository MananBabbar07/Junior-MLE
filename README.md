# 🤖 Junior-MLE: Agentic AutoML

[![Ollama](https://img.shields.io/badge/LLM-Ollama-red.svg)](https://ollama.ai/)
[![Model](https://img.shields.io/badge/Model-Qwen--2.5--Coder-7b-blue.svg)](https://huggingface.co/Qwen)
[![Hardware](https://img.shields.io/badge/GPU-RTX--3050-green.svg)](https://nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Junior-MLE** is a local, autonomous, self-healing machine learning pipeline. It leverages Large Language Models (LLMs) to transform raw tabular data into optimized, deployment-ready models through a recursive "Think-Code-Correct" loop.

---

## 🏗️ Architecture

The system operates as a committee of specialized AI agents that handle the end-to-end ML lifecycle:

* **🕵️ Planner Agent**: Audits dataset metadata (schemas/samples) to dynamically detect the **Target Column** and **Task Type** (Classification vs. Regression).
* **💻 Coder Agent**: Generates production-grade Python scripts for data preprocessing, null imputation, and categorical encoding.
* **🛠️ Executor**: Orchestrates the local environment, runs generated scripts, and captures real-time logs.
* **⚖️ Critic Agent**: The "Self-Correction" brain. If a script fails, it analyzes the traceback, fixes the code logic, and triggers a retry.
* **📈 Tuner Agent**: Performs automated hyperparameter optimization using `GridSearchCV` to maximize baseline performance.

---

## 🛡️ Self-Healing in Action (The Critic Loop)

Unlike static scripts, Junior-MLE adapts to library deprecations, task pivots, and messy data in real-time. 

* **Library Deprecation Fix:** During a Classification run, the LLM generated deprecated Scikit-Learn code (`OneHotEncoder(sparse=False)`). The pipeline crashed. The **Critic Agent** intercepted the `TypeError`, analyzed the Python traceback, rewrote the syntax to the modern `sparse_output=False`, and successfully resumed the pipeline.
* **Autonomous Task Pivoting:** When fed the California Housing dataset, the pipeline dynamically recognized the continuous target variable, purged its default classification logic, and self-corrected its training scripts to utilize a `RandomForestRegressor` with MSE/R2 metrics.

---

## 📊 Benchmarked Datasets

The pipeline has been stress-tested on multiple distinct, real-world datasets of varying sizes with zero human intervention in the code generation or model selection process.

| Dataset | Volume (Rows) | Target Detected | Task Type | Final Tuned Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes Indicators** | 253,680 | `Diabetes_012` | Classification | **85.03% Accuracy** |
| **Bank Marketing** | 45,211 | `deposit` | Classification | **85.6% Accuracy** |
| **California Housing** | 20,640 | `median_house_value` | Regression | **0.817 R² Score** |
| **Travel Package** | ~4,900 | `ProdTaken` | Classification | **91.0% Accuracy** 🏆 |
| **Titanic Survival** | 891 | `Survived` | Classification | **83.1% Accuracy** |

---

## 🚀 Project Status

- [x] **Directory Architecture**: Modular source structure for high scalability.
- [x] **Metadata Extraction**: Automated schema auditing and task identification.
- [x] **Code Generation**: LLM-driven Python scripting for data preprocessing.
- [x] **Self-Healing Loop**: Real-time error handling and automated code correction.
- [x] **Model Tuning**: Automated baseline optimization.
- [ ] **Deployment Module**: (Planned) Automated FastAPI wrapper generation.

---

## 📂 Repository Roadmap

```text
junior-mle/
├── data/
│   ├── raw/               # 📥 Input CSVs (e.g., train.csv)
│   └── processed/         # 📤 Cleaned, encoded datasets
├── src/
│   ├── agents/            # 🧠 Planner, Coder, Critic, Modeler, Tuner
│   ├── tools/             # ⚙️ Script Executor
│   └── main.py            # 🚀 Core Orchestrator
├── generated_scripts/     # 📝 Live scripts written by the Agent
└── models/                # 🏆 Final .pkl model artifacts