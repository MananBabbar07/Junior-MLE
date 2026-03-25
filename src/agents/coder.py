import ollama
import re

def extract_code(raw_text):
    """Strips Markdown formatting from LLM responses."""
    text = raw_text.strip()
    # Remove the starting ```python, ```json, or just ```
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text, flags=re.IGNORECASE)
    # Remove the ending ```
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

class Coder:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate_code(self, plan, schema, target_col):
        """Generates the DATA CLEANING script (clean_data.py)."""
        prompt = f"""
        You are an Expert MLE Coder. Write a Python script for DATA CLEANING.
        
        PLAN: {plan}
        SCHEMA: {schema}
        
        STRICT RULES:
        1. LOAD DATA: Use `pd.read_csv('data/raw/train.csv')`. No other path.
        2. TARGET PRESERVATION: Keep '{target_col}' in the final dataframe.
        3. DROP TRASH: Drop unique IDs, names, or hashes.
        4. ENCODE CATEGORICALS: Use pd.get_dummies or LabelEncoder so all data is NUMERIC (Mandatory for GPU XGBoost).
        5. MISSING VALUES: Impute numerical (median) and categorical (mode).
        6. SAVE: Save result to 'data/processed/cleaned_data.csv' using index=False.
        
        STRICT LIMITATION:
        - DO NOT include any Machine Learning models (No XGBoost, No Scikit-Learn classifiers).
        - ONLY perform data manipulation and saving.
        
        FORMATTING: Output ONLY raw python code. No backticks. Start with 'import pandas as pd'.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return extract_code(response['response'])

    def generate_training_script(self, task_type, target_col, processed_data_path):
        """Generates the GPU-ACCELERATED TRAINING script (train_model.py)."""
        # Determine the correct XGBoost class
        algo = "XGBRegressor" if "Regression" in task_type else "XGBClassifier"
        
        prompt = f"""
        Write a Python script to train an {algo} using NVIDIA GPU acceleration.
        
        TASK: {task_type}
        TARGET: {target_col}
        DATA: {processed_data_path}
        
        MANDATORY GPU REQUIREMENTS:
        1. IMPORT: `import xgboost as xgb` and `import joblib`.
        2. INITIALIZE: `model = xgb.{algo}(tree_method='hist', device='cuda', n_estimators=1000)`.
        3. EXECUTION: Load the CSV, separate X and y (target is '{target_col}'), and fit the model.
        4. SAVE: Save the model to 'models/baseline_model.joblib' using joblib.
        
        FORMATTING: Output ONLY raw python code. No markdown backticks.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return extract_code(response['response'])