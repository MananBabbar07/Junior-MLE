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

class Tuner:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate_tuning_code(self, schema, target_col, task_type):
        """Generates a GPU-accelerated hyperparameter optimization script."""
        # Determine the correct XGBoost class
        algo = "XGBRegressor" if "Regression" in task_type else "XGBClassifier"
        
        prompt = f"""
        You are an Expert ML Engineer. Write a Python script for GPU-ACCELERATED Hyperparameter Tuning.
        
        DATA CONTEXT:
        - SOURCE: 'data/processed/cleaned_data.csv'
        - TARGET: '{target_col}'
        - TASK: {task_type}
        
        HARDWARE: NVIDIA RTX 3050 (Use CUDA).
        
        STRICT CODING REQUIREMENTS:
        1. IMPORTS: `import pandas as pd`, `from xgboost import {algo}`, `from sklearn.model_selection import GridSearchCV`, and `import joblib`.
        2. BASE MODEL: Initialize `{algo}(tree_method='hist', device='cuda', random_state=42)`.
        3. GRID PARAMETERS: 
           - 'max_depth': [3, 6, 9]
           - 'learning_rate': [0.01, 0.1]
           - 'n_estimators': [100, 500]
        4. CROSS-VALIDATION: Use `GridSearchCV` with `cv=3` and `verbose=1`.
        5. SCORING: Use {'accuracy' if 'Classification' in task_type else 'neg_mean_squared_error'}.
        6. RESULTS: Print the 'Best parameters' and 'Best score'.
        7. FINAL SAVE: Train the best estimator on the full set and save to 'models/tuned_model.joblib'.
        
        FORMATTING: Output ONLY raw python code. No markdown backticks. Start with 'import pandas as pd'.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return extract_code(response['response'])