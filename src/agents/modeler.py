import ollama

class Modeler:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate_training_code(self, schema, target_col, task_type):
        prompt = f"""
        You are an Expert MLE. Write a Python script to build a baseline model.
        
        TASK TYPE: {task_type}
        TARGET: {target_col}
        SCHEMA: {schema}
        
        CRITICAL RULES:
        1. ALGORITHM: 
           - If {task_type} is Classification, use RandomForestClassifier and report Accuracy.
           - If {task_type} is Regression, use RandomForestRegressor and report RMSE and R2.
        2. DATA: Load from 'data/processed/cleaned_data.csv'. Separate X and y correctly.
        3. SAVE MODEL: Save the trained model to 'models/baseline_model.pkl' using joblib.
        4. FORMAT: ONLY output python code. No markdown. Start with 'import pandas as pd'.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']