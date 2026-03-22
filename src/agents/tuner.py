
import ollama

class Tuner:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate_tuning_code(self, schema, target_col, task_type):
        prompt = f"""
        You are an Expert ML Engineer specializing in Hyperparameter Tuning.
        Write a Python script to tune a model using GridSearchCV.
        
        TASK TYPE: {task_type}
        TARGET: {target_col}
        SCHEMA: {schema}
        
        CRITICAL RULES:
        1. LOAD FILE: Load from 'data/processed/cleaned_data.csv'.
        2. SCORING: 
           - If {task_type} is Classification, use scoring='accuracy'.
           - If {task_type} is Regression, use scoring='neg_mean_squared_error'.
        3. TUNE: Define a small parameter grid (max_depth, n_estimators) for the appropriate {task_type} model.
        4. OUTPUT: Print the best_params_ and best_score_.
        5. FORMAT: ONLY output python code. No markdown. Start with 'import pandas as pd'.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']