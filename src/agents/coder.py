import ollama

class Coder:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate_code(self, plan, schema, target_col):
        prompt = f"""
        You are an Expert MLE Coder. Based on this PLAN and SCHEMA, write a Python script.
        
        PLAN:
        {plan}
        
        SCHEMA:
        {schema}
        
        CRITICAL RULES FOR DATA CLEANING:
        1. TARGET COLUMN PRESERVATION: You MUST keep the target column ('{target_col}') in the final cleaned dataframe. NEVER drop it.
        2. DROP TRASH: Identify and drop high-cardinality string columns (e.g., unique IDs, names, or hashes) that have no predictive power.
        3. ENCODE CATEGORICALS: Ensure all remaining categorical columns are properly encoded into numbers.
        4. MISSING VALUES: Impute missing values for numerical columns (median) and categorical columns (mode).
        5. SAVE THE FILE: You MUST save the final cleaned dataframe (including '{target_col}') to 'data/processed/cleaned_data.csv' using pandas to_csv(index=False).
        
        FORMATTING REQUIREMENTS:
        1. Use pandas to load the data from 'data/raw/train.csv'.
        2. ONLY output the python code. No explanations. No markdown backticks. 
        3. Start directly with 'import pandas as pd'.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']