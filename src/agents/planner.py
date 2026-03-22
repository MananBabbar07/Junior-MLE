import ollama
import ast
import re

def extract_code(raw_text):
    """Strips Markdown formatting (like ```python or ```json) from LLM responses."""
    text = raw_text.strip()
    # Remove the starting ```python, ```json, or just ```
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text, flags=re.IGNORECASE)
    # Remove the ending ```
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

class Planner:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model
    
    def analyze_data(self, schema_summary, sample_data, target):
        prompt = f"""
        You are an MLE Planner. Look at this dataset metadata:
        
        SCHEMA:
        {schema_summary}
        
        SAMPLE:
        {sample_data}
        
        GOAL: Predict '{target}'
        
        Respond with:
        1. Task Type (Regression/Classification)
        2. Preprocessing steps needed (e.g., handle NaNs, encode categories)
        3. Recommended initial model.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']
    
    def identify_task(self, schema, sample_data):
        prompt = f"""
        Analyze this dataset SCHEMA and SAMPLE:
        SCHEMA: {schema}
        SAMPLE: {sample_data}

        You are a Data Science Auditor. Identify:
        1. Which column is the most likely 'Target' variable?
        2. Is this a CLASSIFICATION or REGRESSION task?

        Return ONLY a valid Python dictionary. No chat.
        Example: {{'target': 'Survived', 'task': 'Classification'}}
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        
        # 1. Extract the raw string from the LLM
        raw_response = response['response']
        
        # 2. Strip out any Markdown code blocks
        clean_response = extract_code(raw_response)
        
        # 3. Safely evaluate the clean string into a Python dictionary
        discovery = ast.literal_eval(clean_response)
        
        return discovery