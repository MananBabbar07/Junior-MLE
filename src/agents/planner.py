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
        """Generates a high-level strategy for the Coder and Tuner agents."""
        prompt = f"""
        You are an Expert MLE Planner. Look at this dataset metadata:
        
        SCHEMA:
        {schema_summary}
        
        SAMPLE:
        {sample_data}
        
        GOAL: Predict '{target}'
        
        HARDWARE ENVIRONMENT: NVIDIA RTX 3050 GPU
        
        MANDATORY STRATEGY RULES:
        1. Identify Task Type (Regression/Classification).
        2. Identify Preprocessing (Handle NaNs, Encode Categories to Numeric).
        3. RECOMMENDED MODEL: You MUST use XGBoost (XGBClassifier or XGBRegressor).
        4. GPU ACCELERATION: Explicitly instruct the Coder to use `tree_method='hist'` and `device='cuda'`.
        5. DATA PATHS: Instruct the Coder to load from 'data/raw/train.csv' and save to 'data/processed/cleaned_data.csv'.
        
        Respond with a clear step-by-step strategy for the Coder.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']
    
    def identify_task(self, schema, sample_data):
        """Audits the raw data to find the target column and task type."""
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
        
        try:
            # 3. Safely evaluate the clean string into a Python dictionary
            discovery = ast.literal_eval(clean_response)
            return discovery
        except (ValueError, SyntaxError):
            # Fallback: Use Regex to find the dictionary if the LLM included extra text
            dict_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if dict_match:
                try:
                    return ast.literal_eval(dict_match.group())
                except:
                    pass
            
            # Final Fallback if everything fails
            return {"target": "unknown", "task": "Classification"}