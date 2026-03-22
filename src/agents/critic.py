import ollama

class Critic:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def analyze_error(self, code, error_logs):
        prompt = f"""
        The following Python code failed with an error. 
        
        CODE:
        {code}
        
        ERROR LOGS:
        {error_logs}
        
        Identify the fix. Output ONLY the corrected Python code. 
        No explanations. No markdown backticks.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']