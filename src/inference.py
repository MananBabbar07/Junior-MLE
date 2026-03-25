from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn

# 1. Initialize the API
app = FastAPI(
    title="Junior-MLE Inference Engine",
    description="High-speed API for autonomous XGBoost models.",
    version="1.0.0"
)

# 2. Global Model Variable
MODEL_PATH = "models/tuned_model.joblib"
model = None

# 3. Define the Input Schema
class PredictionRequest(BaseModel):
    # Expects a dictionary of features: {"Age": 45, "BMI": 28.5, "HighBP": 1}
    features: dict 

# 4. Load the model on startup
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")

# 5. The Prediction Endpoint
@app.post("/predict")
def make_prediction(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert the incoming JSON dictionary directly into a Pandas DataFrame
        # The index=[0] is required because we are passing a single row of data
        input_data = pd.DataFrame(request.features, index=[0])
        
        # Execute the prediction
        prediction = model.predict(input_data)
        
        # XGBoost returns numpy arrays, we convert to standard Python types (int/float) for JSON
        result = prediction[0].item() 
        
        return {
            "status": "success",
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Allow running directly for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)