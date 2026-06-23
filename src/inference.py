from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI(
    title="Junior-MLE Inference Engine",
    description="High-speed API for autonomous XGBoost models.",
    version="1.0.0"
)

MODEL_PATH = "models/tuned_model.joblib"
model = None


class PredictionRequest(BaseModel):
    features: dict


@app.on_event("startup")
def load_model():
    global model

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found: {MODEL_PATH}")


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:
        input_data = pd.DataFrame(
            request.features,
            index=[0]
        )

        prediction = model.predict(input_data)

        return {
            "status": "success",
            "prediction": prediction[0].item()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )