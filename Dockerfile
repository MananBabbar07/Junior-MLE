# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the core dependencies list
# (Make sure to create a requirements.txt with fastapi, uvicorn, xgboost, pandas, scikit-learn, joblib)
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script and the trained model into the container
COPY src/inference.py /app/src/
COPY models/tuned_model.joblib /app/models/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the API using Uvicorn
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]