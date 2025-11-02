from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model_v1.joblib")

@app.get("/")
def root():
    return {"message": "IRIS API is live!"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
