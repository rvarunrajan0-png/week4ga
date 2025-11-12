from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("models/model_v1.joblib")

class Features(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "IRIS API is live!"}

@app.post("/predict")
def predict(data: Features):
    #predict
    prediction = model.predict([data.features])
    return {"prediction": prediction[0]}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
