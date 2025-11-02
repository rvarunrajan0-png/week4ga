import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    df = pd.read_csv("data/v1.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    model = joblib.load("models/model_v1.joblib")
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"Model accuracy: {acc}")
    assert acc > 0.7, f"Model accuracy too low: {acc}"
