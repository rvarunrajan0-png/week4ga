import os
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "models/model_v1.joblib"
DATA_V1 = "data/v1.csv"
DATA_V2 = "data/v2.csv"

def test_training_creates_model():
    """Test that train.py runs and produces a saved model."""
    os.system("python train.py")
    assert os.path.exists(MODEL_PATH), "Model file not created!"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded object is not a model!"

def test_model_prediction_sanity():
    """Test model can make a valid prediction on sample input."""
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_V1)
    X_sample = df.iloc[:1, :-1].values
    pred = model.predict(X_sample)
    assert len(pred) == 1, "Model did not return a single prediction!"
    assert pred is not None, "Model prediction failed!"

def test_augmentation_increases_rows():
    """Test that augment.py increases dataset size."""
    df_before = pd.read_csv(DATA_V2)
    n_before = len(df_before)
    os.system("python augment.py")
    df_after = pd.read_csv(DATA_V2)
    n_after = len(df_after)
    assert n_after > n_before, f"Augmentation failed: before={n_before}, after={n_after}"