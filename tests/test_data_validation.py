import pandas as pd

def test_data_shape():
    df = pd.read_csv("data/v1.csv")
    # Iris dataset should have 4 features + 1 label column
    assert df.shape[1] == 5, "Dataset should have 5 columns!"

def test_no_missing_values():
    df = pd.read_csv("data/v1.csv")
    assert not df.isnull().values.any(), "Data contains missing values!"
