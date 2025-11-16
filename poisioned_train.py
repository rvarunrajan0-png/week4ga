import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def poison_data(X, y, poison_fraction):
    X = X.copy()
    y = y.copy()
    n_samples = int(len(X) * poison_fraction)
    if n_samples == 0:
        return X, y
    idx = np.random.choice(len(X), n_samples, replace=False)
    X.iloc[idx] = X.iloc[idx] + np.random.normal(0, 5, X.iloc[idx].shape)
    y.iloc[idx] = np.random.choice(y.unique(), size=n_samples)
    return X, y

def train_and_log(X_train, y_train, X_val, y_val, poison_fraction):
    label = "clean" if poison_fraction == 0 else f"poison_{poison_fraction}"
    safe_label = label.replace(".", "_")
    with mlflow.start_run(run_name=label):
        Xp, yp = poison_data(X_train, y_train, poison_fraction)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(Xp, yp)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_param("poison_fraction", poison_fraction)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, safe_label)
        print(f"Logged run '{label}' — accuracy={acc:.4f}")
        return acc

def main():
    train_df = pd.read_csv("data/v1.csv")
    label_col = "species"
    X_train = train_df.drop(label_col, axis=1)
    y_train = train_df[label_col]

    val_df = pd.read_csv("data/iris.csv")
    X_val = val_df.drop(label_col, axis=1)
    y_val = val_df[label_col]

    mlflow.set_tracking_uri("http://34.180.13.149:5000")
    mlflow.set_experiment("iris_poisoning_study")

    train_and_log(X_train, y_train, X_val, y_val, poison_fraction=0.0)
    for p in [0.05, 0.10, 0.50]:
        train_and_log(X_train, y_train, X_val, y_val, poison_fraction=p)

    print("Done. View results in MLflow UI.")

if __name__ == "__main__":
    main()
