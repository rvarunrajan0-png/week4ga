import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/v1.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
m = RandomForestClassifier(n_estimators=50, random_state=42)
m.fit(X_train,y_train)
joblib.dump(m, "models/model_v1.joblib")
print("Saved models/model_v1.joblib")