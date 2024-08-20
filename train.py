# train.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple RandomForest model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save the model to a file
os.makedirs('/model', exist_ok=True)
model_path = '/model/iris_rf_model.pkl'
joblib.dump(clf, model_path)

print(f"Model saved to {model_path}")
