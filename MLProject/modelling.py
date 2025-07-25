import mlflow
import mlflow.sklearn
import mlflow.models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Load data
df = pd.read_csv("MLProject/namadataset_preprocessing/winequality_preprocessed.csv")
X = df.drop("quality_label", axis=1)
y = df["quality_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set experiment
mlflow.set_experiment("Wine_Autolog")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model directly to MLProject/model (NOT log_model)
    output_dir = "MLProject/model"
    os.makedirs(output_dir, exist_ok=True)
    mlflow.sklearn.save_model(sk_model=model, path=output_dir)
