import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("MLProject/namadataset_preprocessing/winequality_preprocessed.csv")
X = df.drop("quality_label", axis=1)
y = df["quality_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Set experiment
mlflow.set_experiment("Wine_Autolog")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Simpan model secara eksplisit
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  
        registered_model_name="wine_model"
    )
