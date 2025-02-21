import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Convert input data to a pandas DataFrame for better logging
X = pd.DataFrame(X, columns=feature_names)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation settings
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Start an MLflow run with a specific run name
with mlflow.start_run(run_name="RandomForest_CrossValidation_Iris") as run:
    try:
        fold_metrics = []

        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

            # Log each fold's metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"fold_{fold}_{metric_name}", metric_value)

            fold_metrics.append(metrics)

            # Generate and log a confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Fold {fold}")

            # Save plot as an artifact
            cm_filename = f"confusion_matrix_fold_{fold}.png"
            plt.savefig(cm_filename)
            
            if os.path.isfile(cm_filename):
                mlflow.log_artifact(cm_filename)
                os.remove(cm_filename)
            plt.close()

        # Log average metrics
        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        for metric_name, metric_value in avg_metrics.items():
            mlflow.log_metric(f"avg_{metric_name}", metric_value)

        # Log model parameters
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # Provide an input example for model signature
        input_example = X.sample(1)

        # Save and log the final model as an artifact with input example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print(f"Model training and logging completed successfully. Run ID: {run.info.run_id}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.end_run(status="FAILED")
