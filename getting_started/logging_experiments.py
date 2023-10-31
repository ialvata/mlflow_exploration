import mlflow
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
from mlflow.models import infer_signature


###################                Setting MLFlow environment             #####################
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Sets the current active experiment to the "Wine_Models" experiment and
# returns the Experiment metadata
wine_experiment = mlflow.set_experiment("Wine_Models")

# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
run_name = "wine_rand_forest"

# Define an artifact path that the model will be saved to.
artifact_path = "wine_artifact_models"

def eval_metrics(actual, pred) -> tuple:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


###################                   Data Preprocessing                  #####################
# Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
wine_path = Path("data/winequality-white.csv")
here_path = Path("getting_started")
data = pd.read_csv(here_path/wine_path, sep=";")

# Split the data into training and test sets. (0.9, 0.1) split.
train, test = train_test_split(data, test_size=0.1)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
}
random_forest = RandomForestRegressor(**params, random_state=42)
random_forest.fit(train_x, train_y)
model_metadata = {
    "model":"RegressionForestRegressor",
    "package":"sklearn",
    "task":"regression"
}
with mlflow.start_run(run_name=run_name) as run:
    # logging parameters
    mlflow.log_params(params)
    predicted_qualities = random_forest.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    mlflow.log_metrics(metrics)
    # Log all files in local_dir as artifacts
    mlflow.log_artifacts(local_dir = "./getting_started/outputs", artifact_path = artifact_path)
    # Log the model 
    signature = infer_signature(model_input = test_x, model_output = predicted_qualities,
                                params = params)
    mlflow.sklearn.log_model(sk_model=random_forest, 
                             artifact_path=artifact_path, signature=signature)
model_uri = f"runs:/{run.info.run_id}/RandForest"
mv = mlflow.register_model(model_uri, "sklear_rf", tags = model_metadata)
print(f"Name: {mv.name}")
print(f"Version: {mv.version}")