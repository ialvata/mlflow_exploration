import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from urllib.parse import urlparse
import pandas as pd
import json

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
 # Create and train models.
params = {"n_estimators": 100, "max_depth": 6, "max_features":3}
# creating the model
rf = RandomForestRegressor(**params) # type: ignore
rf.fit(X_train, y_train)
# Build the Evaluation Dataset from the test set
eval_data = pd.DataFrame(X_test, columns = db.feature_names)
eval_data["label"] = y_test
with mlflow.start_run(run_name="diabetes_2") as run:
      result = mlflow.evaluate(
        rf.predict,
        eval_data,
        targets="label",
        model_type="regressor",# for classification -> "classifier"
        evaluators=["default"],
    )
print(f"metrics:\n{result.metrics}")
# we'll get something like:
# {'example_count': 111, 'mean_absolute_error': 50.46662787639848, 
#  'mean_squared_error': 3702.815538146983, 'root_mean_squared_error': 60.850764482847566, 
#  'sum_on_target': 17491.0, 'mean_on_target': 157.57657657657657, 
#  'r2_score': 0.4042874556312859, 'max_error': 155.8545720070091, 
#  'mean_absolute_percentage_error': 0.44688828674652437}

print(f"artifacts:\n{result.artifacts}") # no artifacts...
