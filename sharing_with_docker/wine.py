import argparse
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", default=100)
    parser.add_argument("--max_depth", default=10)
    parser.add_argument("--max_features", default=6)
    args = parser.parse_args()

    
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = Path("winequality-white.csv")
    here_path = Path("sharing_with_docker")
    data = pd.read_csv(here_path/wine_path, sep=";")

    # Split the data into training and test sets. (0.9, 0.1) split.
    train, test = train_test_split(data, test_size=0.1)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    params = {"n_estimators": int(args.n_estimators), "max_depth": int(args.max_depth), 
              "max_features":int(args.max_features)}
    # mlflow.set_experiment(experiment_name="SHARING_WITH_DOCKER_EXPERIMENT")
    with mlflow.start_run() as run:#run_name="sharing_with_docker_run") as run:
        # logging parameters
        mlflow.log_params(params)
        random_forest = RandomForestRegressor(**params, random_state=42)
        random_forest.fit(train_x, train_y)
        predicted_qualities = random_forest.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
    # there are different ways to log a model 
    # see https://mlflow.org/docs/latest/model-registry.html#api-workflow
    # see also quickstart folder
    model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv = mlflow.register_model(model_uri, "RandomForestRegression")