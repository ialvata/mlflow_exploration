"""
In this script we'll perform model validation (and model comparison) with the use of thresholds
"""
import pandas as pd

import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

import mlflow
from mlflow import MlflowClient
from mlflow.models import MetricThreshold, infer_signature, make_metric

tracking_uri="http://127.0.0.1:8080"
client = MlflowClient(tracking_uri=tracking_uri)
# Provide an Experiment description that will appear in the UI
experiment_description = (
    """
    This is model validation and comparison experiment. 
    We will create 3 models: random uniform classifier (dummy), a default random forest 
    classifier (default_model), and a costumized random forest classifier ()
    
    This a classification experiment.
    """
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "model_validation",
    "baseline_model": "dummy_uniform",
    "candidate_model": "random_forest_scikit",
    "task":"classification",
    "mlflow.note.content": experiment_description,
}

print("Creating and Setting Experiment")
# Create the Experiment, providing a unique name
md_val_experiment = client.search_experiments(
    filter_string="name = 'Model_val::rf_vs_Dummy'" 
    # pay attention to `` is different from ''
)
if len(md_val_experiment)==0:
    md_val_experiment = client.create_experiment(
        name="Model_val::rf_vs_Dummy", tags=experiment_tags
    )
# Use the fluent API to set the tracking uri and the active experiment
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Model_val::rf_vs_Dummy")

print("Loading Data")
# load UCI Adult Data Set; segment it into training and test sets
X, y = shap.datasets.adult()
# X = pd.DataFrame(X,dtype=float)
# y = pd.DataFrame(y, columns =["label"],dtype=float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("Training Default Model")
# train a default RandomForestClassifier scikit model
default_model = RandomForestClassifier()
default_model.fit(X_train, y_train)
default_model_signature = infer_signature(X_train, default_model.predict(X_train))

print("Training Costumized Model")
# train a costumized RandomForestClassifier scikit model
params = {"n_estimators": 120, "max_depth": 6, "max_features":"log2"}
costume_model = RandomForestClassifier(**params)
costume_model.fit(X_train, y_train)
costume_model_signature = infer_signature(X_train, costume_model.predict(X_train))

print("Training Dummy Model")
# train a baseline dummy model
baseline_model = DummyClassifier(strategy="uniform").fit(X_train, y_train)
# "uniform": generates predictions uniformly at random from the list of unique classes 
# observed in y, i.e. each class has equal probability.
baseline_signature = infer_signature(X_train, baseline_model.predict(X_train))

# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["label"] = y_test

print("Evaluation Criteria")
# Define a custom metric to evaluate against
def double_positive(_eval_df, builtin_metrics):
    return builtin_metrics["true_positives"] * 2
# Define criteria for model to be validated against
thresholds = {
    # Specify metric value threshold
    "precision_score": MetricThreshold(
        threshold=0.7, greater_is_better=True
    ),  # precision should be >=0.7
    # Specify model comparison thresholds
    "recall_score": MetricThreshold(
        min_absolute_change=0.1,  
        # recall should be at least 0.1 greater than baseline model recall
        min_relative_change=0.1,  
        # recall should be at least 10 percent greater than baseline model recall
        greater_is_better=True,
    ),
    # Specify both metric value and model comparison thresholds
    "accuracy_score": MetricThreshold(
        threshold=0.8,  # accuracy should be >=0.8
        min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=True,
    ),
    # Specify threshold for custom metric
    "double_positive": MetricThreshold(
        threshold=1e5, greater_is_better=False  # double_positive should be <=1e5
    ),
}

#########         Currently mlflow.evaluate has a bug for sckit's RFClassifier     ############
#########         I've already opened an issue on github                           ############


run_name = "rf_default_vs_Dummy"
print(f"Starting Run -> {run_name}")
with mlflow.start_run(run_name=run_name) as run:
    print("Creating Model URIs")
    # Note: in most model validation use-cases the baseline model should instead be a previously
    # trained model (such as the current production model), specified directly in the
    # mlflow.evaluate() call via model URI.
    baseline_model_uri = mlflow.sklearn.log_model(
        baseline_model, "baseline_model", signature=baseline_signature
    ).model_uri
    # creating uri for default candidate model. 
    # we could also simply pass its .predict method in the evaluation below...
    default_model_uri = mlflow.sklearn.log_model(
        default_model, "default_model", signature=default_model_signature
    ).model_uri
    # creating uri for costume candidate model.
    costume_model_uri = mlflow.sklearn.log_model(
        costume_model, "costume_model", signature=costume_model_signature
    ).model_uri

    print("Evaluating Default vs Dummy") 
    mlflow.evaluate(
        default_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
        validation_thresholds=thresholds,
        extra_metrics=[
            make_metric(
                eval_fn=double_positive,
                greater_is_better=False,
            )
        ],
        baseline_model=baseline_model_uri,
        # set to env_manager to "virtualenv" or "conda" to score the candidate and baseline models
        # in isolated Python environments where their dependencies are restored.
        env_manager="local", # "local"
    )
    print("Evaluating Costume vs Dummy") 
    mlflow.evaluate(
        costume_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
        validation_thresholds=thresholds,
        extra_metrics=[
            make_metric(
                eval_fn=double_positive,
                greater_is_better=False,
            )
        ],
        baseline_model=baseline_model_uri,
        # set to env_manager to "virtualenv" or "conda" to score the candidate and baseline models
        # in isolated Python environments where their dependencies are restored.
        env_manager="local",
    )
    # If you would like to catch model validation failures, you can add try except clauses around
    # the mlflow.evaluate() call and catch the ModelValidationFailedException, imported at the top
    # of this file.