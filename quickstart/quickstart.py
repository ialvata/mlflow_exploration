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

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

with mlflow.start_run(run_name="diabetes_2"):
    # Create and train models.
    params = {"n_estimators": 100, "max_depth": 6, "max_features":3}
    # Log a dictionary of parameters
    log_params(params)
    # creating the model
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    # Log a metric; metrics can be updated throughout the run
    log_metric("mean absolute error", mean_absolute_error(y_test,predictions))
    log_metric("mean squared error", mean_squared_error(y_test,predictions))
    log_metric("R2 - coefficient of determination", r2_score(y_test,predictions))
    # Plot outputs
    plt.scatter(X_test[:,0], y_test, color="black")
    plt.scatter(X_test[:,0], predictions, color="blue")
    plt.xticks(())
    plt.yticks(())
    plt.savefig("./quickstart/outputs/plot_diabetes")
    # Log an artifact (output file)
    log_artifacts("./quickstart/outputs")
    # Log the model 
    signature = infer_signature(model_input = X_test, model_output = predictions,
                                params = params)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            rf, "model", registered_model_name="RandomForestRegression", signature=signature
        )
    else:
        mlflow.sklearn.log_model(rf, "model", signature=signature)