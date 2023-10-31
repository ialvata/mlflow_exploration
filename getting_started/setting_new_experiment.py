from mlflow import MlflowClient
from pprint import pprint


client = MlflowClient(tracking_uri="http://127.0.0.1:8080")


# Provide an Experiment description that will appear in the UI
experiment_description = (
    """
    This is the wine quality project. This experiment contains the produce models 
    for wine quality. 
    
    This a regression experiment.
    """
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "wine-quality",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
}

# Create the Experiment, providing a unique name
produce_wines_experiment = client.create_experiment(
    name="Wine_Models", tags=experiment_tags
)

client.set_experiment_tag(experiment_id = produce_wines_experiment,
                          key = "team", value="stores-machine_learning")

