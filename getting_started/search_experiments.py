from mlflow import MlflowClient
from pprint import pprint



client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

all_experiments = client.search_experiments()

print(all_experiments)

produce_wines_exp_search = [
    {"name": experiment.name,"id": experiment.experiment_id, 
     "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Wine_Models"
][0]

pprint(produce_wines_exp_search) # {'lifecycle_stage': 'active', 'name': 'wine_Models'}

produce_wines_exp_search_2 = [
    experiment
    for experiment in all_experiments
    if experiment.name == "Wine_Models"
][0]
# Use search_experiments() to search on the project_name tag key
pprint(produce_wines_exp_search_2)

wines_experiment = client.search_experiments(
    filter_string="tags.`project_name` = 'wine-quality'" 
    # pay attention to `` is different from ''
)

pprint(vars(wines_experiment[0]))

wines_experiment = client.search_experiments(
    filter_string="tags.`team` = 'stores-machine_learning'" 
    # pay attention to `` is different from ''
)
assert len(wines_experiment) == 1
pprint(vars(wines_experiment[0]))

wines_experiment = client.search_experiments(
    filter_string="tags.`team` = 'stores-ml'" 
    # pay attention to `` is different from ''
)
assert len(wines_experiment) == 0

print("Olá")