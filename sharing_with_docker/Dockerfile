FROM python:3.10

RUN pip install mlflow numpy scipy pandas scikit-learn
RUN mkdir ./sharing_with_docker && export MLFLOW_EXPERIMENT_NAME="SHARING_WITH_DOCKER" \
MLFLOW_RUN_ID="sharing_with_docker"
COPY sharing_with_docker/wine.py ./sharing_with_docker/wine.py
COPY sharing_with_docker/winequality-white.csv ./sharing_with_docker/winequality-white.csv