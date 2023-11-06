# mlflow_exploration

## About

This repo contains various notebooks and scripts where I experiment with MLflow regarding experiment tracking, model evaluation, and registry, while using a docker container emulating MLflow server. 

The following is a short description of what can be found in each folder.

### Getting started
A quickstart tutorial, using MLflow 2.8.0, where learn how to start a MLflow Tracking Server and the MLflow UI Server locally. 
We create a MLflow experiment with a unique name and identifying tags. 
We connect to the Tracking Server with the MLflow Client and search for experiments, while using relevant identifying tag values.
Then we train a model using the wine dataset and log the trained model, metrics, parameters, and artifacts.

### Quickstart
Another quickstart tutorial, based on an older version of the MLflow docs, similar to the previous folder, where we train a model using the wine dataset and log the trained model, metrics, parameters, and artifacts.

### sharing_with_docker
This folder contains an MLflow project that trains a RandomForestRegressor scikit model on the UC Irvine Wine Quality Dataset. The project is an example of how we can share ML code and improve reproducibility. It uses a Docker image to capture the dependencies needed to run training code. Running a project in a Docker environment allows for capturing non-Python dependencies, as opposed to a conda environment.

## Useful commands

Launching MLflow Tracking Server:
- mlflow server --host 127.0.0.1 --port 8080

With the Tracking Server operational, it’s time to start the MLflow UI. Launch it from a new command prompt. As with the Tracking Server, ensure this window remains open:
- mlflow ui --host 127.0.0.1 --port 8090

