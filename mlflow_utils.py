import mlflow
from typing import Any

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags:dict[str,Any]) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
        print(f'Successfully created experiment env! ------ Experiment name is {experiment_name}')
    except:
        print(f'Experiment {experiment_name} already exists.')
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id

def get_mlflow_experiment(experiment_id:str=None, experiment_name: str=None) -> mlflow.entities.Experiment:

    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError('experiment_id나 experiment_name 입력하셈.')
    
    return experiment


