import mlflow
from typing import Any
import numpy as np
from functools import wraps
from sklearn.pipeline import Pipeline
def set_or_create_experiment(experiment_name: str) -> str:
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)
    
    return experiment_id

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags:dict[str,Any], func) -> str:
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
        raise ValueError('Input experiment_id or experiment_name!')
    
    return experiment

def log_mlflow_metric(func):
    @wraps(func)
    def wrapper(self, *args, **kargs):
        # mlflow 기록 제목 설정
        if isinstance(self.pipeline, Pipeline):
            run_name = self.pipeline.steps[-1][0]
        else:
            run_name = 'set pipeline'
        
        with mlflow.start_run(run_name=run_name, nested=True):
            results = func(self, *args, **kargs)
            # print(f'cv results: {results}')
            
            # 평균값
            for key, value in results.items():
                mlflow.log_metric(key, np.round(np.mean(value), 4))
                print(f'{key} : {np.round(np.mean(value), 4)} saved in mlflow.')
        return results
    return wrapper

def log_cv_mlflow_metric(func):
    @wraps(func)
    def wrapper(self, *args, **kargs):
        # mlflow 기록 제목 설정
        if isinstance(self.pipeline, Pipeline):
            run_name = self.pipeline.steps[-1][0]
        else:
            run_name = 'set pipeline'

        with mlflow.start_run(run_name='cross-validate'):
            results = func(self, *args, **kargs)
            # print(f'cv results: {results}')
            with mlflow.start_run(run_name=run_name, nested=True):       
                for key, value in results.items():
                    mlflow.log_metric(key, np.round(np.mean(value), 4))
                    print(f'{key} : {np.round(np.mean(value), 4)} saved in mlflow.')
        return results
    return wrapper