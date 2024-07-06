import mlflow 
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment

create_mlflow_experiment(experiment_name='test', artifact_location='test_artifacts', tags={'env': 'dev', 'version': '1.0'})
