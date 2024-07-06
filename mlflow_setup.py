import mlflow 
from mlflow_utils import create_mlflow_experiment

if __name__ == '__main__':
    experiment_id = create_mlflow_experiment(experiment_name='subscription_classification',
                                             artifact_location='subscription_classification_artifacts',
                                             tags={'env':'dev', 'version':'1.0.0'})
    print(f'experiment_id : {experiment_id}')