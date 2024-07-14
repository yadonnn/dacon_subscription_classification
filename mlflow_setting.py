from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
    experiment_info = {}
    experiment_info['experiment_name'] = input(str)
    experiment_info['artifact_location'] = experiment_info['experiment_name'] + "_artifacts"
    experiment_info['tags'] = {'env': 'dev',
                            'version': '0.0.1'}

get_mlflow_experiment(experiment_name=experiment_info['experiment_name'])