from dataloader import get_path, read_df
from preprocess import *
from pipeline import create_pipeline
from crossvalidate import CrossValidate
import mlflow
from mlflow_utils import set_or_create_experiment
from models import create_models_pipeline, models
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    #mlflow experiment 환경설정
    experiment_name = 'te'
    set_or_create_experiment(experiment_name=experiment_name)

    train, test = read_df()
    X, y = set_train_feature_target(train, user_id='user_id', target='target')
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = create_preprocessor(X)
    print(preprocessor)
    pipeline = Pipeline(steps=[
        (create_pipeline(preprocessor=preprocessor)),
        (create_models_pipeline(models=models))
        ]
        )

    cv = CrossValidate(pipeline, X_train, y_train)
    cv.classifier()

