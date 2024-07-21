import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from dataloader import get_path, read_df
from preprocess import *
from pipeline import create_pipeline
from crossvalidate import CrossValidate
import mlflow
from mlflow_utils import set_or_create_experiment
from models import create_models_pipeline, models
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    dir = '../data/'
    #mlflow experiment 환경설정
    experiment_name = 'subscription_classification'
    set_or_create_experiment(experiment_name=experiment_name)

    train = pd.read_csv(dir+'train.csv')
    test = pd.read_csv(dir+'test.csv')
    X, y = set_train_feature_target(train, user_id='user_id', target='target')
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = create_preprocessor(X)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(X_train_t.shape, X_test_t.shape)
    print(X_train_t)
    with open('test.npy', 'wb') as f:
        np.save(f, X_train_t)