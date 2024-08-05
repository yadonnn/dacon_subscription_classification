from dataloader import get_path, read_df
from preprocess import *
from pipeline import create_pipeline
from crossvalidate import CrossValidate
import mlflow
from mlflow_utils import set_or_create_experiment
from models import create_models_pipeline, models
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    #mlflow experiment 환경설정
    experiment_name = 'subscription_classification'
    set_or_create_experiment(experiment_name=experiment_name)

    train, test = read_df()
    X, y = set_train_feature_target(train, user_id='user_id', target='target')
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = split_data(X, y)

    

    preprocessor = create_preprocessor(X)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    #smote 추가
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)

    # X_train = np.load('test/X_train_transformed.npy')
    # X_test = np.load('test/X_test_transformed.npy')
    # pipeline = Pipeline(steps=[
    #     (create_pipeline(preprocessor=preprocessor)),
    #     (create_models_pipeline(models=models))
    #     ]
    #     )
    print(X_train_t.shape, X_test_t.shape)
    with mlflow.start_run(run_name='cross-validate'):
        for name, model in models.items():
            pipeline = Pipeline(steps=[(name, model)])
            cv = CrossValidate(pipeline, X_train_res, y_train_res)
            cv.classifier()

