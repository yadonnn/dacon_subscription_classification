import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
from mlflow_utils import get_mlflow_experiment

mlflow.set_experiment(experiment_name = 'subscription_classification')
dir = 'data/'

train = pd.read_csv(dir+'train.csv')
test = pd.read_csv(dir+'test.csv')
submission = pd.read_csv(dir+'sample_submission.csv')

train.head()

from sklearn.model_selection import train_test_split

X = train.drop(['user_id', 'target'], axis=1)
y = train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class num_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ss = StandardScaler()

    def fit(self, X, y=None):
        self.ss.fit(X)
        return self 
    
    def transform(self, X, y=None):
        return self.ss.transform(X)

class label_scaler(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        return X.map(self.mapping).values.reshape(-1, 1)
    
class ohe_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        self.ohe.fit(X.values.reshape(-1, 1))
        return self 
    
    def transform(self, X, y=None):
        return self.ohe.transform(X.values.reshape(-1, 1))
    
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
# 수치형에 왜 실수만 넣었냐?
# 요일 수 스케일링 해야하나? 일단 몰라서 passthrough
num_cols = X_train.select_dtypes('float').columns
label_cols = X_train.select_dtypes(exclude=[np.number]).columns[0]
ohe_cols = X_train.select_dtypes(exclude=[np.number]).columns[1]

mapping_difficulty = {'Low' : 0
					  ,'Medium' : 1
					  ,'High' : 2
					  }

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_scaler(), num_cols),
        ('label', label_scaler(mapping=mapping_difficulty), label_cols),
        ('ohe', ohe_scaler(), ohe_cols)
    ],
    remainder='passthrough'
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier())
])
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate
# 교차검증 
scoring = {
    'f1' : make_scorer(f1_score),
    'accuracy' : 'accuracy'
}

with mlflow.start_run(run_name = 'init rf'):
    results = cross_validate(pipe, X_train, y_train, scoring=scoring, cv=5)
    print(f'cv results: {results}')
    for key, value in results.items():
        mlflow.log_metric(key, np.round(np.mean(value), 4))
    
    mlflow.log_param('n_estimators', pipe['rf'].get_params()['n_estimators'])
    mlflow.log_param('max_depth', pipe['rf'].get_params()['max_depth'])
