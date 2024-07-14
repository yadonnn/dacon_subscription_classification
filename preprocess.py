import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder

def set_train_feature_target(train, user_id=str, target=str):
    X = train.drop([user_id, target], axis=1)
    y = train[target]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_preprocessor(X, numeric_scaler_type='standard'):
    numeric_scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'norm': Normalizer()
        }
    
    numeric_features = X.select_dtypes([np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    numeric_transformer = Pipeline(
        steps=[
        ('NumericScaler', numeric_scalers[numeric_scaler_type])
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
        ('OneHot', OneHotEncoder(sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
