import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

mapping_difficulty = {'Low' : 0
					  ,'Medium' : 1
					  ,'High' : 2
					  }
mapping_sub_type = {'Basic' : 0
					,'Premium' : 1
					}
mapping_payment = {0 : [0, 0, 0]
                    ,1 : [0, 0, 1]
                    ,2 : [0, 1, 0]
                    ,3 : [0, 1, 1]
                    ,4 : [1, 0, 0]
                    ,5 : [1, 0, 1]
                    ,6 : [1, 1, 0]
                    ,7 : [1, 1, 1]
                    }

def set_train_feature_target(train, user_id=str, target=str):
    X = train.drop([user_id, target], axis=1)
    y = train[target]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

class MappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.map(self.mapping).values.reshape(-1, 1)

def payment_pattern_transformer(X):
    return np.array([mapping_payment[x] for x in X])

def create_mapping_transformer(name:str, scaler) -> Pipeline:
    return Pipeline(
        steps=[
            (name, scaler)
        ]
    )

def create_preprocessor(X, numeric_scaler_type='standard'):
    # 컬럼 선택
    numeric_features = X.select_dtypes([np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns
    payment_col = 'payment_pattern'
    difficulty_col = 'preferred_difficulty_level'
    sub_type_col = 'subscription_type'
    # 컬럼별 파이프라인
    numeric_scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'norm': Normalizer()
        }
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

    # 커스텀 매핑
    payment_transformer = FunctionTransformer(payment_pattern_transformer, validate=False)
    difficulty_transformer = create_mapping_transformer('difficulty', MappingTransformer(mapping=mapping_difficulty))
    sub_type_transformer = create_mapping_transformer('sub_type', MappingTransformer(mapping=mapping_sub_type))

    # 파이프라인 통합
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('payment', payment_transformer, payment_col),
            ('difficulty', difficulty_transformer, difficulty_col),
            ('sub_type', sub_type_transformer, sub_type_col)

        ],
        remainder='passthrough'
    )
    return preprocessor

# class ordinal_scaler(BaseEstimator, TransformerMixin):
#     def __init__(self, mapping):
#         self.mapping = mapping

#     def fit(self, X, y=None):
#         return self 
    
#     def transform(self, X, y=None):
#         return X.map(self.mapping).values.reshape(-1, 1)