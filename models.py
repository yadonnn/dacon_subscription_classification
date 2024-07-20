from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
random_state = 42

#추후 grid search 또는 random search 추가 예정
clf_best_params = {
    "LGBM": {'n_estimators': 100, 'learning_rate': 0.1},
    "RandomForest": {'n_estimators': 100},
    "CatBoost": {'iterations': 100, 'learning_rate': 0.1},
    "XGBoost": {'n_estimators': 100, 'learning_rate': 0.1}
}

models = {
    "LGBM": LGBMClassifier(**clf_best_params['LGBM'], random_state=random_state),
    "RandomForest": RandomForestClassifier(**clf_best_params['RandomForest'], random_state=random_state),
    'CatBoost': CatBoostClassifier(**clf_best_params['CatBoost'], random_state=random_state),
    'XGBoost': XGBClassifier(**clf_best_params['XGBoost'], random_state=random_state)
}

def create_models_pipeline(models):
    pipeline = Pipeline(steps=[('models', models)])
    return pipeline
"""
마저 해야됨!!!!!!!!!!!
self.type if문으로 해야 될 것 같은데?
파이프라인 전처리 단계랑 모델링 단계 구분해야 될 듯..
240716
"""
# class EnsembleLearningCV(BaseEstimator):
#     def __init__(self, models, type='voting', cv=5, random_state=42):
#         self.models = models
#         self.cv = cv
#         self.random_state = random_state
#         self.type = {'voting': VotingClassifier(),
#                          'stacking': StackingClassifier(),
#                          'bagging': BaggingClassifier()}
#         self.ensemble = None

#     def decide_type(self):
#         estimators = [(name, model) for name, model in models.items()]
#         try:
#             self.ensemble = self.type[self.type]
#         except:
#             raise ValueError("type must be one of ['voting', 'stacking', 'bagging'].")

#     def fit(self, X, y):
#         return self
    
#     def predict(self, X):
#         return self.predict(X)
    
#     def predict_proba(self, X):
#         return self.predict_proba(X)