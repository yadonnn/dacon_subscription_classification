from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate
from mlflow_utils import log_mlflow_metric

class CrossValidate:
    def __init__(self, pipeline, X_train, y_train):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
    
    @log_mlflow_metric
    def classifier(self):
        scoring = {
            'F1_score' : make_scorer(f1_score),
            'Accuracy' : 'accuracy',
            'Precision' : make_scorer(precision_score),
            'Recall' : make_scorer(recall_score)
        }
        results = cross_validate(self.pipeline, self.X_train, self.y_train, scoring=scoring, cv=5)
        print(results)

        return results
        
    def regressor():
        scoring = {
            
        }
        pass