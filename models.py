import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier()
n_folds = 3
def get_stacking_data(model, X_train, y_train, X_test, n_folds = 5):

    skf = StratifiedKFold(n_splits=n_folds)
    train_fold_predict = np.zeros((X_train.shape[0], 1))
    test_predict = np.zeros((X_test.shape[0], n_folds))
    print(train_fold_predict.shape, test_predict.shape)
    for cnt, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        X_val = X_train[val_idx]

        model.fit(X_train, y_train)

        train_fold_predict[val_idx, :] = model.predict(X_val)

        test_predict[val_idx, :] = model.predict(X_test)
    
    print(test_predict)
    print('*'*50)
    print(train_fold_predict)

    return train_fold_predict, test_predict

models = {'svm': SVC()
          ,'lr': LogisticRegression()
          ,'rf': RandomForestClassifier()}

for name, model in models.items():
    train_preds, test_preds = get_stacking_data(model, X_train, y_train, X_test, n_folds=n_folds)
    break
# get_stacking_data(model=model, X_train=X_train, y_train=y_train, X_test=X_test, n_folds=n_folds)

        
