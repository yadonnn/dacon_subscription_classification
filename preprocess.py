import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class getData:
    def __init__(self, train_path, test_path, submission_path=None):
        self.train_path = pd.train_path
        self.test_path = pd.test_path

    def get_csv_data(self, train, test):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    if submission_path:
        submission = pd.read_csv(submission_path)
    else:
        submission = None

    return train, test, submission

def split_data():
    train_test_split(X_train,)
def train_model():
