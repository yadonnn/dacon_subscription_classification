import os
import pandas as pd

def get_path():
    Path = os.getcwd() + '\\data\\'
    return Path

def read_df():
    train = pd.read_csv(get_path() + 'train.csv')
    test = pd.read_csv(get_path() + 'test.csv')
    return train, test
