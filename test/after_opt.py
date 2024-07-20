import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time, datetime 
import mlflow
start_time = time.time()
X_train_transformed = np.load('test/X_train_transformed.npy')
X_test_transformed = np.load('test/X_test_transformed.npy')
end_time = time.time()
execution_time = np.round(end_time - start_time, 4)

with mlflow.start_run(run_name='loadtime',
                      experiment_id='439615873627327450'):
    mlflow.log_metric('execution time', execution_time)
