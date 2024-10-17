#Create a cotainer
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


#set the seed
np.random.seed(123)

#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
f_rate = 1 #rate of data sampling.
data_file_csv = 'data/env.sel3.scaled.csv' 
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')

# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0,3.5,3.7, 4.0]),          
          'kernel':['rbf'] }

#Create the GridSearchCV object
grid = GridSearchCV(SVC(), params_grid, verbose=1)

#get time
t_start = time.time()


#Fit the data with the best possible parameters
grid_clf = grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid.best_params_}')

#perform grid prediction on test set
grid_predictions = grid.predict(X_test) 

# print classification report 
print(classification_report(y_test, grid_predictions))

#print scro a comparison
score = accuracy_score(grid_predictions, y_test)
print(f'Accuracy Score on data: {score}')

# GridSearchCV 
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Fitting 5 folds for each of 234 candidates, totalling 1170 fits
# Best paramenter: {'C': 8.0, 'gamma': 3.0, 'kernel': 'rbf'}
# /home/francesco/git/QK/.venv/lib/python3.12/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but SVC was fitted without feature names
#   warnings.warn(
#               precision    recall  f1-score   support

#           -1       0.94      0.81      0.87       410
#            1       0.79      0.93      0.86       307

#     accuracy                           0.86       717
#    macro avg       0.87      0.87      0.86       717
# weighted avg       0.88      0.86      0.87       717
# Accuracy Score on data: 0.8647140864714087

#WRONG
# Shape of dataset: (286, 7)
# Training shape dataset (286, 6)
# Label for traing (286,)
# Best paramenter: {'C': 8.0, 'gamma': 0.25, 'kernel': 'rbf'}

