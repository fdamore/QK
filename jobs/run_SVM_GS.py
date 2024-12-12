#Create a cotainer
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datetime import datetime
import sys
import os

current_wd = os.getcwd()
sys.path.append(current_wd)
sys.path.append('./jobs/' + current_wd)

#set the seed
seed=123              # the seed
np.random.seed(seed)

nfolds = 10 #set number of folds in CV
f_rate = .05 #rate of data sampling fot testing pourpose
nj = 1     # number of processors on the host machine. CAREFUL: it uses ALL PROCESSORS if n_jopbs = -1
number_of_feature_copies = 1    # this is the number of times we stack the same identical feats (Annalisa's suggestion)


#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
#data_file_csv = 'data/env.sel3.scaled.csv' 
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = np.tile(X.to_numpy(), (1,number_of_feature_copies))  # to implement Annalisa's test
y_train_np = Y.to_numpy()

# defining a unique label for the simulation 
id_string = f'_SVM_{X_train_np.shape[1]}feats_{nfolds}folds_seed{seed}_frate{f_rate}'


#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {
    'C': 2.**np.arange(1,12,2),
    'gamma': 10**np.arange(-7,0.,2),
    'kernel':['rbf'] 
    }

#Create the GridSearchCV object
grid = GridSearchCV(SVC(), params_grid, verbose=1, cv=nfolds)

#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid_clf = grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid.best_params_}')
print(f'Time training: {t_training - t_start} seconds')
print(f'Best score {grid.best_score_}')
print(f'Results: {grid.cv_results_.keys()}')

# taking the largest average accuracy of the grid search and the corresponding standard dev.
cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
cv_std = grid.cv_results_['std_test_score'][grid.best_index_]

results = grid.cv_results_
for i in range(nfolds):
    print(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}")

print(f'\nAverage accuracy, best score: {cv_mean:.6f}')
print(f'Standard deviation, best score: {cv_std:.6f}')

print(f'{t_training-t_start} seconds elapsed.')

# the confidence interval is given by:   mean +/- 2 * stdev / sqrt(N)
final_msg = f'Accuracy (95% confidence) = {cv_mean:.6f} +/- {2*cv_std/np.sqrt(nfolds):.6f} == [{cv_mean - 2*cv_std/np.sqrt(nfolds):.6f}, {cv_mean + 2*cv_std/np.sqrt(nfolds):.6f}]'
print(final_msg)

# INFORMATION SAVED IN THE 'accuracy*.txt' OUTPUT FILES
with open(f'jobs/scores/accuracy' + id_string + '.txt', "w") as file:
    file.write(final_msg + '\n\n')
    file.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    file.write(f'{t_training-t_start:.1f} seconds elapsed.\n')
    file.write(f'Best parameter: {grid.best_params_}\n')
    file.write(f'N job param = {nj}\n')
    file.write(f'GridSearch Dict: {params_grid}\n')
    #check the shape of test and training dataset
    file.write(f'Source file: {data_file_csv}\n')
    file.write(f'Shape of dataset: {env.shape}\n')
    file.write(f'Shape of training dataset {X_train_np.shape}\n')
    file.write(f'Shape of training labels {y_train_np.shape}\n')
    file.write(f'Seed: {seed}\n')
    file.write(f'Fitting {nfolds} folds for each of {len(ParameterGrid(grid.param_grid))} candidates, totalling 240 fits\n')
    for i in range(nfolds):
        file.write(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}\n")


#Using a different scale method (standard sklearn)
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Fitting 5 folds for each of 270 candidates, totalling 1350 fits
# Best paramenter: {'C': 16.0, 'gamma': 1.75, 'kernel': 'rbf'}
# /home/francesco/git/QK/.venv/lib/python3.12/site-packages/sklearn/base.py:486: UserWarning: X has feature names, but SVC was fitted without feature names
#   warnings.warn(
#               precision    recall  f1-score   support

#           -1       0.93      0.83      0.88       410
#            1       0.80      0.92      0.86       307

#     accuracy                           0.87       717
#    macro avg       0.87      0.88      0.87       717
# weighted avg       0.88      0.87      0.87       717

# Accuracy Score on data: 0.8688981868898187

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

