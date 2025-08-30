#Create a cotainer
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ParameterGrid
import numpy as np
from datetime import datetime
import sys
import os

current_wd = os.getcwd()
sys.path.append(current_wd)
sys.path.append('./jobs/' + current_wd)

def get_origin_data(f_rate, source_file) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    #load dataset with panda    
    data_file_csv = source_file
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)      

    #DEFINE design matrix
    Y = env['occupancy']
    X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

    #WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
    X_train_np = X.to_numpy()
    y_train_np = Y.to_numpy()

    return (X_train_np,y_train_np,env)

def get_qencoded_data(f_rate, source_file) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    #load dataset with panda
    data_file_csv = source_file
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)      

    #DEFINE design matrix
    Y = env['label']
    X = env[env.columns.difference(['label'])]
    
    #WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
    X_train_np = X.to_numpy()
    y_train_np = Y.to_numpy()

    return (X_train_np,y_train_np,env)



#set the seed
seed=123              # the seed
np.random.seed(seed)

nfolds = 10 #set number of folds in CV
f_rate = 1 #rate of data sampling fot testing pourpose
nj = 1     # number of processors on the host machine. CAREFUL: it uses ALL PROCESSORS if n_jopbs = -1
eval_score = 'f1'  #evaluation score for the grid search

#source_file = 'data/env.sel3.sk_sc.csv'
#source_file = 'qfm/fm/qencoding/QC_X_OBS_M1_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_M1_ENT_TRUE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_M1_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_ZZ_OBS_M1.csv'
#source_file = 'qfm/fm/qencoding/QC_IQP_OBS_M1.csv'
#source_file = 'qfm/fm/qencoding/QC_TROTTER_OBS_M1.csv'
#source_file = 'qfm/fm/qencoding/QC_X_OBS_M2_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_M2_ENT_TRUE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_M2_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_ZZ_OBS_M2.csv'
#source_file = 'qfm/fm/qencoding/QC_IQP_OBS_M2.csv'
#source_file = 'qfm/fm/qencoding/QC_TROTTER_OBS_M2.csv'
#source_file = 'qfm/fm/qencoding/QC_X_OBS_MM_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_MM_ENT_TRUE.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_MM_ENT_FALSE.csv'
#source_file = 'qfm/fm/qencoding/QC_ZZ_OBS_MM.csv'
#source_file = 'qfm/fm/qencoding/QC_IQP_OBS_MM.csv'
source_file = 'qfm/fm/qencoding/QC_TROTTER_OBS_MM.csv'
#source_file = 'qfm/fm/qencoding/QC_3D_OBS_M2_L2.csv'



#get origin data
#X_train_np,y_train_np,env = get_origin_data(f_rate, source_file)
X_train_np,y_train_np,env = get_qencoded_data(f_rate, source_file)

# defining a unique label for the simulation 
id_string = f'_SVM_{X_train_np.shape[1]}feats_{nfolds}folds_seed{seed}_frate{f_rate}'


#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'File with data: {source_file}')


# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {
    'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    'gamma': [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
    'kernel':['rbf'] 
    }

#Create the GridSearchCV object
grid = GridSearchCV(SVC(), params_grid, verbose=1, cv=nfolds, n_jobs=nj, scoring=eval_score)


#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid_clf = grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

num_support_vectors = grid.best_estimator_.n_support_

#Print the best estimator with it's parameters
print(f'Used score: {grid.scoring}')
print(f'Best paramenter: {grid.best_params_}')
print(f'Time training: {t_training - t_start} seconds')
print(f'Best score {grid.best_score_}')
print(f'Results: {grid.cv_results_.keys()}')
print(f'Number of SV of the best model: {num_support_vectors.sum()}. ({num_support_vectors})')

# taking the largest average accuracy of the grid search and the corresponding standard dev.
cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
cv_std = grid.cv_results_['std_test_score'][grid.best_index_]

results = grid.cv_results_
for i in range(nfolds):
    print(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}")

print(f'\nBest score (Average): {cv_mean:.6f}')
print(f'Best score (Standard Deviation): {cv_std:.6f}')

print(f'{t_training-t_start} seconds elapsed.')

# the confidence interval is given by:   mean +/- 2 * stdev / sqrt(N)
final_msg = f'Score (95% confidence) = {cv_mean:.6f} +/- {2*cv_std/np.sqrt(nfolds):.6f} == [{cv_mean - 2*cv_std/np.sqrt(nfolds):.6f}, {cv_mean + 2*cv_std/np.sqrt(nfolds):.6f}]'
print(final_msg)

# INFORMATION SAVED IN THE 'accuracy*.txt' OUTPUT FILES
with open(f'accuracy' + id_string + '.txt', "w+") as file:    
    file.write(final_msg + '\n\n')
    file.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    file.write(f'{t_training-t_start:.1f} seconds elapsed.\n')
    file.write(f'Used score: {grid.scoring}\n')
    file.write(f'Best parameter: {grid.best_params_}\n') 
    file.write(f'Number of SV of the best model: {num_support_vectors.sum()}. ({num_support_vectors})\n')
    file.write(f'N job param = {nj}\n')
    file.write(f'GridSearch Dict: {params_grid}\n')
    #check the shape of test and training dataset
    file.write(f'Source file: {source_file}\n')
    file.write(f'Shape of dataset: {env.shape}\n')
    file.write(f'Shape of training dataset {X_train_np.shape}\n')
    file.write(f'Shape of training labels {y_train_np.shape}\n')
    file.write(f'Seed: {seed}\n')
    file.write(f'Fitting {nfolds} folds for each of {len(ParameterGrid(grid.param_grid))} candidates.\n')
    for i in range(nfolds):
        file.write(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}\n")