import sys
import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

from qiskit_machine_learning.kernels import FidelityStatevectorKernel 
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_algorithms.utils import algorithm_globals

import numpy as np
from datetime import datetime


#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)
sys.path.append('./jobs/' + current_wd)

from pqk.Circuits import Circuits

#set the seed
seed=123              # the seed
np.random.seed(seed)
algorithm_globals.random_seed = seed

# dictionary of available encodings
encoding_dict = {
    'xyz': Circuits.xyz_encoded, 
    'zz': Circuits.zzfeaturemap, 
    'x': Circuits.x_encoded,
    'iqp': Circuits.IQP_HuangE2,
    'trotter': Circuits.Trotter_HuangE3
    }   

# select parameters for the run
encoding_key = 'trotter' # choosing the encoding (see dict)
full_ent = False  #True or False for full entanglement in the feature map
nfolds = 10 #set number of folds in CV
f_rate = 1 #rate of data sampling fot testing pourpose
nj = 1     # number of processors on the host machine. CAREFUL: it uses ALL PROCESSORS if n_jopbs = -1
eval_score = 'f1'  #evaluation score for the grid search

#load dataset with panda
#data are scaled outside the notebook

if encoding_key == 'uniform':
    data_file_csv = 'data/env.sel3.2pi_minmax.csv'
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)  
else:
    data_file_csv = 'data/env.sel3.sk_sc.csv'
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X.to_numpy()
y_train_np = Y.to_numpy()

params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]}

# params_grid = {
#     'C': [2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]    
# }

#take a feature map and setting QSVC
NUM_QBIT = 6 #commonly, this is the number of the features
fm = encoding_dict[encoding_key](n_wire=NUM_QBIT, full_ent=full_ent)   
#fm = ZZFeatureMap(feature_dimension=NUM_QBIT)
q_kernel = FidelityStatevectorKernel(feature_map=fm)
svm_quantum = QSVC(quantum_kernel=q_kernel)

print(f'Feature map: {fm.name}\n')



#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
grid = GridSearchCV(svm_quantum, params_grid, verbose=1, cv=nfolds,n_jobs=nj, scoring=eval_score)

print('***INFO RUN***')
print(f'Name of Feature Map: {fm.name}')
print(fm.draw())
print(f'N job param = {nj}')
print(f'GridSearch Dict: {params_grid}')
print(f'Source file: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Used score: {grid.scoring}')


#get time
t_start = time.time()


#Fit the data with the best possible parameters
grid_clf = grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

num_support_vectors = grid.best_estimator_.n_support_

#Print the best estimator with it's parameters
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

# defining a unique label for the simulation 
id_string = f'_QSVM_{encoding_key}_ent{full_ent}_{nfolds}folds_seed{seed}_frate{f_rate}_{eval_score}_fm_{fm.name}'

# INFORMATION SAVED IN THE 'run*.txt' OUTPUT FILES
with open(f'run' + id_string + '.txt', "w+") as file:    
    file.write(final_msg + '\n\n')
    file.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    file.write(f'{t_training-t_start:.1f} seconds elapsed.\n')
    file.write(f'Future map: {fm.name}\n')
    file.write(f'Entanglement: {full_ent}\n')
    file.write(f'Used score: {grid.scoring}\n')
    file.write(f'Best parameter: {grid.best_params_}\n') 
    file.write(f'Number of SV of the best model: {num_support_vectors.sum()}. ({num_support_vectors})\n')
    file.write(f'N job param = {nj}\n')
    file.write(f'GridSearch Dict: {params_grid}\n')
    #check the shape of test and training dataset
    file.write(f'Source file: {data_file_csv}\n')
    file.write(f'Shape of dataset: {env.shape}\n')
    file.write(f'Shape of training dataset {X_train_np.shape}\n')
    file.write(f'Shape of training labels {y_train_np.shape}\n')
    file.write(f'Seed: {seed}\n')
    file.write(f'Fitting {nfolds} folds for each of {len(ParameterGrid(grid.param_grid))} candidates.\n')
    for i in range(nfolds):
        file.write(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}\n") 
