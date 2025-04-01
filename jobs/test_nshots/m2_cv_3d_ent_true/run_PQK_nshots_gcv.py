import sys
import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from sklearn.svm import SVC

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

print(current_wd)

from pqk.Circuits import Circuits
from pqk.QEncoding import QEncoding

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123


#quantum stuff: observable, circuit to encoding and number type of function used to measure that uses shots
my_obs = ['XXIIII', 'IXXIII','IIXXII', 'IIIXXI','IIIIXI','XIIIIX',
          'YYIIII', 'IYYIII','IIYYII', 'IIIYYI','IIIIYY','YIIIIY',
          'ZZIIII', 'IZZIII','IIZZII', 'IIIZZI','IIIIZZ','ZIIIIZ']
q_c = Circuits.xyz_encoded(n_wire=6, full_ent = True)


source_file = 'data/env.sel3.sk_sc.csv'
f_rate = 1 #rate of data sampling fot testing pourpose
env = pd.read_csv(source_file).sample(frac=f_rate, random_state=123)  
#env = pd.read_csv(source_file)

#DEFINE design matrix
Y_env = env['occupancy']
X_env = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]
X_np = X_env.to_numpy()
y_np = Y_env.to_numpy()

#Best parameter: {'C': 2, 'gamma': 4.0, 'kernel': 'rbf'}
#define hyperparamenter PQK_M2_ENT_TRUE_18obs
# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {
    'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    'gamma': [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
    'kernel':['rbf'] 
    }
nfolds = 10 #set number of folds in CV
n_shot_ = 20 #number of shots

#print some info about runs and simulation
print('Shape of data to encode:')
print(f'Data: {X_np.shape}')
print(f'Label: {y_np.shape}')
print(f'Observations: used to project quantum states: {my_obs}')
print(q_c.draw('text'))
print(f'File used for this run: {source_file}')
print(f'Shape of dataset: {env.shape}')
print(f'Number of folds of the CV {nfolds}')
print(f'OBS: {my_obs}')
print(f'Quantum C: {q_c.name}')
print(f'Number of shots: {n_shot_}')

#get time
t_start = time.time()



q_enc = QEncoding(data=X_np, obs=my_obs, qcircuit=q_c, use_pe=True)
q_enc.encode(nshots=n_shot_, shots_seed=123)
env_encoded = q_enc.get_encoding(y_label=y_np)    

#DEFINE design matrix
Y = env_encoded['label']
X = env_encoded.loc[:,0:17]     

grid = GridSearchCV(SVC(), params_grid, verbose=1, cv=nfolds)

#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid_clf = grid.fit(X=X, y=Y)

#get time training
t_training = time.time()
    
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