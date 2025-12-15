#################################################################
# IL PROBLEMA CON QUESTO SCRIPT E' CHE GRIDSEARCHCV NON E' BEN DEFINITO PER IL NOSTRO OCPQK_SVC
#################################################################


import sys
import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from datetime import datetime
from sklearn.metrics import make_scorer, accuracy_score, f1_score

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from pqk.PQK_SVC import PQK_OCSVC
from pqk.aux_funcs import *

seed=123              # the seed
np.random.seed(seed)
algorithm_globals.random_seed = seed

#circuit paramenters
full_ent=False
encoding_key = 'xyz'
my_obs_key = 'XYZ'
measure_fn_key = 'CPU'
#cv parameters
nfolds = 3 #set number of folds in CV
f_rate = .05 #rate of data sampling fot testing pourpose
nj = 1     # number of processors on the host machine. CAREFUL: it uses ALL PROCESSORS if n_jopbs = -1

encoding_dict = {
    'xyz': Circuits.xyz_encoded(full_ent=full_ent, n_wire=6),   # change to 3d ? 
    'zz': Circuits.zzfeaturemap(full_ent=full_ent, n_wire=6), 
    'x': Circuits.x_encoded(full_ent=full_ent, n_wire=6), 
    'spiral': Circuits.spiral_encoding(full_ent=full_ent, n_wire=6, n_windings=1),
    'uniform': Circuits.uniform_bloch_encoding(full_ent=full_ent, n_wire=6),
    'corrxyz': Circuits.corr3_encoded(n_wire=6),
    'anticorrxyz': Circuits.anticorr3_encoded(n_wire=6),
    'IQP': Circuits.IQP_HuangE2(n_wire=6),
    'Trotter': Circuits.Trotter_HuangE3(n_wire=6),
    #'q_data': Circuits.quantum_data_encoding(n_wire=2),
    'q_porco': Circuits.quantum_data_encoding_porco(n_wire=2),
    }   
 
pauli_meas_dict = {
    'XYZ' : generate_my_obs(['X','Y','Z'], n_qub=6),
    'XY' : generate_my_obs(['X','Y'], n_qub=6),
    'X' : generate_my_obs(['X'], n_qub=6),
    'Y' : generate_my_obs(['Y'], n_qub=6),
    'Z' : generate_my_obs(['Z'], n_qub=6),
    'BLOCH_XYZ' : generate_my_obs(['X','Y','Z'], n_qub=3), #for the uniform bloch:   (couldnt make it work yet - Luca)
    'NON_LOCAL_XX' : generate_my_obs(['X','Y','Z','XX'], n_qub=6),
    'ADJAC_2QUB' : adjacent_qub_obs(['X','Y','Z'], n_qub=6, n_measured_qub=2),
    'ADJAC_XX' : adjacent_qub_obs(['X'], n_qub=6, n_measured_qub=2),
    'ADJAC_YY' : adjacent_qub_obs(['Y'], n_qub=6, n_measured_qub=2),
    'ADJAC_ZZ' : adjacent_qub_obs(['Z'], n_qub=6, n_measured_qub=2),
}
pauli_meas_dict['ADJAC_2QUB_EXTRA'] = pauli_meas_dict['XYZ'] + pauli_meas_dict['ADJAC_2QUB']

measure_fn_dict = {
    'CPU' : QMeasures.StateVectorEstimator,
    'GPU' : QMeasures.GPUAerStateVectorEstimator,
    'GPUfakenoise' : QMeasures.GPUAerVigoNoiseStateVectorEstimator,
    #'QPU' : ???
}

clear_cache = False
my_obs = pauli_meas_dict[my_obs_key]



# defining a unique label for the simulation 
id_string = f'_PQK_{measure_fn_key}_{encoding_key}_ent{full_ent}_{len(my_obs)}obs{my_obs_key}_{nfolds}folds_seed{seed}_frate{f_rate}'


#load dataset with panda
#data are scaled outside the notebook
#data_file_csv = 'data/env.sel3.scaled.csv'

if encoding_key == 'uniform':
    data_file_csv = 'data/env.sel3.2pi_minmax.csv'
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)  
elif encoding_key == 'q_data' or encoding_key == 'q_porco':
    data_file_csv = 'data/quantum_states_dataset.csv'
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

scorer = make_scorer(f1_score, pos_label=-1)

# params_grid = {'nu': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5], 
#                'gamma': [0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0]}
params_grid = {'nu': [0.1], 
               'gamma': [0.02]} 

ocpqk = PQK_OCSVC(circuit=encoding_dict[encoding_key], fit_clear=clear_cache, obs=my_obs, measure_fn=measure_fn_dict[measure_fn_key], c_kernel='rbf')
#print metadata
ocpqk.metadata()
 
#run ocpqk to pre compute the enconding
ocpqk.fit(X_train_np)

ocpqk_g = PQK_OCSVC(_fm_dict=ocpqk._fm_dict,
                 circuit=encoding_dict[encoding_key], fit_clear=clear_cache, obs=my_obs, measure_fn=measure_fn_dict[measure_fn_key], c_kernel='rbf')



#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
grid = GridSearchCV(ocpqk_g, params_grid, verbose=1, n_jobs=nj, scoring=scorer, cv=nfolds )


print('***INFO RUN***')
print(f'Clear cache: {clear_cache}')
print(f'N job param = {nj}')
print(f'GridSearch Dict: {params_grid}')
#check the shape of test and training dataset
print(f'Source file: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Shape of training dataset {X_train_np.shape}')
print(f'Shape of training labels {y_train_np.shape}')
print(f'Seed: {seed}')
print(f'Entangling layer={full_ent}')

#get time
t_start = time.time()

#Fit the data with the best possible parameters
grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid.best_params_}')
print(f'Best F1 score {grid.best_score_}')
print(f'Results: {grid.cv_results_.keys()}')

# taking the largest average accuracy of the grid search and the corresponding standard dev.
results = grid.cv_results_
cv_mean = results['mean_test_score'][grid.best_index_]
cv_std = results['std_test_score'][grid.best_index_]

for i in range(nfolds):
    print(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}")
print(f'\nAverage accuracy, best score: {cv_mean:.6f}')
print(f'Standard deviation, best score: {cv_std:.6f}')
print(f'{t_training-t_start} seconds elapsed.')

# the confidence interval is given by:   mean +/- 2 * stdev / sqrt(N)
final_msg = f'Accuracy (95% confidence) = {cv_mean:.6f} +/- {2*cv_std/np.sqrt(nfolds):.6f} == [{cv_mean - 2*cv_std/np.sqrt(nfolds):.6f}, {cv_mean + 2*cv_std/np.sqrt(nfolds):.6f}]'
print(final_msg)

# INFORMATION SAVED IN THE 'accuracy*.txt' OUTPUT FILES

# ATTENZIONE: CAMBIARE IL NOME DEL FILE, ALTRIMENTI SOVRASCRIVE I RISULTATI DI PQK_SVC
# with open(f'jobs/scores/accuracy' + id_string + '.txt', "w+") as file:
#     file.write(final_msg + '\n\n')
#     file.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
#     file.write(f'{t_training-t_start:.1f} seconds elapsed.\n')
#     file.write(f'Feature map: {encoding_dict[encoding_key].name}\n')
#     file.write(f'Entangling layer: {full_ent}\n')    
#     file.write(f'Required observables: {ocpqk_g.obs}\n')
#     file.write(f'Measure procedure: {ocpqk_g.measure_fn.__name__}\n')
#     file.write(f'CKernel function used: {ocpqk_g.c_kernel}\n')
#     file.write(f'Best parameter: {grid.best_params_}\n')
#     file.write(f'Clear cache: {clear_cache}\n')
#     file.write(f'N job param = {nj}\n')
#     file.write(f'GridSearch Dict: {params_grid}\n')
#     #check the shape of test and training dataset
#     file.write(f'Source file: {data_file_csv}\n')
#     file.write(f'Shape of dataset: {env.shape}\n')
#     file.write(f'Shape of training dataset {X_train_np.shape}\n')
#     file.write(f'Shape of training labels {y_train_np.shape}\n')
#     file.write(f'Seed: {seed}\n')
#     file.write(f'Fitting {nfolds} folds for each of {len(ParameterGrid(grid.param_grid))} candidates\n')
#     for i in range(nfolds):
#         file.write(f"Fold {i+1}: {results[f'split{i}_test_score'][grid.best_index_]}\n")


