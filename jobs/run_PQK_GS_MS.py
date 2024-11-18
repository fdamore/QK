import sys
import os
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from pqk.PQK_SVC import PQK_SVC



#data are scaled outside the notebook
f_rate = 0.01 #rate of data sampling fot testing pourpose
#data_file_csv = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.sk_sc.csv'

#clear the cache of quantum latent state at each iteration
clear_cache = True

#try different seeds
l_accuracy = []

#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
nj = 1

#grid search verbosity (the higher, the more messages)
g_verbose=1

for current_seed in range(1,5):

    print(f'Run using seed: {current_seed}')

    np.random.seed(current_seed)
    algorithm_globals.random_seed = current_seed


    my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
    #my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
    #my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
    #my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

    
    pqk = PQK_SVC(circuit_template=Circuits.xyz_encoded, fit_clear=clear_cache, full_ent=False, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

    #print metadata
    pqk.metadata()


    
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=current_seed)  

    #DEFINE design matrix
    Y = env['occupancy']
    X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

    #split design matrix (25% of the design matrix used for test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=current_seed)
    #WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    #check the shape of test and training dataset
    print(f'Source file: {data_file_csv}')
    print(f'Shape of dataset: {env.shape}')
    print(f'Training shape dataset {X_train_np.shape}')
    print(f'Label for traing {y_train_np.shape}')
    print(f'Test shape dataset {X_test_np.shape}')
    print(f'Label for test {y_test_np.shape}')


    # define grid search strategy
    #Create a dictionary of possible parameters
    #params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    #          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0,3.5,3.7, 4.0])}

    #params_grid = {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    #        'gamma': np.array([0.01,0.05,0.75, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.50, 1.75, 2.0, 2.5])}

    params_grid = {'C': [0.5, 1.0], 'gamma': np.array([0.01,0.05])}



    grid = GridSearchCV(pqk, params_grid, verbose=g_verbose, n_jobs=nj)

    print('***INFO RUN***')
    print(f'Clear cache: {clear_cache}')
    print(f'N job param = {nj}')
    print(f'GridSearch Dict: {params_grid}')


    #get time
    t_start = time.time()


    #Fit the data with the best possible parameters
    grid_clf = grid.fit(X_train_np, y_train_np)

    #get time training
    t_training = time.time()

    #Print the best estimator with it's parameters
    print(f'Best paramenter: {grid.best_params_}')

    #perform grid prediction on test set
    grid_predictions = grid.predict(X_test_np)

    # print classification report 
    print(classification_report(y_test, grid_predictions))

    #print scro a comparison
    score = accuracy_score(grid_predictions, y_test)
    print(f'Accuracy Score on data: {score}')

    #add result to the accuracy list
    l_accuracy.append(score)

print('***************************************************')
print('SHOW ACCURACY LIST*********************************')
print(l_accuracy)



