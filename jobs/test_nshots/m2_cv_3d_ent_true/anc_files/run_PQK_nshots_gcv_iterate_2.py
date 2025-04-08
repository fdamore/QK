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


# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {
    'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
    'gamma': [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
    'kernel':['rbf'] 
    }
nfolds = 10 #set number of folds in CV


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


#deinf the number of the shots
n_shots_list = range(50,100,1)#run_1

#get time
t_start = time.time()

list_score = []

for n_shot_ in n_shots_list:

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



    # taking the largest average accuracy of the grid search and the corresponding standard dev.
    cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
    cv_std = grid.cv_results_['std_test_score'][grid.best_index_]

    list_score.append(cv_mean)

#final time (trainign + predict)
t_final = time.time()


print(f'Final time {t_final - t_start} seconds')

#save info.
np.savetxt("nhsots_2.txt", np.array(n_shots_list))
np.savetxt("scores_2.txt", np.array(list_score))




