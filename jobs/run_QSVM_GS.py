import sys
import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

from qiskit_machine_learning.kernels import FidelityStatevectorKernel 
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_algorithms.utils import algorithm_globals

import numpy as np

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
#data_file_csv = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)   


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
#in this context this casting is for script alignment only.
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

# define grid search strategy
#Create a dictionary of possible parameters
#params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
#          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0,3.5,3.7, 4.0])}

params_grid = {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.01,0.05,0.75, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.50, 1.75, 2.0, 2.5])}


#take a feature map and setting QSVC
NUM_QBIT = 6 #commonly, this is the number of the features
fm = ZZFeatureMap(feature_dimension=NUM_QBIT)
q_kernel = FidelityStatevectorKernel(feature_map=fm)
svm_quantum = QSVC(quantum_kernel=q_kernel)

#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
nj = -1
grid = GridSearchCV(svm_quantum, params_grid, verbose=1, n_jobs=nj)

print('***INFO RUN***')
print(f'N job param = {nj}')
print(f'GridSearch Dict: {params_grid}')
#check the shape of test and training dataset
print(f'Source file: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')



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

# #RUN USING MINMAX
# Using dataset in datafile: data/env.sel3.minmax.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# *** FEATURE MAP used in QSVC
#      ┌──────────────────────────────────────────────┐
# q_0: ┤0                                             ├
#      │                                              │
# q_1: ┤1                                             ├
#      │                                              │
# q_2: ┤2                                             ├
#      │  ZZFeatureMap(x[0],x[1],x[2],x[3],x[4],x[5]) │
# q_3: ┤3                                             ├
#      │                                              │
# q_4: ┤4                                             ├
#      │                                              │
# q_5: ┤5                                             ├
#      └──────────────────────────────────────────────┘
# *******SCORE: 0.8382147838214784
# Time training: 18.973737955093384 seconds. Final time 30.72221088409424 seconds

#run
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# *** FEATURE MAP used in QSVC
#      ┌──────────────────────────────────────────────┐
# q_0: ┤0                                             ├
#      │                                              │
# q_1: ┤1                                             ├
#      │                                              │
# q_2: ┤2                                             ├
#      │  ZZFeatureMap(x[0],x[1],x[2],x[3],x[4],x[5]) │
# q_3: ┤3                                             ├
#      │                                              │
# q_4: ┤4                                             ├
#      │                                              │
# q_5: ┤5                                             ├
#      └──────────────────────────────────────────────┘
# *******SCORE: 0.8172942817294282
# Time training: 19.143609523773193 seconds. Final time 30.912684202194214 seconds
