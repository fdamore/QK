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

from pqk.Circuits import Circuits

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

params_grid = {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.01,0.05,0.75, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.50, 1.75, 2.0, 2.5])}


#take a feature map and setting QSVC
NUM_QBIT = 6 #commonly, this is the number of the features
fm = Circuits.xyz_encoded(n_wire=NUM_QBIT, full_ent=False)   
#fm = ZZFeatureMap(feature_dimension=NUM_QBIT)
q_kernel = FidelityStatevectorKernel(feature_map=fm)
svm_quantum = QSVC(quantum_kernel=q_kernel)


#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
nj = 1
grid = GridSearchCV(svm_quantum, params_grid, verbose=1, n_jobs=nj)

print('***INFO RUN***')
print(f'Name of Feature Map: {fm.name}')
print(fm.draw())
print(f'N job param = {nj}')
print(f'GridSearch Dict: {params_grid}')
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

#final time (trainign + predict)
t_final = time.time()
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')

# ***INFO RUN***
# Name of Feature Map: XYZ
#      ┌───────────┐┌───────────┐┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤├───────────┤├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤├───────────┤├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤├───────────┤├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤├───────────┤├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤├───────────┤├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘└───────────┘└───────────┘                    └───┘     
# N job param = 1
# GridSearch Dict: {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024], 'gamma': array([0.01, 0.05, 0.75, 0.1 , 0.15, 0.25, 0.5 , 0.75, 1.  , 1.5 , 1.75,
#        2.  , 2.5 ])}
# Source file: data/env.sel3.sk_sc.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Fitting 5 folds for each of 156 candidates, totalling 780 fits
# Best paramenter: {'C': 1024, 'gamma': 0.01}
#               precision    recall  f1-score   support

#           -1       0.92      0.84      0.88       410
#            1       0.81      0.91      0.85       307

#     accuracy                           0.87       717
#    macro avg       0.86      0.87      0.86       717
# weighted avg       0.87      0.87      0.87       717

# Accuracy Score on data: 0.8661087866108786
# Time training: 10348.096881628036 seconds. Final time 10352.890522480011 seconds