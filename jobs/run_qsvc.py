import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#sclaled_data_file = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.minmax.csv' 
env = pd.read_csv(data_file_csv)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)


#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')

print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')

#take a feature map and setting QSVC
NUM_QBIT = X_train.shape[1]
fm = ZZFeatureMap(feature_dimension=NUM_QBIT)
q_kernel = FidelityStatevectorKernel(feature_map=fm)
svm_quantum = QSVC(quantum_kernel=q_kernel)

#show feature map
print(f'*** FEATURE MAP used in QSVC')
print(fm.draw())

#get time
training_start = time.time()

svm_quantum.fit(X_train, y_train)

#get time training
training_end = time.time()

#result...
predictions = svm_quantum.predict(X_test)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
jobs_final_time = time.time()

print(f'*******SCORE: {score}')
print(f'Time training: {training_end - training_start} seconds. Final time {jobs_final_time - training_start} seconds')

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
