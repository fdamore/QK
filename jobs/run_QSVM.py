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

from pqk.Circuits import Circuits

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
#data_file_csv = 'data/env.sel3.minmax.csv' 
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123,test_size=0.1) #90% training, 10% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123, test_size=1000) #n for training, n-1 for test


C_value = 8.0


#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')
print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')
print(f'Using C value: {C_value}.')

#take a feature map and setting QSVC
NUM_QBIT = X_train.shape[1]
fm = Circuits.Trotter_HuangE3(n_wire=NUM_QBIT)
q_kernel = FidelityStatevectorKernel(feature_map=fm)
svm_quantum = QSVC(quantum_kernel=q_kernel,C=C_value)

#show feature map
print(f'*** FEATURE MAP used in QSVC')
print('fm.name=',fm.name)
#print(fm.draw())

print(f'Number of QUBITS used: {NUM_QBIT}')

#get time
t_start = time.time()

svm_quantum.fit(X_train, y_train)

#get time training
t_training = time.time()

#result...
predictions = svm_quantum.predict(X_test)

t_prediction = time.time()

score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

#print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds.')
print(f'Time prediction: {t_prediction - t_training} seconds.')
print(f'Final time {t_final - t_start} seconds')

