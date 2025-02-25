import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.kernels import FidelityStatevectorKernel 
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
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  

#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=2580, test_size=280,random_state=123)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')

print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')

#take a feature map and setting QSVC
NUM_QBIT = X_train.shape[1]
fm = Circuits.xyz_encoded(full_ent=False, n_wire=NUM_QBIT)


#get time
t_start = time.time()

list_score = []

#Best parameter: {'C': 512.0, 'gamma': 1e-07}
# #define hyperparamenter QK_3D_FALSE
C_ = 512
gamma_ = 1e-7

#deinf the number of the shots
n_shots_list = range(150,200,1)#run_1

for n_shot_ in n_shots_list:       

    q_kernel = FidelityStatevectorKernel(feature_map=fm, shots=n_shot_) 
    svm_quantum = QSVC(quantum_kernel=q_kernel)

    svm_quantum.fit(X_train, y_train)

    #result...
    predictions = svm_quantum.predict(X_test)
    score = accuracy_score(predictions, y_test)

    list_score.append(score)


#get time training
t_training = time.time()


#final time (trainign + predict)
t_final = time.time()


print(f'Final time {t_final - t_start} seconds')

#save info.
np.savetxt("nhsots_4.txt", np.array(n_shots_list))
np.savetxt("scores_4.txt", np.array(list_score))
