
import sys
import os
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from qke.qproc import CircuitContainer
from qke.qproc import Circuits
from qke.qproc import kernel_matrix


#set the seed
np.random.seed(123)



my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

c = CircuitContainer(qtemplate=Circuits.ansatz_encoded, nwire=6, obs=my_obs)


#load dataset with panda
#data are scaled outside the notebook
env = pd.read_csv('data/env.sel3.scaled.csv')  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')


#get time
t_start = time.time()

svm_quantum = SVC(kernel=kernel_matrix).fit(X_train_np, y_train_np);
print(f'Sanity check. Dict len after training: {len(c.fm_dict)}')

#get time training
t_training = time.time()

#result...
predictions = svm_quantum.predict(X_test_np)
score = accuracy_score(predictions, y_test)


#final time (trainign + predict)
t_final = time.time()

print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')
print(f'Sanity check. Dict len after prediction: {len(c.fm_dict)}')

#LAST RESULT: SEL3 - NO DUPLICATED
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───┐┌───────────┐                         ┌───┐ ░ ┌───┐
# q_0: ┤ H ├┤ Rz(phi_0) ├──■──────────────────────┤ X ├─░─┤ H ├
#      ├───┤├───────────┤┌─┴─┐                    └─┬─┘ ░ ├───┤
# q_1: ┤ H ├┤ Rz(phi_1) ├┤ X ├──■───────────────────┼───░─┤ H ├
#      ├───┤├───────────┤└───┘┌─┴─┐                 │   ░ ├───┤
# q_2: ┤ H ├┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼───░─┤ H ├
#      ├───┤├───────────┤     └───┘┌─┴─┐            │   ░ ├───┤
# q_3: ┤ H ├┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼───░─┤ H ├
#      ├───┤├───────────┤          └───┘┌─┴─┐       │   ░ ├───┤
# q_4: ┤ H ├┤ Rz(phi_4) ├───────────────┤ X ├──■────┼───░─┤ H ├
#      ├───┤├───────────┤               └───┘┌─┴─┐  │   ░ ├───┤
# q_5: ┤ H ├┤ Rz(phi_5) ├────────────────────┤ X ├──■───░─┤ H ├
#      └───┘└───────────┘                    └───┘      ░ └───┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7796373779637378
# Time training: 34.19785833358765 seconds. Final time 44.86307430267334 seconds
# Sanity check. Dict len after prediction: 2865


#LAST RESULT: SEL3
# *** Create a Container *** 
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───┐┌───────────┐                         ┌───┐ ░ ┌───┐
# q_0: ┤ H ├┤ Rz(phi_0) ├──■──────────────────────┤ X ├─░─┤ H ├
#      ├───┤├───────────┤┌─┴─┐                    └─┬─┘ ░ ├───┤
# q_1: ┤ H ├┤ Rz(phi_1) ├┤ X ├──■───────────────────┼───░─┤ H ├
#      ├───┤├───────────┤└───┘┌─┴─┐                 │   ░ ├───┤
# q_2: ┤ H ├┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼───░─┤ H ├
#      ├───┤├───────────┤     └───┘┌─┴─┐            │   ░ ├───┤
# q_3: ┤ H ├┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼───░─┤ H ├
#      ├───┤├───────────┤          └───┘┌─┴─┐       │   ░ ├───┤
# q_4: ┤ H ├┤ Rz(phi_4) ├───────────────┤ X ├──■────┼───░─┤ H ├
#      ├───┤├───────────┤               └───┘┌─┴─┐  │   ░ ├───┤
# q_5: ┤ H ├┤ Rz(phi_5) ├────────────────────┤ X ├──■───░─┤ H ├
#      └───┘└───────────┘                    └───┘      ░ └───┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (3233, 7)
# Training shape dataset (2424, 6)
# Label for traing (2424,)
# Test shape dataset (809, 6)
# Label for test (809,)
# USING 2424 data point for training
# Sanity check. Dict len after training: 2214
# *******SCORE: 0.7367119901112484
# Time training: 406.8062620162964 seconds. Final time 541.1335113048553 seconds
# Sanity check. Dict len after prediction: 2865



