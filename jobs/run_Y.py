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
c = CircuitContainer(qtemplate=Circuits.y_encoded, full_ent= True,  nwire=6, obs=my_obs)


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

#c.save_feature_map(prefix='run_y')

#RUN WITH ENT and NO PHASE SCALING
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Ry(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Ry(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Ry(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Ry(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Ry(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Ry(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7866108786610879
# Time training: 33.309821367263794 seconds. Final time 44.152793884277344 seconds
# Sanity check. Dict len after prediction: 2865


# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐
# q_0: ┤ Ry(phi_0) ├
#      ├───────────┤
# q_1: ┤ Ry(phi_1) ├
#      ├───────────┤
# q_2: ┤ Ry(phi_2) ├
#      ├───────────┤
# q_3: ┤ Ry(phi_3) ├
#      ├───────────┤
# q_4: ┤ Ry(phi_4) ├
#      ├───────────┤
# q_5: ┤ Ry(phi_5) ├
#      └───────────┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7796373779637378
# Time training: 35.13290810585022 seconds. Final time 47.46423530578613 seconds
# Sanity check. Dict len after prediction: 2865


#LAST RESULT: SEL3 - NO DUPLICATED
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌─────────────────────┐                         ┌───┐
# q_0: ┤ Ry(π/2*phi_0 + π/2) ├──■──────────────────────┤ X ├
#      ├─────────────────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Ry(π/2*phi_1 + π/2) ├┤ X ├──■───────────────────┼──
#      ├─────────────────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Ry(π/2*phi_2 + π/2) ├─────┤ X ├──■──────────────┼──
#      ├─────────────────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Ry(π/2*phi_3 + π/2) ├──────────┤ X ├──■─────────┼──
#      ├─────────────────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Ry(π/2*phi_4 + π/2) ├───────────────┤ X ├──■────┼──
#      ├─────────────────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Ry(π/2*phi_5 + π/2) ├────────────────────┤ X ├──■──
#      └─────────────────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7280334728033473
# Time training: 33.98324680328369 seconds. Final time 44.50042963027954 seconds
# Sanity check. Dict len after prediction: 2865


# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌─────────────────────┐
# q_0: ┤ Ry(π/2*phi_0 + π/2) ├
#      ├─────────────────────┤
# q_1: ┤ Ry(π/2*phi_1 + π/2) ├
#      ├─────────────────────┤
# q_2: ┤ Ry(π/2*phi_2 + π/2) ├
#      ├─────────────────────┤
# q_3: ┤ Ry(π/2*phi_3 + π/2) ├
#      ├─────────────────────┤
# q_4: ┤ Ry(π/2*phi_4 + π/2) ├
#      ├─────────────────────┤
# q_5: ┤ Ry(π/2*phi_5 + π/2) ├
#      └─────────────────────┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7810320781032078
# Time training: 34.76724123954773 seconds. Final time 46.101622104644775 seconds
# Sanity check. Dict len after prediction: 2865