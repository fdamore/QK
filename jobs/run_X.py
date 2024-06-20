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
from qke.QMeasures import QMeasures
from qke.qproc import kernel_matrix




#set the seed
np.random.seed(123)


my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
c = CircuitContainer(qtemplate=Circuits.x_encoded, full_ent=True, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator)

#load dataset with panda
#data are scaled outside the notebook
data_file_csv = 'data/env.sel3.minmax.csv'
env = pd.read_csv(data_file_csv)  


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
print(f'File used for this run: {data_file_csv}')
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

c.save_feature_map(prefix='run_x_')

#RUN USING MINMAX
# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# File used for this run: data/env.sel3.minmax.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.6959553695955369
# Time training: 328.95602011680603 seconds. Final time 441.45304918289185 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240531170026

# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7866108786610879
# Time training: 351.012647151947 seconds. Final time 486.23611307144165 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240528172138

# USING STATE VECTOR ESTIMATOR
# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7866108786610879
# Time training: 331.76958894729614 seconds. Final time 456.21892976760864 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240528155815

# 7 observables
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ', 'ZZZZZZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7866108786610879
# Time training: 328.2446565628052 seconds. Final time 442.40824270248413 seconds
# Sanity check. Dict len after prediction: 2865

#RUN WITH ENT
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rx(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rx(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rx(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rx(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rx(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rx(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7866108786610879
# Time training: 33.22173047065735 seconds. Final time 43.69630837440491 seconds
# Sanity check. Dict len after prediction: 2865


#RUN NO ENT
# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐
# q_0: ┤ Rx(phi_0) ├
#      ├───────────┤
# q_1: ┤ Rx(phi_1) ├
#      ├───────────┤
# q_2: ┤ Rx(phi_2) ├
#      ├───────────┤
# q_3: ┤ Rx(phi_3) ├
#      ├───────────┤
# q_4: ┤ Rx(phi_4) ├
#      ├───────────┤
# q_5: ┤ Rx(phi_5) ├
#      └───────────┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7796373779637378
# Time training: 32.69387602806091 seconds. Final time 42.835983753204346 seconds
# Sanity check. Dict len after prediction: 2865