import sys
import os
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from qke.CircuitContainer import CircuitContainer
from qke.Circuits import Circuits
from qke.QMeasures import QMeasures
from qke.CKernels import CKernels

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
c = CircuitContainer(qtemplate=Circuits.zzfeaturemap, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)


#load dataset with panda
#data are scaled outside the notebook
data_file = 'data/env.sel3.scaled.csv'
#data_file = 'data/env.sel3.minmax.csv' 
env = pd.read_csv(data_file)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')

#get time
t_start = time.time()

svm_quantum = SVC(kernel=CircuitContainer.kernel_matrix).fit(X_train_np, y_train_np);
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

c.save_feature_map(prefix='run_zzfm')

# *** Create a Container *** RUN ON SERVER
# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Using dataset in datafile: data/env.sel3.scaled.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5857740585774058
# Time training: 325.1087284088135 seconds. Final time 433.3743648529053 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240730232653


# *** Create a Container *** RUN ON SERVER
# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: linear
# Using dataset in datafile: data/env.sel3.scaled.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 312.9512767791748 seconds. Final time 417.10990023612976 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240730230925

#RUN WITH MINMAX DATASET
# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX']
# *** Measure procedure: StateVectorEstimator
# Using dataset in datafile: data/env.sel3.minmax.csv
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 361.1748855113983 seconds. Final time 481.3913857936859 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240603175004

# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX']
# *** Measure procedure: StateVectorEstimator
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 353.06747341156006 seconds. Final time 495.4989585876465 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20240529124852

# *** Create a Container ***
# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 356.9466998577118 seconds. Final time 490.4231843948364 seconds
# Sanity check. Dict len after prediction: 2865