import datetime
import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.PQK_SVC import PQK_SVC
from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels



#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123


# Best paramenter: {'C': 256, 'gamma': 0.1}
my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
pqk = PQK_SVC(C=256, gamma=0.1, circuit_template=Circuits.xyz_encoded, full_ent=False, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

#my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
#my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']


#print metadata
pqk.metadata()

#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
data_file_csv = 'data/env.sel3.scaled.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  

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
print(f'File used for this run: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')

#get time
t_start = time.time()

svm_quantum = pqk.fit(X_train_np, y_train_np);
print(f'Sanity check. Dict len after training: {len(pqk._fm_dict)}')

#get time training
t_training = time.time()

#result...
predictions = svm_quantum.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')
print(f'Sanity check. Dict len after prediction: {len(pqk._fm_dict)}')

#time simulation info
datetime_object = datetime.datetime.fromtimestamp(t_final)
formatted_datetime = datetime_object.strftime("%Y-%m-%d %H-%M-%S")
print(f'Simulation end at: {formatted_datetime}')

pqk.save_feature_map(prefix='run_xyz_pqk_')

# Best paramenter: {'C': 256, 'gamma': 0.1}
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐┌───────────┐┌───────────┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├
#      ├───────────┤├───────────┤├───────────┤
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├
#      ├───────────┤├───────────┤├───────────┤
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├
#      ├───────────┤├───────────┤├───────────┤
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├
#      ├───────────┤├───────────┤├───────────┤
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├
#      ├───────────┤├───────────┤├───────────┤
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├
#      └───────────┘└───────────┘└───────────┘
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# File used for this run: data/env.sel3.scaled.csv
# Fraction rate used for this run: 100%
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.8744769874476988
# Time training: 373.48385190963745 seconds. Final time 500.45435881614685 seconds
# Sanity check. Dict len after prediction: 2865
# Simulation end at: 2024-10-19 20-29-37
# Timestamp of the file storing data: 20241019202937


## Best paramenter: {'C': 0.5, 'gamma': 0.1} from gird search
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐┌───────────┐┌───────────┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├
#      ├───────────┤├───────────┤├───────────┤
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├
#      ├───────────┤├───────────┤├───────────┤
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├
#      ├───────────┤├───────────┤├───────────┤
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├
#      ├───────────┤├───────────┤├───────────┤
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├
#      ├───────────┤├───────────┤├───────────┤
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├
#      └───────────┘└───────────┘└───────────┘
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# File used for this run: data/env.sel3.scaled.csv
# Fraction rate used for this run: 100%
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.8312412831241283
# Time training: 287.428414106369 seconds. Final time 379.1956877708435 seconds
# Sanity check. Dict len after prediction: 2865
# Simulation end at: 2024-10-15 23-04-22
# Timestamp of the file storing data: 20241015230422

# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐┌───────────┐┌───────────┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├
#      ├───────────┤├───────────┤├───────────┤
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├
#      ├───────────┤├───────────┤├───────────┤
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├
#      ├───────────┤├───────────┤├───────────┤
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├
#      ├───────────┤├───────────┤├───────────┤
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├
#      ├───────────┤├───────────┤├───────────┤
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├
#      └───────────┘└───────────┘└───────────┘
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# File used for this run: data/env.sel3.scaled.csv
# Fraction rate used for this run: 100%
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.8368200836820083
# Time training: 291.133686542511 seconds. Final time 385.3376569747925 seconds
# Sanity check. Dict len after prediction: 2865
# Simulation end at: 2024-10-15 14-57-59
# Timestamp of the file storing data: 20241015145759



# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐┌───────────┐┌───────────┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├
#      ├───────────┤├───────────┤├───────────┤
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├
#      ├───────────┤├───────────┤├───────────┤
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├
#      ├───────────┤├───────────┤├───────────┤
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├
#      ├───────────┤├───────────┤├───────────┤
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├
#      ├───────────┤├───────────┤├───────────┤
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├
#      └───────────┘└───────────┘└───────────┘
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: __kernel_matrix
# File used for this run: data/env.sel3.2pi_minmax.csv
# Fraction rate used for this run: 100%
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.8145048814504882
# Time training: 283.4478943347931 seconds. Final time 374.387814283371 seconds
# Sanity check. Dict len after prediction: 2865
# Simulation end at: 2024-10-15 14-45-30
# Timestamp of the file storing data: 20241015144530


# *** Quantum template for feature map using 6 qubit ***
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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: __kernel_matrix
# File used for this run: data/env.sel3.2pi_minmax.csv
# Fraction rate used for this run: 100%
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.7573221757322176
# Time training: 285.8079137802124 seconds. Final time 378.3320405483246 seconds
# Sanity check. Dict len after prediction: 2865
# Timestamp of the file storing data: 20241015142950

