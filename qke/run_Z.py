#Create a cotainer
from qproc import CircuitContainer
from qproc import Circuits
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qproc import kernel_matrix
import numpy as np


#set the seed
np.random.seed(123)


my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

c = CircuitContainer(qtemplate=Circuits.z_encoded, full_ent=True, nwire=6, obs=my_obs)


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

#c.save_feature_map(prefix='run_z')

# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐                         ┌───┐
# q_0: ┤ Rz(phi_0) ├──■──────────────────────┤ X ├
#      ├───────────┤┌─┴─┐                    └─┬─┘
# q_1: ┤ Rz(phi_1) ├┤ X ├──■───────────────────┼──
#      ├───────────┤└───┘┌─┴─┐                 │  
# q_2: ┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼──
#      ├───────────┤     └───┘┌─┴─┐            │  
# q_3: ┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼──
#      ├───────────┤          └───┘┌─┴─┐       │  
# q_4: ┤ Rz(phi_4) ├───────────────┤ X ├──■────┼──
#      ├───────────┤               └───┘┌─┴─┐  │  
# q_5: ┤ Rz(phi_5) ├────────────────────┤ X ├──■──
#      └───────────┘                    └───┘     
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 35.17373847961426 seconds. Final time 46.213316679000854 seconds
# Sanity check. Dict len after prediction: 2865


# *** Create a Container ***
# *** Created quantum template for feature map using 6 qubit ***
#      ┌───────────┐
# q_0: ┤ Rz(phi_0) ├
#      ├───────────┤
# q_1: ┤ Rz(phi_1) ├
#      ├───────────┤
# q_2: ┤ Rz(phi_2) ├
#      ├───────────┤
# q_3: ┤ Rz(phi_3) ├
#      ├───────────┤
# q_4: ┤ Rz(phi_4) ├
#      ├───────────┤
# q_5: ┤ Rz(phi_5) ├
#      └───────────┘
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Sanity check. Dict len after training: 2148
# *******SCORE: 0.5467224546722455
# Time training: 34.45022487640381 seconds. Final time 44.97129940986633 seconds
# Sanity check. Dict len after prediction: 2865




