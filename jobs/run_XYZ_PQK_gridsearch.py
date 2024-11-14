import sys
import os
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from pqk.PQK_SVC import PQK_SVC



#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123


my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
#my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
#my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

clear_cache = True
pqk = PQK_SVC(circuit_template=Circuits.xyz_encoded, fit_clear=clear_cache, full_ent=False, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

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
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')


# define grid search strategy
#Create a dictionary of possible parameters
#params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
#          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0,3.5,3.7, 4.0])}

params_grid = {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.01,0.05,0.75, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.50, 1.75, 2.0, 2.5])}



#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
nj = -1
grid = GridSearchCV(pqk, params_grid, verbose=2, n_jobs=nj)

print('***INFO RUN***')
print(f'Clear cache: {clear_cache}')
print(f'N job param = {nj}')
print(f'GridSearch Dict: {params_grid}')


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
# *** Required observables: ['YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x757d6f7b0160>,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x757d6f79feb0>,
#         nwire=6,
#         obs=['YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY'])>
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024, 2048, 3000, 3500], 'gamma': array([1.0e-04, 5.0e-03, 1.0e-02, 5.0e-02, 7.5e-01, 1.0e-01, 1.5e-01,
#        2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.5e+00, 3.0e+00, 3.5e+00])}
# Fitting 5 folds for each of 210 candidates, totalling 1050 fits
# Best paramenter: {'C': 512, 'gamma': 0.0001}
#               precision    recall  f1-score   support

#           -1       0.84      0.68      0.75       410
#            1       0.66      0.83      0.74       307

#     accuracy                           0.74       717
#    macro avg       0.75      0.76      0.74       717
# weighted avg       0.77      0.74      0.75       717

# Accuracy Score on data: 0.7447698744769874

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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x7ef89e7c4160>,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x7ef89e7afeb0>,
#         nwire=6,
#         obs=['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX'])>
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024, 2048, 3000, 3500], 'gamma': array([1.0e-04, 5.0e-03, 1.0e-02, 5.0e-02, 7.5e-01, 1.0e-01, 1.5e-01,
#        2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.5e+00, 3.0e+00, 3.5e+00])}
# Fitting 5 folds for each of 210 candidates, totalling 1050 fits
# Best paramenter: {'C': 3500, 'gamma': 0.0001}
#               precision    recall  f1-score   support

#           -1       0.91      0.78      0.84       410
#            1       0.75      0.89      0.81       307

#     accuracy                           0.83       717
#    macro avg       0.83      0.83      0.83       717
# weighted avg       0.84      0.83      0.83       717

# Accuracy Score on data: 0.8256624825662483

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
# *** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x74357abb0160>, full_ent=False,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x74357ab9beb0>,
#         nwire=6,
#         obs=['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ'])>
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024, 2048, 3000, 3500], 'gamma': array([1.0e-04, 5.0e-03, 1.0e-02, 5.0e-02, 7.5e-01, 1.0e-01, 1.5e-01,
#        2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.5e+00, 3.0e+00, 3.5e+00])}
# Fitting 5 folds for each of 210 candidates, totalling 1050 fits
# Best paramenter: {'C': 3500, 'gamma': 0.0001}
#               precision    recall  f1-score   support

#           -1       0.94      0.76      0.84       410
#            1       0.75      0.94      0.83       307

#     accuracy                           0.84       717
#    macro avg       0.85      0.85      0.84       717
# weighted avg       0.86      0.84      0.84       717

# Accuracy Score on data: 0.8382147838214784

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
# *** Required observables: ['YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x7b28277b8160>, full_ent=False,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x7b28277a3eb0>,
#         nwire=6,
#         obs=['YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY'])>
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024, 2048, 3000, 3500], 'gamma': array([1.0e-04, 5.0e-03, 1.0e-02, 5.0e-02, 7.5e-01, 1.0e-01, 1.5e-01,
#        2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.5e+00, 3.0e+00, 3.5e+00])}
# Fitting 5 folds for each of 210 candidates, totalling 1050 fits
# Best paramenter: {'C': 3500, 'gamma': 0.0001}
#               precision    recall  f1-score   support

#           -1       0.93      0.74      0.82       410
#            1       0.73      0.92      0.81       307

#     accuracy                           0.82       717
#    macro avg       0.83      0.83      0.82       717
# weighted avg       0.84      0.82      0.82       717

# Accuracy Score on data: 0.8186889818688982



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
# *** Required observables: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x79a9961c4160>, full_ent=False,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x79a9961afeb0>,
#         nwire=6,
#         obs=['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX'])>
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024, 2048, 3000, 3500], 'gamma': array([1.0e-04, 5.0e-03, 1.0e-02, 5.0e-02, 7.5e-01, 1.0e-01, 1.5e-01,
#        2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.5e+00, 3.0e+00, 3.5e+00])}
# Fitting 5 folds for each of 210 candidates, totalling 1050 fits
# Best paramenter: {'C': 3500, 'gamma': 0.0001}
#               precision    recall  f1-score   support

#           -1       0.93      0.80      0.86       410
#            1       0.77      0.93      0.84       307

#     accuracy                           0.85       717
#    macro avg       0.85      0.86      0.85       717
# weighted avg       0.87      0.85      0.85       717

# Accuracy Score on data: 0.8521617852161785


#RESULT 2024-10-21. ERROR ON BEST ESTIMATOR
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
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024], 'gamma': array([0.01, 0.05, 0.75, 0.1 , 0.15, 0.25, 0.5 , 0.75, 1.  , 1.5 , 1.75,
#        2.  , 2.5 ])}
# Best paramenter: {'C': 256, 'gamma': 0.01}


#RESULT 2024-10-19....  Erro on get info about accuracy score and prediction 
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
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# Fitting 5 folds for each of 270 candidates, totalling 1350 fits
# Best paramenter: {'C': 256, 'gamma': 0.1}
# Traceback (most recent call last):
#   File "/home/damore/git/QK/jobs/run_XYZ_PQK_gridsearch.py", line 90, in <module>
#     grid_predictions = grid.predict(X_test) 
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/sklearn/model_selection/_search.py", line 546, in predict
#     return self.best_estimator_.predict(X)
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/sklearn/svm/_base.py", line 814, in predict
#     y = super().predict(X)
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/sklearn/svm/_base.py", line 431, in predict
#     return predict(X)
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/sklearn/svm/_base.py", line 434, in _dense_predict
#     X = self._compute_kernel(X)
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/sklearn/svm/_base.py", line 508, in _compute_kernel
#     kernel = self.kernel(X, self.__Xfit)
#   File "/home/damore/git/QK/pqk/PQK_SVC.py", line 118, in _kernel_matrix
#     return np.array([[self._qfKernel(a, b) for b in B] for a in A])
#   File "/home/damore/git/QK/pqk/PQK_SVC.py", line 118, in <listcomp>
#     return np.array([[self._qfKernel(a, b) for b in B] for a in A])
#   File "/home/damore/git/QK/pqk/PQK_SVC.py", line 118, in <listcomp>
#     return np.array([[self._qfKernel(a, b) for b in B] for a in A])
#   File "/home/damore/git/QK/pqk/PQK_SVC.py", line 94, in _qfKernel
#     x1_qc = self._qEncoding(qc_template, x1)
#   File "/home/damore/git/QK/pqk/PQK_SVC.py", line 75, in _qEncoding
#     qc_assigned = qc.assign_parameters(data, inplace = False)
#   File "/home/damore/git/QK/.venv/lib/python3.10/site-packages/qiskit/circuit/quantumcircuit.py", line 2705, in assign_parameters
#     raise ValueError(
# ValueError: Mismatching number of values and parameters. For partial binding please pass a dictionary of {parameter: value} pairs.



#GRID SEARCH HIGH VERBOSITY
# [CV 4/5] END .................C=0.25, gamma=4.0;, score=0.767 total time=   2.0s
# [CV 5/5] END .................C=0.25, gamma=4.0;, score=0.738 total time=   2.0s
# [CV 1/5] END ..................C=0.5, gamma=0.1;, score=0.767 total time=   2.0s
# [CV 2/5] END ..................C=0.5, gamma=0.1;, score=0.767 total time=   2.0s
# [CV 3/5] END ..................C=0.5, gamma=0.1;, score=0.814 total time=   2.0s
# [CV 4/5] END ..................C=0.5, gamma=0.1;, score=0.814 total time=   2.0s
# [CV 5/5] END ..................C=0.5, gamma=0.1;, score=0.786 total time=   2.1s
# [CV 1/5] END .................C=0.5, gamma=0.15;, score=0.767 total time=   2.0s
# [CV 2/5] END .................C=0.5, gamma=0.15;, score=0.767 total time=   2.0s
# [CV 3/5] END .................C=0.5, gamma=0.15;, score=0.814 total time=   2.0s
# [CV 4/5] END .................C=0.5, gamma=0.15;, score=0.814 total time=   2.0s
# [CV 5/5] END .................C=0.5, gamma=0.15;, score=0.786 total time=   2.1s

#CUT

# [CV 2/5] END ................C=128.0, gamma=3.0;, score=0.698 total time=   2.0s
# [CV 3/5] END ................C=128.0, gamma=3.0;, score=0.791 total time=   2.0s
# [CV 4/5] END ................C=128.0, gamma=3.0;, score=0.744 total time=   2.0s
# [CV 5/5] END ................C=128.0, gamma=3.0;, score=0.762 total time=   2.0s
# [CV 1/5] END ................C=128.0, gamma=4.0;, score=0.698 total time=   2.0s
# [CV 2/5] END ................C=128.0, gamma=4.0;, score=0.698 total time=   2.0s
# [CV 3/5] END ................C=128.0, gamma=4.0;, score=0.791 total time=   2.0s
# [CV 4/5] END ................C=128.0, gamma=4.0;, score=0.744 total time=   2.0s
# [CV 5/5] END ................C=128.0, gamma=4.0;, score=0.762 total time=   2.0s
# [CV 1/5] END ..................C=256, gamma=0.1;, score=0.721 total time=   2.0s
# [CV 2/5] END ..................C=256, gamma=0.1;, score=0.651 total time=   2.0s

# Best paramenter: {'C': 0.5, 'gamma': 0.1}



#GRID SEARCH 2024-10-15 #1
# *** Quantum template for feature map using 6 qubit ***
#      ┌───────────┐┌───────────┐┌───────────┐
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├-1
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
# Fraction rate used for this run: 10.0%
# Shape of dataset: (286, 7)
# Training shape dataset (214, 6)
# Label for traing (214,)
# Test shape dataset (72, 6)
# Label for test (72,)
# Best paramenter: {'C': 0.5, 'gamma': 0.1}

