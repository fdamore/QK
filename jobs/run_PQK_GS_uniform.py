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


my_obs = ['XII', 'IXI','IIX', 'YII', 'IYI','IIY','ZII', 'IZI','IIZ']
#my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
#my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

clear_cache = True

qc = Circuits.uniform_bloch_encoding(n_wire=3,full_ent=False)

pqk = PQK_SVC(circuit=qc, fit_clear=clear_cache, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

#print metadata
pqk.metadata()


#load dataset with panda
#data are scaled outside the notebook
f_rate = 0.01 #rate of data sampling fot testing pourpose
data_file_csv = 'data/env.sel3.bw_0_1.csv'
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
grid = GridSearchCV(pqk, params_grid, verbose=1, n_jobs=nj)

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

# *** Quantum template for feature map using 3 qubit ***
#      ┌────────────────────────┐┌─────────────┐
# q_0: ┤ Ry(2*acos(phi_0**0.5)) ├┤ Rz(π*phi_1) ├
#      ├────────────────────────┤├─────────────┤
# q_1: ┤ Ry(2*acos(phi_2**0.5)) ├┤ Rz(π*phi_3) ├
#      ├────────────────────────┤├─────────────┤
# q_2: ┤ Ry(2*acos(phi_4**0.5)) ├┤ Rz(π*phi_5) ├
#      └────────────────────────┘└─────────────┘
# *** Required observables: ['XII', 'IXI', 'IIX', 'YII', 'IYI', 'IIY', 'ZII', 'IZI', 'IIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x76eeb6fb0040>,
#         circuit_template=<function Circuits.uniform_bloch_encoding at 0x76eeb70701f0>,
#         full_ent=False,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x76eeb6f97d90>,
#         nwire=3,
#         obs=['XII', 'IXI', 'IIX', 'YII', 'IYI', 'IIY', 'ZII', 'IZI', 'IIZ'])>
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
# Fitting 5 folds for each of 156 candidates, totalling 780 fits
# Best paramenter: {'C': 256, 'gamma': 0.01}
#               precision    recall  f1-score   support

#           -1       0.95      0.77      0.85       410
#            1       0.76      0.94      0.84       307

#     accuracy                           0.85       717
#    macro avg       0.85      0.86      0.85       717
# weighted avg       0.87      0.85      0.85       717

# Accuracy Score on data: 0.8465829846582985

# *** Quantum template for feature map using 3 qubit ***
#      ┌────────────────────────┐┌─────────────┐
# q_0: ┤ Ry(2*acos(phi_0**0.5)) ├┤ Rz(π*phi_1) ├
#      ├────────────────────────┤├─────────────┤
# q_1: ┤ Ry(2*acos(phi_2**0.5)) ├┤ Rz(π*phi_3) ├
#      ├────────────────────────┤├─────────────┤
# q_2: ┤ Ry(2*acos(phi_4**0.5)) ├┤ Rz(π*phi_5) ├
#      └────────────────────────┘└─────────────┘
# *** Required observables: ['XII', 'IXI', 'IIX', 'YII', 'IYI', 'IIY', 'ZII', 'IZI', 'IIZ']
# *** Measure procedure: StateVectorEstimator
# *** CKernel function used: rbf
# Param: <bound method BaseEstimator.get_params of PQK_SVC(c_kernel=<function CKernels.rbf at 0x726495b21300>,
#         circuit_template=<function Circuits.uniform_bloch_encoding at 0x726495de7ce0>,
#         full_ent=False,
#         measure_fn=<function QMeasures.StateVectorEstimator at 0x726495b07b00>,
#         nwire=3,
#         obs=['XII', 'IXI', 'IIX', 'YII', 'IYI', 'IIY', 'ZII', 'IZI', 'IIZ'])>
# Shape of dataset: (29, 7)
# Training shape dataset (21, 6)
# Label for traing (21,)
# Test shape dataset (8, 6)
# Label for test (8,)
# ***INFO RUN***
# Clear cache: True
# N job param = -1
# GridSearch Dict: {'C': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024], 'gamma': array([0.01, 0.05, 0.75, 0.1 , 0.15, 0.25, 0.5 , 0.75, 1.  , 1.5 , 1.75,
#        2.  , 2.5 ])}
# Fitting 5 folds for each of 156 candidates, totalling 780 fits
# Best paramenter: {'C': 64.0, 'gamma': 0.01}
#               precision    recall  f1-score   support

#           -1       1.00      0.40      0.57         5
#            1       0.50      1.00      0.67         3

#     accuracy                           0.62         8
#    macro avg       0.75      0.70      0.62         8
# weighted avg       0.81      0.62      0.61         8

# Accuracy Score on data: 0.625