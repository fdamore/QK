import sys
import os
import time
import pandas as pd
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

pqk = PQK_SVC(template=Circuits.xyz_encoded, full_ent=True, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

#print metadata
pqk.metadata()


#load dataset with panda
#data are scaled outside the notebook
f_rate = 0.1 #rate of data sampling fot testing pourpose (GRID SEARCH RATE)
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

# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0, 4.0])}



#Create the GridSearchCV object
grid_clf = GridSearchCV(pqk, params_grid)

#get time
t_start = time.time()


#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid_clf.best_params_}')




