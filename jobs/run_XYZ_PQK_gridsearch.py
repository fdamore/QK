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

pqk = PQK_SVC(template=Circuits.xyz_encoded, fit_clear=True, full_ent=False, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

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
#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976
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
params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0,3.5,3.7, 4.0])}



#Create the GridSearchCV object (be carefull... it uses all processors on the host machine if you use n_jopbs = -1)
grid = GridSearchCV(pqk, params_grid, verbose=1, n_jobs=-1)

#get time
t_start = time.time()


#Fit the data with the best possible parameters
grid_clf = grid.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid.best_params_}')

#perform grid prediction on test set
grid_predictions = grid.predict(X_test) 

# print classification report 
print(classification_report(y_test, grid_predictions))

#print scro a comparison
score = accuracy_score(grid_predictions, y_test)
print(f'Accuracy Score on data: {score}')

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
# Fraction rate used for this run: 10.0%
# Shape of dataset: (286, 7)
# Training shape dataset (214, 6)
# Label for traing (214,)
# Test shape dataset (72, 6)
# Label for test (72,)
# Best paramenter: {'C': 0.5, 'gamma': 0.1}

